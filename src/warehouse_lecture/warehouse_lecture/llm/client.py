"""Claude (Anthropic) client wrapper: config-driven models, retries, structured output, usage log.

Every lecture example talks to the LLM through this module so that model names
and cost bookkeeping live in one place (config/llm.yaml + .env). The public
methods keep the OpenAI-era signatures (chat / parse / tools / vision) so the
callers did not change; OpenAI-style messages are converted here.
"""

import base64
from dataclasses import dataclass
import datetime as dt
import io
import json
import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def _find_workspace():
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "warehouse_lecture").is_dir():
            return candidate
    return here


def load_config(path=None):
    file = Path(path) if path else Path(get_package_share_directory("warehouse_lecture")) / "config/llm.yaml"
    return yaml.safe_load(file.read_text())


def load_env():
    """Load ANTHROPIC_API_KEY from the workspace .env without overriding a real env var."""
    from dotenv import load_dotenv

    for candidate in (_find_workspace() / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; copy .env.example to .env and fill in the key")


def split_system(messages):
    """OpenAI puts system prompts inside messages; Claude takes them as a separate ``system`` argument."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    purpose: str


class LLM:
    def __init__(self, config=None, *, usage_log=None):
        import anthropic

        load_env()
        self.config = config or load_config()
        d = self.config["defaults"]
        self.client = anthropic.Anthropic(timeout=d.get("timeout_s", 60), max_retries=d.get("max_retries", 2))
        self.models = self.config["models"]
        self.pricing = self.config.get("pricing", {})
        self.log_path = Path(usage_log or _find_workspace() / self.config.get("usage_log", "logs/llm_usage.jsonl"))
        self.session_cost = 0.0

    # --- bookkeeping ----------------------------------------------------
    def _record(self, model, usage, purpose):
        if usage is None:
            return None
        inp, out = usage.input_tokens or 0, usage.output_tokens or 0      # output includes thinking tokens
        price = self.pricing.get(model, {"input": 0.0, "output": 0.0})
        cost = inp / 1e6 * price["input"] + out / 1e6 * price["output"]
        self.session_cost += cost
        entry = Usage(model, inp, out, cost, purpose)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), **entry.__dict__}) + "\n")
        return entry

    def _request(self, model, messages, max_tokens):
        """Arguments shared by every call. Current Claude models reject temperature; effort sets how much they think."""
        d = self.config["defaults"]
        system, rest = split_system(messages)
        args = dict(model=model, messages=rest, max_tokens=max_tokens or d["max_output_tokens"],
                    output_config={"effort": d.get("effort", "low")})
        if system:
            args["system"] = system
        return args

    @staticmethod
    def _check(response):
        if response.stop_reason == "refusal":
            raise ValueError(f"Model refused: {getattr(response.stop_details, 'explanation', None)}")
        if response.stop_reason == "max_tokens":
            raise ValueError("Model output hit max_tokens; raise defaults.max_output_tokens in config/llm.yaml")

    # --- text -----------------------------------------------------------
    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, purpose="chat", **kwargs):
        """Plain text completion. messages: [{"role": ..., "content": ...}, ...].

        ``temperature`` and ``max_tokens`` are accepted for the old callers but not sent: current Claude
        models reject temperature, and they think before answering, so a small token cap cuts the answer off.
        """
        model = model or self.models["chat"]
        response = self.client.messages.create(**self._request(model, messages, None), **kwargs)
        self._record(model, response.usage, purpose)
        self._check(response)
        return "".join(b.text for b in response.content if b.type == "text")

    def parse(self, messages, schema, *, model=None, temperature=None, purpose="parse"):
        """Structured output: returns an instance of the pydantic ``schema``."""
        model = model or self.models["chat"]
        response = self.client.messages.parse(**self._request(model, messages, None), output_format=schema)
        self._record(model, response.usage, purpose)
        self._check(response)
        return response.parsed_output

    def tools(self, messages, tools, *, model=None, temperature=None, purpose="tools", tool_choice="auto"):
        """One tool-calling round. Returns (assistant_content, [(name, args_dict, call_id), ...])."""
        model = model or self.models["chat"]
        response = self.client.messages.create(**self._request(model, messages, None), tools=tools,
                                               tool_choice={"type": tool_choice})
        self._record(model, response.usage, purpose)
        self._check(response)
        calls = [(b.name, b.input, b.id) for b in response.content if b.type == "tool_use"]
        return response.content, calls

    # --- vision ---------------------------------------------------------
    @staticmethod
    def image_content(rgb, *, detail="low", fmt="JPEG", quality=85):
        """Encode an RGB uint8 array as a Claude image block (``detail`` is an OpenAI option, ignored)."""
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format=fmt, quality=quality)
        data = base64.b64encode(buffer.getvalue()).decode()
        mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}

    def describe_image(self, rgb, prompt, *, model=None, purpose="vision", detail="low"):
        model = model or self.models["vision"]
        messages = [{"role": "user", "content": [self.image_content(rgb), {"type": "text", "text": prompt}]}]
        return self.chat(messages, model=model, purpose=purpose)

    def parse_image(self, rgb, prompt, schema, *, model=None, purpose="vision_parse", detail="low"):
        model = model or self.models["vision"]
        messages = [{"role": "user", "content": [self.image_content(rgb), {"type": "text", "text": prompt}]}]
        return self.parse(messages, schema, model=model, purpose=purpose)


def tool_schema(name, description, parameters, required=None):
    """Helper to build one Claude tool definition from a JSON-schema properties dict."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": parameters,
            "required": required or list(parameters),
            "additionalProperties": False,
        },
    }
