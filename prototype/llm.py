"""Minimal Ollama client with schema-constrained JSON output (runs fully on-premise)."""
from __future__ import annotations

import base64
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

if sys.platform == "win32":
    # Keep the machine awake while this process runs (Modern Standby throttled earlier runs ~50x).
    # Released automatically when the process exits; the display may still turn off.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

OLLAMA = "http://localhost:11434"
TEXT_MODEL = "qwen3:8b"
VISION_MODEL = "qwen2.5vl:7b"
SEED: int | None = 42  # fixed sampling seed for repeatable runs; None leaves it to Ollama
KEEP_ALIVE: int | None = None  # 0 unloads the model after every call (each call starts cold); None = Ollama default

T = TypeVar("T", bound=BaseModel)


def chat(messages: list[dict], model: str = TEXT_MODEL, schema: dict | None = None,
         tools: list[dict] | None = None, num_ctx: int = 12288, temperature: float = 0.0,
         think: bool = False, num_predict: int = 4096) -> dict:
    # num_predict + repeat_penalty stop runaway list generation (seen on dense mud-log pages)
    body = {"model": model, "messages": messages, "stream": False, "think": think,
            "options": {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict,
                        "repeat_penalty": 1.1}}
    if SEED is not None:
        body["options"]["seed"] = SEED
    if KEEP_ALIVE is not None:
        body["keep_alive"] = KEEP_ALIVE
    if schema:
        body["format"] = schema
    if tools:
        body["tools"] = tools
    if model == VISION_MODEL:
        body.pop("think")  # not a reasoning model
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=900)
    r.raise_for_status()
    return r.json()


def extract(model_cls: type[T], system: str, user: str, images: list[str] | None = None,
            model: str = TEXT_MODEL, retries: int = 2, **kw) -> tuple[T | None, dict]:
    """Ask the LLM to fill `model_cls`; validate with pydantic and retry with the error message."""
    msg = {"role": "user", "content": user}
    if images:
        msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
    messages = [{"role": "system", "content": system}, msg]
    meta = {"model": model, "seed": SEED, "attempts": 0, "seconds": 0.0}
    for _ in range(retries + 1):
        meta["attempts"] += 1
        t0 = time.time()
        try:
            resp = chat(messages, model=model, schema=model_cls.model_json_schema(), **kw)
        except requests.RequestException as e:  # transient server errors: count as a failed attempt
            meta["error"] = str(e)[:300]
            meta["seconds"] += time.time() - t0
            continue
        meta["seconds"] += time.time() - t0
        meta["prompt_tokens"] = resp.get("prompt_eval_count")
        meta["output_tokens"] = resp.get("eval_count")
        content = resp["message"]["content"]
        try:
            return model_cls.model_validate(json.loads(content)), meta
        except (ValidationError, json.JSONDecodeError) as e:
            meta["error"] = str(e)[:500]
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"The JSON was invalid: {str(e)[:800]}\nReturn corrected JSON only."}]
    return None, meta
