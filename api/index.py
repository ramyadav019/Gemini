"""Vercel entrypoint — Flask WSGI wrapper around gemini_web2api."""

import json
import sys
import os

# Ensure repo root is on the path so gemini_web2api package is importable
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from flask import Flask, request, Response, stream_with_context
from gemini_web2api.config import CONFIG, load_config
from gemini_web2api.models import MODELS, resolve_model
from gemini_web2api.gemini import generate, generate_stream
from gemini_web2api.tools import (
    messages_to_prompt,
    parse_tool_calls,
    google_contents_to_prompt,
    parse_google_function_calls,
)
from gemini_web2api.multimodal import detect_image_mime, fetch_image_bytes, upload_image
from gemini_web2api.server import _upload_images
from gemini_web2api import __version__
import time, uuid, re

# Load config from environment variables (Vercel env vars)
for key in ("cookie", "xsrf_token", "auth_user", "gemini_bl", "proxy"):
    val = os.environ.get(f"GEMINI_{key.upper()}")
    if val:
        CONFIG[key] = val

api_keys_env = os.environ.get("API_KEYS")
if api_keys_env:
    CONFIG["api_keys"] = [k.strip() for k in api_keys_env.split(",") if k.strip()]

app = Flask(__name__)


def _authorized():
    keys = CONFIG.get("api_keys") or []
    if not keys:
        return True
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] in keys:
        return True
    for h in ("x-api-key", "x-goog-api-key"):
        if request.headers.get(h, "") in keys:
            return True
    if request.args.get("key") in keys:
        return True
    return False


def _unauth():
    return Response(
        json.dumps({"error": {"message": "invalid api key"}}),
        401,
        content_type="application/json",
    )


# ── GET endpoints ─────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return {"status": "ok", "version": __version__, "models": list(MODELS.keys())}


@app.route("/v1/models")
def list_models():
    if not _authorized():
        return _unauth()
    return {
        "object": "list",
        "data": [
            {
                "id": n,
                "object": "model",
                "created": 1700000000,
                "owned_by": "google",
                "description": c["desc"],
            }
            for n, c in MODELS.items()
        ],
    }


@app.route("/v1beta/models")
def list_models_google():
    if not _authorized():
        return _unauth()
    return {
        "models": [
            {
                "name": f"models/{n}",
                "displayName": n,
                "description": c["desc"],
                "supportedGenerationMethods": [
                    "generateContent",
                    "streamGenerateContent",
                ],
            }
            for n, c in MODELS.items()
        ]
    }


# ── POST /v1/chat/completions ─────────────────────────────────────────────────


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if not _authorized():
        return _unauth()
    req = request.get_json(force=True, silent=True)
    if not req:
        return {"error": {"message": "invalid JSON"}}, 400

    model_name, model_id, think_mode, err, extra_fields = resolve_model(
        req.get("model", CONFIG["default_model"])
    )
    if err:
        return {"error": {"message": err}}, 400

    tools = req.get("tools")
    tool_choice = req.get("tool_choice", "auto")
    prompt, images = messages_to_prompt(req.get("messages", []), tools, tool_choice)
    if not prompt.strip():
        return {"error": {"message": "empty prompt"}}, 400

    stream = req.get("stream", False)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    try:
        file_refs = _upload_images(images)
    except RuntimeError as e:
        return {"error": {"message": f"upstream error: {e}"}}, 502

    if stream and (not tools or tool_choice == "none"):

        def generate_sse():
            first = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(first)}\n\n"
            for delta in generate_stream(
                prompt, model_id, think_mode, file_refs, extra_fields
            ):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"content": delta}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            end = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n"

        return Response(
            stream_with_context(generate_sse()), content_type="text/event-stream"
        )

    try:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
    except Exception as e:
        return {"error": {"message": f"upstream error: {e}"}}, 502

    tool_calls = None
    if tools and text and tool_choice != "none":
        text, tool_calls = parse_tool_calls(text)
    msg = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    finish = "tool_calls" if tool_calls else "stop"

    if stream:
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": msg, "finish_reason": finish}],
        }

        def _single():
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"

        return Response(
            stream_with_context(_single()), content_type="text/event-stream"
        )

    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(text or "") // 4,
            "total_tokens": (len(prompt) + len(text or "")) // 4,
        },
    }


# ── POST /v1beta/models/{model}:generateContent ───────────────────────────────


@app.route("/v1beta/models/<path:model_path>", methods=["POST"])
def google_generate(model_path):
    if not _authorized():
        return _unauth()
    req = request.get_json(force=True, silent=True)
    if not req:
        return {"error": {"message": "invalid JSON"}}, 400

    model_raw = model_path.split(":")[0]
    stream = model_path.endswith(":streamGenerateContent")
    model_name, model_id, think_mode, err, extra_fields = resolve_model(model_raw)
    if err:
        return {"error": {"message": err}}, 400

    tool_config = req.get("toolConfig", {})
    fc_mode = tool_config.get("functionCallingConfig", {}).get("mode", "AUTO")
    has_tools = bool(req.get("tools")) and fc_mode != "NONE"
    prompt, images = google_contents_to_prompt(req)
    if not prompt.strip():
        return {"error": {"message": "empty content"}}, 400

    try:
        file_refs = _upload_images(images)
    except RuntimeError as e:
        return {"error": {"message": f"upstream error: {e}"}}, 502

    if stream and not has_tools:

        def generate_sse():
            full_text = ""
            for delta in generate_stream(
                prompt, model_id, think_mode, file_refs, extra_fields
            ):
                if not delta:
                    continue
                full_text += delta
                chunk_obj = {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": delta}], "role": "model"},
                            "index": 0,
                        }
                    ],
                    "modelVersion": model_name,
                }
                yield f"data: {json.dumps(chunk_obj, ensure_ascii=False)}\n\n"
            final = {
                "candidates": [{"finishReason": "STOP", "index": 0}],
                "usageMetadata": {
                    "promptTokenCount": len(prompt) // 4,
                    "candidatesTokenCount": len(full_text) // 4,
                    "totalTokenCount": (len(prompt) + len(full_text)) // 4,
                },
                "modelVersion": model_name,
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(generate_sse()), content_type="text/event-stream"
        )

    try:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
    except Exception as e:
        return {"error": {"message": f"upstream error: {e}"}}, 502

    response_parts = []
    if has_tools and text:
        clean_text, function_calls = parse_google_function_calls(text)
        if function_calls:
            if clean_text:
                response_parts.append({"text": clean_text})
            for fc in function_calls:
                response_parts.append(
                    {"functionCall": {"name": fc["name"], "args": fc["args"]}}
                )
        else:
            response_parts.append({"text": text})
    else:
        response_parts.append(
            {
                "text": text
                or "I apologize, but I was unable to generate a response. Please try again."
            }
        )

    response_obj = {
        "candidates": [
            {
                "content": {"parts": response_parts, "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": len(prompt) // 4,
            "candidatesTokenCount": len(text or "") // 4,
            "totalTokenCount": (len(prompt) + len(text or "")) // 4,
        },
        "modelVersion": model_name,
    }
    if stream:

        def _single():
            yield f"data: {json.dumps(response_obj, ensure_ascii=False)}\n\n"

        return Response(
            stream_with_context(_single()), content_type="text/event-stream"
        )
    return response_obj
