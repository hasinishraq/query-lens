"""A tiny OpenAI-compatible HTTP server for tests (no network)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def chat_reply(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "fake-model-2026-01",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments", {}))},
            }
            for i, tc in enumerate(tool_calls)
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class FakeServer:
    """Queue of (status, body, headers) responses served in order to /chat/completions."""

    def __init__(self) -> None:
        self.queue: list[tuple[int, dict, dict[str, str]]] = []
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: dict, headers: dict[str, str]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                self._send(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}, {})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                server.requests.append(json.loads(self.rfile.read(n) or b"{}"))
                if server.queue:
                    status, body, headers = server.queue.pop(0)
                else:
                    status, body, headers = 200, chat_reply("default"), {}
                self._send(status, body, headers)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def push(self, body: dict, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.queue.append((status, body, headers or {}))

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
