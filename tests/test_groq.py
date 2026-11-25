"""Full crew run against a local server that is as strict as Groq's documentation."""

import json
import re
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from procurement.config import COMPACT, Company, ConfigError, Limits, build_llm, limits_for
from procurement.crew import run_procurement
from procurement.demo import _OFFERS, DEMO_REQUEST, _page, demo_script, demo_services
from procurement.groq_llm import build_groq_llm
from procurement.tools import shorten

ROLES = {
    "Procurement Planner": "plan",
    "Product Search Specialist": "search",
    "Web Scraping Specialist": "scrape",
    "Product Comparison Analyst": "analyse",
    "Procurement Report Writer": "report",
}
OFFERS = _OFFERS[:4]
LONG_SPECS = [{"name": f"مواصفة {i}", "value": "قيمة طويلة جدا " * 6} for i in range(40)]


def check_strict_schema(schema, problems):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            if schema.get("additionalProperties") is not False:
                problems.append("additionalProperties must be false")
            if set(schema.get("required", [])) != set(schema.get("properties", {})):
                problems.append("every property must be required")
        for value in schema.values():
            check_strict_schema(value, problems)
    elif isinstance(schema, list):
        for value in schema:
            check_strict_schema(value, problems)


@pytest.fixture
def groq(monkeypatch):
    script = demo_script()
    scrape = json.loads(script["scrape"][-1].split("Final Answer: ")[1])
    scrape["products"] = scrape["products"][:4]
    script["scrape"] = script["scrape"][:4] + [f"Final Answer: {json.dumps(scrape)}"]
    position = defaultdict(int)
    fake = SimpleNamespace(requests=[], problems=[], rate_limited=False, answer_as_tool_call=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, **headers):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            if not fake.rate_limited:  # the first call hits the rate limit
                fake.rate_limited = True
                return self.reply(429, {"error": {"message": "rate limit"}}, retry_after="1")
            fake.requests.append(body)
            if any("name" in m for m in body["messages"]):
                fake.problems.append("messages[].name")
            if body.get("tools") and body.get("response_format"):
                fake.problems.append("tools together with structured output")
            if body.get("response_format"):
                spec = body["response_format"]["json_schema"]
                if spec.get("strict") is not True:
                    fake.problems.append("not strict")
                check_strict_schema(spec["schema"], fake.problems)

            system = json.dumps(body["messages"])
            task = next(v for k, v in ROLES.items() if k in system)
            line = script[task][min(position[task], len(script[task]) - 1)]
            message = {"role": "assistant", "content": None}
            if "Action: " in line:
                position[task] += 1
                message["tool_calls"] = [
                    {
                        "id": f"call_{task}{position[task]}",
                        "type": "function",
                        "function": {
                            "name": re.search(r"Action: (\w+)", line)[1],
                            "arguments": line.split("Action Input: ")[1],
                        },
                    }
                ]
            elif task == "search" and not fake.answer_as_tool_call:
                fake.answer_as_tool_call = True  # seen on the real service
                answer = json.loads(line.split("Final Answer: ")[1])
                call = {"name": "candidateList", "arguments": answer}
                error = {
                    "message": "attempted to call tool 'candidateList' "
                    "which was not in request.tools",
                    "type": "invalid_request_error",
                    "code": "tool_use_failed",
                    "failed_generation": json.dumps(call),
                }
                return self.reply(400, {"error": error})
            else:
                message["content"] = line.split("Final Answer: ")[1]
            self.reply(
                200,
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fake.url = f"http://127.0.0.1:{server.server_port}/openai/v1"
    yield fake
    server.shutdown()
    server.server_close()


def test_full_run_on_groq(groq):
    services = demo_services()
    services.llm = build_groq_llm("openai/gpt-oss-120b", "test", COMPACT.output_tokens, groq.url)
    pages = {o[1]: {**_page(o), "specs": _page(o)["specs"] + LONG_SPECS} for o in OFFERS}

    def extract(prompt, *, url, **_):
        data = SimpleNamespace(json_data=pages[url], raw=None)
        return SimpleNamespace(status="success", data=data, error=None)

    services.scrapegraph = SimpleNamespace(extract=extract)
    state, content = run_procurement(
        DEMO_REQUEST, Company.load(), services, limits=COMPACT, verbose=False
    )
    assert content.recommended_product_id == "P1"
    assert len(state.pages) == 4 and not state.warnings
    assert groq.problems == []
    assert groq.rate_limited and groq.answer_as_tool_call
    assert len(groq.requests) == 13  # the repaired answer cost no extra call

    for body in groq.requests:
        assert body["model"] == "openai/gpt-oss-120b"
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == COMPACT.output_tokens
        # Rough upper bound: 3 characters per token, output reserved in full.
        size = len(json.dumps(body, ensure_ascii=False)) / 3 + COMPACT.output_tokens
        assert size < 8000, f"request needs about {size:.0f} tokens"


def test_groq_settings(monkeypatch):
    monkeypatch.delenv("COMPACT", raising=False)
    assert limits_for("groq/openai/gpt-oss-120b") is COMPACT
    assert limits_for("anthropic/claude-opus-5-5") == Limits()
    monkeypatch.setenv("COMPACT", "false")
    assert limits_for("groq/openai/gpt-oss-120b") == Limits()
    monkeypatch.setenv("LLM_MODEL", "groq/openai/gpt-oss-120b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent-to-groq")
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        build_llm()
    monkeypatch.setenv("GROQ_API_KEY", "g")
    llm = build_llm()
    assert (llm.api_key, llm.model) == ("g", "openai/gpt-oss-120b")


def test_shorten_keeps_the_price():
    text = json.dumps({"specs": LONG_SPECS, "price": 38500}, ensure_ascii=False)
    short = shorten(text, 500)
    assert len(short) <= 500 and json.loads(short)["price"] == 38500
