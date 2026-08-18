"""Shared fixtures for llama-swap-exporter tests."""

import threading
from http.server import HTTPServer

import pytest
import responses as responses_lib
from prometheus_client import CollectorRegistry

import app
from app import LlamaSwapApi

BASE_URL = "http://llamaswap.test:8080"

SAMPLE_METRICS_TEXT = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1234
# HELP llamacpp:tokens_predicted_total Number of generated tokens.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 567
# HELP llamacpp:prompt_tokens_seconds Prompt tokens per second.
# TYPE llamacpp:prompt_tokens_seconds gauge
llamacpp:prompt_tokens_seconds 88.5
# HELP llamacpp:requests_processing Request slots occupied by processing requests.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing{mode="generate",server="srv1"} 1
llamacpp:requests_processing{mode="chat",server="srv1"} 0
"""


def make_activity_item(model: str = "m1", **overrides) -> dict:
    item = {
        "model": model,
        "timestamp": 1_700_000_000,
        "duration_ms": 1234.5,
        "tokens": {
            "cache_tokens": 10,
            "input_tokens": 20,
            "output_tokens": 30,
            "prompt_per_second": 100.5,
            "tokens_per_second": 50.25,
            "draft_tokens": 5,
            "draft_acc_tokens": 3,
        },
    }
    item.update(overrides)
    return item


@pytest.fixture(autouse=True)
def registry(monkeypatch) -> CollectorRegistry:
    """A fresh Prometheus registry for every test (order-independent isolation)."""
    reg = CollectorRegistry()
    monkeypatch.setattr(app, "REGISTRY", reg)
    return reg


@pytest.fixture
def responses():
    """Reuse the pytest-responses plugin's global mock, not a nested one.

    The plugin's default fixture starts a second RequestsMock on top of the
    global one. When both unpatch at teardown, the global's handler is left
    patched on HTTPAdapter.send, which then rejects real HTTP for later
    ``withoutresponses`` tests (seed-dependent).
    """
    return responses_lib.mock


@pytest.fixture
def client() -> LlamaSwapApi:
    return LlamaSwapApi(BASE_URL)


@pytest.fixture
def collector(client) -> app.LlamaSwapCollector:
    return app.LlamaSwapCollector(client)


@pytest.fixture
def clock(mocker) -> dict:
    """A controllable clock for app.time.time (cache-interval tests)."""
    state = {"t": 1000.0}
    mocker.patch("app.time.time", side_effect=lambda: state["t"])
    return state


@pytest.fixture
def server():
    """An ephemeral HTTP server serving app.MetricsHandler."""
    httpd = HTTPServer(("127.0.0.1", 0), app.MetricsHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)
