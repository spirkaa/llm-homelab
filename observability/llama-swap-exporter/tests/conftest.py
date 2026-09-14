"""Shared fixtures for llama-swap-exporter tests."""

import functools
import threading
from http.server import HTTPServer

import pytest
import responses as responses_lib
from prometheus_client import CollectorRegistry

import app
from app import LlamaSwapApi

BASE_URL = "http://llamaswap.test:8080"
RUNNING_URL = f"{BASE_URL}/running"
ACTIVITY_URL = f"{BASE_URL}/api/metrics/activity"
USER_AGENT = "llama-swap-exporter/0.1.0"

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

# A model server that is a Python process: on top of llamacpp metrics it
# exposes prometheus_client's own gc/platform/process collectors.
PYTHON_SERVER_METRICS_TEXT = """# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1234
# HELP llamacpp:tokens_predicted_total Number of generated tokens.
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 567
# HELP llamacpp:requests_processing Request slots occupied by processing requests.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing{mode="generate",server="srv1"} 1
# HELP process_virtual_memory_bytes Virtual memory size in bytes.
# TYPE process_virtual_memory_bytes gauge
process_virtual_memory_bytes 12345678
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 1234567
# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.
# TYPE process_start_time_seconds gauge
process_start_time_seconds 1700000000
# HELP process_open_fds Number of open file descriptors.
# TYPE process_open_fds gauge
process_open_fds 25
# HELP process_max_fds Maximum number of open file descriptors.
# TYPE process_max_fds gauge
process_max_fds 1024
# HELP process_cpu_seconds Total user and system CPU time spent in seconds.
# TYPE process_cpu_seconds counter
process_cpu_seconds_total 12.5
# HELP python_info Python platform information
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="14",patchlevel="0",version="3.14.0"} 1
# HELP python_gc_collections Number of times this generation was collected
# TYPE python_gc_collections counter
python_gc_collections{generation="0"} 10
# HELP python_gc_objects_collected Objects collected during gc
# TYPE python_gc_objects_collected counter
python_gc_objects_collected{generation="0"} 100
# HELP python_gc_objects_uncollectable Uncollectable objects found during GC
# TYPE python_gc_objects_uncollectable counter
python_gc_objects_uncollectable{generation="0"} 0
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
    # Short poll interval so shutdown() in teardown doesn't wait ~0.5s.
    thread = threading.Thread(
        target=functools.partial(httpd.serve_forever, poll_interval=0.01), daemon=True
    )
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)
