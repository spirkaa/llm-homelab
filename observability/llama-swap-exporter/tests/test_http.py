"""Tests for the HTTP handler, signal handler, and main()."""

import functools
import signal
import threading
import time

import pytest
import requests
import responses
from conftest import (
    ACTIVITY_URL,
    BASE_URL,
    RUNNING_URL,
    SAMPLE_METRICS_TEXT,
    make_activity_item,
)
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

import app

# The pytest-responses plugin activates the responses mock for every test
# (responses.start() in pytest_runtest_setup), which would intercept the real
# requests made to the ephemeral server. Opt out of the global mock.
pytestmark = pytest.mark.withoutresponses


class DummyCollector(Collector):
    def collect(self):
        family = GaugeMetricFamily("test_dummy", "A test dummy gauge.", labels=["key"])
        family.add_metric(["v1"], 42)
        return [family]


class BrokenCollector(Collector):
    def collect(self):
        raise RuntimeError


def test_metrics_endpoint(server, registry):
    registry.register(DummyCollector())

    response = requests.get(f"{server}/metrics", timeout=5)

    assert response.status_code == 200
    assert response.headers["Content-Type"] == CONTENT_TYPE_LATEST
    assert b'test_dummy{key="v1"} 42.0' in response.content


def test_root_endpoint_returns_html(server):
    response = requests.get(server + "/", timeout=5)

    assert response.status_code == 404


def test_unknown_path_returns_html(server):
    # Unknown paths should return 404
    response = requests.get(server + "/does-not-exist", timeout=5)

    assert response.status_code == 404


def test_health_endpoint(server):
    response = requests.get(f"{server}/health", timeout=5)

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/plain"
    assert response.content == b"OK"


def test_health_endpoint_with_query_string(server):
    response = requests.get(f"{server}/health?probe=true", timeout=5)

    assert response.status_code == 200
    assert response.content == b"OK"


def test_metrics_endpoint_with_query_string(server, registry):
    registry.register(DummyCollector())

    response = requests.get(f"{server}/metrics?name[]=test_dummy", timeout=5)

    assert response.status_code == 200
    assert response.headers["Content-Type"] == CONTENT_TYPE_LATEST
    assert b'test_dummy{key="v1"} 42.0' in response.content


def test_metrics_endpoint_500_on_broken_collector(server, registry):
    registry.register(BrokenCollector())

    response = requests.get(f"{server}/metrics", timeout=5)

    assert response.status_code == 500


def test_scrape_failure_keeps_last_good_metrics(server, registry, clock):
    collector = app.LlamaSwapCollector(app.LlamaSwapApi(BASE_URL))
    registry.register(collector)

    with responses.RequestsMock() as rsps:
        rsps.add_passthru("http://127.0.0.1")
        rsps.add(
            responses.GET,
            RUNNING_URL,
            json={"running": [{"model": "m1", "state": "ready"}]},
        )
        rsps.add(
            responses.GET, f"{BASE_URL}/upstream/m1/metrics", body=SAMPLE_METRICS_TEXT
        )
        rsps.add(
            responses.GET,
            ACTIVITY_URL,
            json={"data": [make_activity_item("m1")], "total_pages": 1},
        )

        good = requests.get(f"{server}/metrics", timeout=5)
        assert good.status_code == 200
        assert b'llamacpp_prompt_tokens_total{model="m1"}' in good.content

        rsps.add(responses.GET, RUNNING_URL, json={"error": "down"}, status=500)
        clock["t"] += 60
        requests.get(f"{server}/metrics", timeout=5)

        clock["t"] += 1
        retry = requests.get(f"{server}/metrics", timeout=5)

    assert retry.status_code == 200
    assert b'llamacpp_prompt_tokens_total{model="m1"}' in retry.content


def test_create_serves_metrics_on_ephemeral_port(registry, monkeypatch):
    monkeypatch.setattr(app, "EXPORTER_PORT", 0)
    httpd = app.create_server()
    port = httpd.server_address[1]
    # Short poll interval so shutdown() below doesn't wait ~0.5s.
    thread = threading.Thread(
        target=functools.partial(httpd.serve_forever, poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        registry.register(DummyCollector())

        response = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=5)
        assert response.status_code == 200
        assert b'test_dummy{key="v1"} 42.0' in response.content
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_run_http_server_serves_metrics(registry, monkeypatch):
    monkeypatch.setattr(app, "EXPORTER_PORT", 0)
    servers = []
    real_create = app.create_server

    def spy():
        server = real_create()
        # Short poll interval so shutdown() below doesn't wait ~0.5s.
        server.serve_forever = functools.partial(
            server.serve_forever, poll_interval=0.01
        )
        servers.append(server)
        return server

    monkeypatch.setattr(app, "create_server", spy)
    thread = threading.Thread(target=app.run_http_server, daemon=True)
    thread.start()
    while not servers:
        time.sleep(0.01)
    httpd = servers[0]
    try:
        registry.register(DummyCollector())

        response = requests.get(
            f"http://127.0.0.1:{httpd.server_address[1]}/metrics", timeout=5
        )
        assert response.status_code == 200
        assert b'test_dummy{key="v1"} 42.0' in response.content
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_signal_handler_terminates():
    with pytest.raises(SystemExit):
        app.signal_handler(signal.SIGTERM)


def test_main_registers_collector(mocker, registry):
    mocker.patch("app.run_http_server")
    register_spy = mocker.spy(registry, "register")

    app.main()

    assert register_spy.call_count == 1
    assert isinstance(register_spy.call_args.args[0], app.LlamaSwapCollector)
