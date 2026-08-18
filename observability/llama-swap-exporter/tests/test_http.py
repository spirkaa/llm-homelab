"""Tests for the HTTP handler, signal handler, and main()."""

import signal

import pytest
import requests
import responses
from conftest import BASE_URL, SAMPLE_METRICS_TEXT, make_activity_item
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

import app

RUNNING_URL = f"{BASE_URL}/running"
ACTIVITY_URL = f"{BASE_URL}/api/metrics/activity"

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

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/html"
    assert b"/metrics" in response.content


def test_unknown_path_returns_html(server):
    # Current behavior: any path is answered with 200 and the index page.
    # (Prometheus convention would be 404; pinned as-is on purpose.)
    response = requests.get(server + "/does-not-exist", timeout=5)

    assert response.status_code == 200
    assert b"/metrics" in response.content


@pytest.mark.xfail(
    reason=(
        "bug: '/metrics' is matched with an exact string compare, so a query string "
        "returns the HTML index instead of metrics (app.py:246)"
    )
)
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


@pytest.mark.xfail(
    reason=(
        "bug: a failed scrape wipes the last good cache, so after an upstream failure "
        "scrapes within the refresh interval return no llamaswap metrics (app.py:224-236)"
    )
)
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


def test_signal_handler_terminates():
    with pytest.raises(SystemExit):
        app.signal_handler(signal.SIGTERM)


def test_main_registers_collector(mocker, registry):
    mocker.patch("app.run_http_server")
    register_spy = mocker.spy(registry, "register")

    app.main()

    assert register_spy.call_count == 1
    assert isinstance(register_spy.call_args.args[0], app.LlamaSwapCollector)
