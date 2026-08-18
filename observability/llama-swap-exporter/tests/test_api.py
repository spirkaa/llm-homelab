"""Tests for LlamaSwapApi."""

import json
from urllib.parse import parse_qs, urlparse

import pytest
import requests
from conftest import ACTIVITY_URL, BASE_URL, RUNNING_URL, USER_AGENT, make_activity_item

from app import LlamaSwapApi


def _running_payload(models: list[dict]) -> dict:
    return {"running": models}


def _page_query(request) -> int:
    return int(parse_qs(urlparse(request.url).query)["page"][0])


def test_get_running_models(client, responses):
    model = {"model": "m1", "state": "ready"}
    responses.add(responses.GET, RUNNING_URL, json=_running_payload([model]))

    assert client.get_running_models() == [model]
    assert responses.calls[0].request.url == RUNNING_URL
    assert responses.calls[0].request.headers["User-Agent"] == USER_AGENT


def test_get_running_models_raises_on_http_error(client, responses):
    responses.add(responses.GET, RUNNING_URL, json={"error": "boom"}, status=500)

    with pytest.raises(requests.HTTPError):
        client.get_running_models()


def test_get_ready_models_filters_ready_only(client, responses):
    models = [
        {"model": "ready", "state": "ready"},
        {"model": "loading", "state": "loading"},
        {"model": "stopped", "state": "stopped"},
    ]
    responses.add(responses.GET, RUNNING_URL, json=_running_payload(models))

    assert client.get_ready_models() == [{"model": "ready", "state": "ready"}]


def test_get_ready_models_missing_state(client, responses):
    models = [{"model": "m1"}, {"model": "m2", "state": "ready"}]
    responses.add(responses.GET, RUNNING_URL, json=_running_payload(models))

    assert client.get_ready_models() == [{"model": "m2", "state": "ready"}]


def test_get_model_metrics_returns_text(client, responses):
    responses.add(
        responses.GET,
        f"{BASE_URL}/upstream/m1/metrics",
        body="# HELP m1 x\n# TYPE m1 counter\nm1 1\n",
    )

    assert client.get_model_metrics("m1") == "# HELP m1 x\n# TYPE m1 counter\nm1 1\n"


def test_get_model_metrics_http_error_returns_empty(client, responses, caplog):
    responses.add(
        responses.GET,
        f"{BASE_URL}/upstream/m1/metrics",
        json={"error": "boom"},
        status=503,
    )

    with caplog.at_level("WARNING"):
        assert client.get_model_metrics("m1") == ""
    assert any("m1" in record.message for record in caplog.records)


def test_get_llama_swap_activity_single_page(client, responses):
    items = [make_activity_item("m1", timestamp=1)]
    responses.add(responses.GET, ACTIVITY_URL, json={"data": items, "total_pages": 1})

    assert client.get_llama_swap_activity() == items
    assert len(responses.calls) == 1


def test_get_llama_swap_activity_multiple_pages(client, responses):
    pages = {
        1: {"data": [make_activity_item("m1", timestamp=1)], "total_pages": 2},
        2: {"data": [make_activity_item("m2", timestamp=2)], "total_pages": 2},
    }

    def callback(request):
        return (
            200,
            {"Content-Type": "application/json"},
            json.dumps(pages[_page_query(request)]),
        )

    responses.add_callback(responses.GET, ACTIVITY_URL, callback=callback)

    assert client.get_llama_swap_activity() == [
        make_activity_item("m1", timestamp=1),
        make_activity_item("m2", timestamp=2),
    ]
    assert len(responses.calls) == 2


def test_get_running_models_invalid_json_returns_empty(client, responses, caplog):
    responses.add(
        responses.GET, RUNNING_URL, body="not json", content_type="text/plain"
    )

    with caplog.at_level("WARNING"):
        assert client.get_running_models() == []
    assert any("Invalid JSON" in record.message for record in caplog.records)


def test_get_llama_swap_activity_invalid_json(client, responses, caplog):
    responses.add(
        responses.GET, ACTIVITY_URL, body="not json", content_type="text/plain"
    )

    with caplog.at_level("WARNING"):
        assert client.get_llama_swap_activity() == []
    assert any("Invalid JSON" in record.message for record in caplog.records)


def test_get_llama_swap_activity_stops_on_empty_data(client, responses):
    responses.add(
        responses.GET,
        ACTIVITY_URL,
        json={"data": [], "page": 1, "limit": 100, "total": 0, "total_pages": 0},
    )

    assert client.get_llama_swap_activity() == []
    assert len(responses.calls) == 1


def test_get_llama_swap_activity_returns_partial_on_later_page_error(client, responses):
    first_page = [make_activity_item("m1", timestamp=1)]
    calls = {"n": 0}

    def callback(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return (
                200,
                {"Content-Type": "application/json"},
                json.dumps({"data": first_page, "total_pages": 2}),
            )
        return 500, {"Content-Type": "application/json"}, json.dumps({"error": "boom"})

    responses.add_callback(responses.GET, ACTIVITY_URL, callback=callback)

    assert client.get_llama_swap_activity() == first_page


def test_get_llama_swap_activity_terminates_on_inconsistent_total_pages(
    client, responses
):
    """A server that always reports more pages must not loop forever."""
    state = {"calls": 0}

    def callback(request):
        state["calls"] += 1
        page = _page_query(request)
        body = json.dumps(
            {
                "data": [make_activity_item(f"m{page}", timestamp=page)],
                "total_pages": page + 1,
            }
        )
        return 200, {"Content-Type": "application/json"}, body

    responses.add_callback(responses.GET, ACTIVITY_URL, callback=callback)

    result = client.get_llama_swap_activity()
    assert len(result) == 100
    # Loop is bounded by ACTIVITY_MAX_PAGES
    assert state["calls"] == 100


def test_get_llama_swap_metrics_keeps_latest_per_model(client, responses):
    # m1 ts=2 after m1 ts=3: the older duplicate must not win
    items = [
        make_activity_item("m1", timestamp=1),
        make_activity_item("m1", timestamp=3),
        make_activity_item("m2", timestamp=2),
        make_activity_item("m1", timestamp=2),
    ]
    responses.add(responses.GET, ACTIVITY_URL, json={"data": items, "total_pages": 1})

    result = client.get_llama_swap_metrics()
    assert {item["model"] for item in result} == {"m1", "m2"}
    assert {item["model"]: item["timestamp"] for item in result} == {"m1": 3, "m2": 2}


def test_get_llama_swap_metrics_malformed_entries(client, responses):
    items = [
        make_activity_item("m1", timestamp=1),
        {"timestamp": 2},  # missing "model"
        {"model": "m2"},  # missing "timestamp"
    ]
    responses.add(responses.GET, ACTIVITY_URL, json={"data": items, "total_pages": 1})

    result = client.get_llama_swap_metrics()
    assert {item["model"] for item in result} == {"m1"}


NETWORK_ERRORS = [
    pytest.param(
        requests.exceptions.ConnectionError("connection refused"), id="connection-error"
    ),
    pytest.param(requests.exceptions.ReadTimeout("read timed out"), id="timeout"),
]


@pytest.mark.parametrize("exc", NETWORK_ERRORS)
def test_get_running_models_network_error(client, responses, exc):
    responses.add(responses.GET, RUNNING_URL, body=exc)

    assert client.get_running_models() == []


@pytest.mark.parametrize("exc", NETWORK_ERRORS)
def test_get_model_metrics_network_error(client, responses, exc):
    responses.add(responses.GET, f"{BASE_URL}/upstream/m1/metrics", body=exc)

    assert client.get_model_metrics("m1") == ""


@pytest.mark.parametrize("exc", NETWORK_ERRORS)
def test_get_llama_swap_activity_network_error(client, responses, exc):
    responses.add(responses.GET, ACTIVITY_URL, body=exc)

    assert client.get_llama_swap_activity() == []


def test_client_keeps_trailing_slash():
    client = LlamaSwapApi(f"{BASE_URL}/")

    assert client.base_url == f"{BASE_URL}/"


def test_get_running_models_base_url_with_path_prefix(responses):
    prefixed_client = LlamaSwapApi(f"{BASE_URL}/proxy")
    responses.add(
        responses.GET,
        f"{BASE_URL}/proxy/running",
        json=_running_payload([{"model": "m1", "state": "ready"}]),
    )

    prefixed_client.get_running_models()

    assert any(
        urlparse(call.request.url).path == "/proxy/running" for call in responses.calls
    )
