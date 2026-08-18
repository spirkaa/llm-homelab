"""Tests for LlamaSwapCollector: json_to_gauges and collect()."""

import pytest
import requests
from conftest import BASE_URL, SAMPLE_METRICS_TEXT, make_activity_item
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

RUNNING_URL = f"{BASE_URL}/running"
ACTIVITY_URL = f"{BASE_URL}/api/metrics/activity"

LLAMASWAP_GAUGE_NAMES = {
    "llamaswap_model_cache_tokens",
    "llamaswap_model_input_tokens",
    "llamaswap_model_output_tokens",
    "llamaswap_model_prompt_per_second",
    "llamaswap_model_tokens_per_second",
    "llamaswap_model_duration_ms",
    "llamaswap_model_draft_tokens",
    "llamaswap_model_draft_acc_tokens",
}


def _family_names(families) -> set[str]:
    return {family.name for family in families}


def _register_scrape(responses, models=("m1",), activity_items=None, status=200):
    if activity_items is None:
        activity_items = [make_activity_item(model) for model in models]
    responses.add(
        responses.GET,
        RUNNING_URL,
        json={"running": [{"model": model, "state": "ready"} for model in models]},
        status=status,
    )
    for model in models:
        responses.add(
            responses.GET,
            f"{BASE_URL}/upstream/{model}/metrics",
            body=SAMPLE_METRICS_TEXT,
        )
    responses.add(
        responses.GET,
        ACTIVITY_URL,
        json={"data": activity_items, "total_pages": 1},
        status=status,
    )


def _sample_value(family, label_value):
    for s in family.samples:
        if s.labels.get("model") == label_value:
            return s.value
    msg = f"sample not found in {family.name!r} for {label_value!r}"
    raise AssertionError(msg)


def test_json_to_gauges_values(collector):
    item = make_activity_item("m1")
    families = collector.json_to_gauges([item])

    assert _family_names(families) == LLAMASWAP_GAUGE_NAMES
    by_name = {family.name: family for family in families}

    expected = {
        "llamaswap_model_cache_tokens": 10.0,
        "llamaswap_model_input_tokens": 20.0,
        "llamaswap_model_output_tokens": 30.0,
        "llamaswap_model_prompt_per_second": 100.5,
        "llamaswap_model_tokens_per_second": 50.25,
        "llamaswap_model_duration_ms": 1234.5,
        "llamaswap_model_draft_tokens": 5.0,
        "llamaswap_model_draft_acc_tokens": 3.0,
    }
    for name, value in expected.items():
        assert by_name[name]._labelnames == ("model",)
        assert _sample_value(by_name[name], "m1") == value


def test_json_to_gauges_missing_token_fields_default_to_zero(collector):
    item = make_activity_item("m1", tokens={"cache_tokens": 10})
    families = collector.json_to_gauges([item])
    by_name = {family.name: family for family in families}

    assert _sample_value(by_name["llamaswap_model_cache_tokens"], "m1") == 10.0
    assert _sample_value(by_name["llamaswap_model_input_tokens"], "m1") == 0.0
    # duration_ms comes from the item, not tokens
    assert _sample_value(by_name["llamaswap_model_duration_ms"], "m1") == 1234.5


@pytest.mark.xfail(reason="bug: null 'duration_ms' raises TypeError (app.py:160)")
def test_json_to_gauges_null_duration(collector):
    item = make_activity_item("m1", duration_ms=None)
    families = collector.json_to_gauges([item])

    by_name = {family.name: family for family in families}
    assert _family_names(by_name) == LLAMASWAP_GAUGE_NAMES


@pytest.mark.xfail(
    reason="bug: null 'tokens' object raises AttributeError (app.py:148)"
)
def test_json_to_gauges_null_tokens(collector):
    item = make_activity_item("m1", tokens=None)
    families = collector.json_to_gauges([item])

    by_name = {family.name: family for family in families}
    assert _sample_value(by_name["llamaswap_model_cache_tokens"], "m1") == 0.0


@pytest.mark.xfail(reason="bug: entry without 'model' raises KeyError (app.py:146)")
def test_json_to_gauges_entry_without_model_skipped(collector):
    families = collector.json_to_gauges(
        [make_activity_item("m1"), {"model_missing": True}]
    )

    label_values = set()
    for family in families:
        for s in family.samples:
            label_values.add(s.labels.get("model"))
    assert _family_names(families) == LLAMASWAP_GAUGE_NAMES
    assert label_values == {"m1"}


def test_collect_exposes_upstream_and_activity_metrics(
    collector, registry, clock, responses
):
    _register_scrape(responses, models=("m1",))

    families = collector.collect()
    names = _family_names(families)

    # CounterMetricFamily strips _total from the family name; the exposition re-adds it
    assert {"llamacpp_prompt_tokens", "llamacpp_requests_processing"} <= names
    assert names >= LLAMASWAP_GAUGE_NAMES

    registry.register(collector)
    output = generate_latest(registry)
    assert b'llamacpp_prompt_tokens_total{model="m1"} 1234.0' in output
    assert b'llamaswap_model_input_tokens{model="m1"} 20.0' in output


def test_collect_caches_within_interval(collector, registry, clock, responses):
    _register_scrape(responses, models=("m1",))

    first = collector.collect()
    calls_after_first = len(responses.calls)

    second = collector.collect()

    assert second == first
    assert len(responses.calls) == calls_after_first


def test_collect_refreshes_after_interval(collector, registry, clock, responses):
    _register_scrape(responses, models=("m1",))

    first = collector.collect()
    calls_after_first = len(responses.calls)

    clock["t"] += 60
    second = collector.collect()

    assert _family_names(second) == _family_names(first)
    assert (
        len(responses.calls) == calls_after_first + 3
    )  # /running + /upstream/m1/metrics + activity


def test_collect_multiple_models_exposes_metrics_for_each_model(
    collector, registry, clock, responses
):
    _register_scrape(responses, models=("m1", "m2"))

    registry.register(collector)
    output = generate_latest(registry)

    # Each ready model gets its own family per metric name; the same name
    # with different labels is valid, and the text parser accepts the output.
    families = list(text_string_to_metric_families(output.decode()))
    assert families

    assert b'llamacpp_prompt_tokens_total{model="m1"} 1234.0' in output
    assert b'llamacpp_prompt_tokens_total{model="m2"} 1234.0' in output
    assert b'llamaswap_model_input_tokens{model="m1"} 20.0' in output
    assert b'llamaswap_model_input_tokens{model="m2"} 20.0' in output


@pytest.mark.xfail(
    reason="bug: a ready model entry without a 'model' key crashes the whole scrape (app.py:229)"
)
def test_collect_ready_model_missing_model_key(collector, clock, responses):
    responses.add(responses.GET, RUNNING_URL, json={"running": [{"state": "ready"}]})
    responses.add(
        responses.GET,
        ACTIVITY_URL,
        json={"data": [make_activity_item("m1")], "total_pages": 1},
    )

    families = collector.collect()

    assert _family_names(families) == LLAMASWAP_GAUGE_NAMES


@pytest.mark.xfail(
    reason=(
        "bug: failed scrape wipes the last good cache, so subsequent scrapes "
        "within the interval return no metrics (app.py:224-236)"
    )
)
def test_failed_scrape_keeps_last_good_cache(collector, clock, responses):
    _register_scrape(responses, models=("m1",))
    good = collector.collect()
    assert good

    # Later matches win (responses matches LIFO): make /running fail from now on.
    responses.add(responses.GET, RUNNING_URL, json={"error": "down"}, status=500)
    clock["t"] += 60

    with pytest.raises(requests.HTTPError):
        collector.collect()

    assert collector.collect() == good


@pytest.mark.xfail(
    reason=(
        "bug: non-JSON response from the activity endpoint crashes the whole "
        "scrape (unhandled res.json(), app.py:87)"
    )
)
def test_collect_tolerates_invalid_activity_json(collector, clock, responses):
    responses.add(responses.GET, RUNNING_URL, json={"running": []})
    responses.add(
        responses.GET,
        ACTIVITY_URL,
        body="<html>Bad gateway</html>",
        content_type="text/html",
    )

    families = collector.collect()

    assert _family_names(families) == LLAMASWAP_GAUGE_NAMES
