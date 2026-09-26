"""Tests for LlamaSwapCollector."""

import pytest
import requests
from conftest import (
    ACTIVITY_URL,
    BASE_URL,
    PYTHON_SERVER_METRICS_TEXT,
    RUNNING_URL,
    SAMPLE_METRICS_TEXT,
    make_activity_item,
)
from prometheus_client import (
    CollectorRegistry,
    GCCollector,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)
from prometheus_client.parser import text_string_to_metric_families

from app import (
    BUILTIN_FAMILY_NAMES,
    GAUGE_SPECS,
    QUANTILE_SPECS,
    WINDOW_SAMPLES_METRIC,
    LlamaSwapCollector,
)

LLAMASWAP_GAUGE_NAMES = set(GAUGE_SPECS)
LLAMASWAP_QUANTILE_NAMES = set(QUANTILE_SPECS)
QUANTILE_LABELS = ("0.5", "0.9", "0.95", "0.99")


def _family_names(families) -> set[str]:
    return {family.name for family in families}


def _register_scrape(
    responses,
    models=("m1",),
    activity_items=None,
    status=200,
    metrics_text=SAMPLE_METRICS_TEXT,
):
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
            body=metrics_text,
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


def _quantile(families, name, q, model):
    for fam in families:
        if fam.name != name:
            continue
        for s in fam.samples:
            if s.labels.get("model") == model and s.labels.get("quantile") == q:
                return s.value
    msg = f"quantile {q!r} of {name!r} not found for {model!r}"
    raise AssertionError(msg)


def _window_sample(families, model):
    for fam in families:
        if fam.name != WINDOW_SAMPLES_METRIC:
            continue
        for s in fam.samples:
            if s.labels.get("model") == model:
                return s.value
    msg = f"window samples not found for {model!r}"
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


def test_json_to_gauges_null_duration(collector):
    item = make_activity_item("m1", duration_ms=None)
    families = collector.json_to_gauges([item])
    by_name = {family.name: family for family in families}

    assert _family_names(families) == LLAMASWAP_GAUGE_NAMES
    assert _sample_value(by_name["llamaswap_model_duration_ms"], "m1") == 0.0


def test_json_to_gauges_null_tokens(collector):
    item = make_activity_item("m1", tokens=None)
    families = collector.json_to_gauges([item])

    by_name = {family.name: family for family in families}
    for name in LLAMASWAP_GAUGE_NAMES - {"llamaswap_model_duration_ms"}:
        assert _sample_value(by_name[name], "m1") == 0.0
    # duration_ms lives on the item itself, not in tokens
    assert _sample_value(by_name["llamaswap_model_duration_ms"], "m1") == 1234.5


def test_json_to_gauges_null_token_fields_default_to_zero(collector):
    item = make_activity_item("m1", tokens={"cache_tokens": None, "draft_tokens": 5})
    families = collector.json_to_gauges([item])
    by_name = {family.name: family for family in families}

    assert _sample_value(by_name["llamaswap_model_cache_tokens"], "m1") == 0.0
    assert _sample_value(by_name["llamaswap_model_draft_tokens"], "m1") == 5.0


def test_json_to_gauges_tokens_not_a_dict(collector):
    item = make_activity_item("m1", tokens="oops")
    families = collector.json_to_gauges([item])
    by_name = {family.name: family for family in families}

    for name in LLAMASWAP_GAUGE_NAMES - {"llamaswap_model_duration_ms"}:
        assert _sample_value(by_name[name], "m1") == 0.0
    # duration_ms lives on the item itself, not in tokens
    assert _sample_value(by_name["llamaswap_model_duration_ms"], "m1") == 1234.5


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


def test_collect_ready_model_missing_model_key(collector, clock, responses):
    responses.add(responses.GET, RUNNING_URL, json={"running": [{"state": "ready"}]})
    responses.add(
        responses.GET,
        ACTIVITY_URL,
        json={"data": [make_activity_item("m1")], "total_pages": 1},
    )

    families = collector.collect()

    assert _family_names(
        families
    ) == LLAMASWAP_GAUGE_NAMES | LLAMASWAP_QUANTILE_NAMES | {WINDOW_SAMPLES_METRIC}


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


def test_parse_model_metrics_skips_builtin_families(collector):
    families = collector.parse_model_metrics("m1", PYTHON_SERVER_METRICS_TEXT)

    names = _family_names(families.values())
    assert not (names & BUILTIN_FAMILY_NAMES)
    assert names == {
        "llamacpp_prompt_tokens",
        "llamacpp_requests_processing",
        "llamacpp_tokens_predicted",
    }


def test_register_does_not_clash_with_builtin_collectors(collector, responses):
    """Regression: registering next to the default registry's builtins must not
    raise DuplicateTimeseries when a model server is a Python process."""
    registry = CollectorRegistry(auto_describe=True)
    GCCollector(registry=registry)
    PlatformCollector(registry=registry)
    ProcessCollector(registry=registry)
    _register_scrape(responses, models=("m1",), metrics_text=PYTHON_SERVER_METRICS_TEXT)

    registry.register(collector)
    output = generate_latest(registry)

    assert b'llamacpp_prompt_tokens_total{model="m1"} 1234.0' in output
    # The exporter's own builtin metrics survive; the model server's copies do
    # not leak in with a model label.
    assert b"python_info{" in output
    assert not any(
        line.startswith(("python_gc_", "process_", "python_info"))
        and '{model="' in line
        for line in output.decode().splitlines()
    )


def test_json_to_quantiles_values(collector):
    items = [
        make_activity_item("m1", duration_ms=1234.5),
        make_activity_item("m1", duration_ms=5000.0),
        make_activity_item("m1", duration_ms=250.0),
    ]

    families = collector.json_to_quantiles(items)

    assert _family_names(families) == LLAMASWAP_QUANTILE_NAMES | {WINDOW_SAMPLES_METRIC}

    # 250, 1234.5, 5000 -> linearly interpolated quantiles
    duration = "llamaswap_model_request_duration_ms"
    assert _quantile(families, duration, "0.5", "m1") == pytest.approx(1234.5)
    assert _quantile(families, duration, "0.9", "m1") == pytest.approx(4246.9)
    assert _quantile(families, duration, "0.95", "m1") == pytest.approx(4623.45)
    assert _quantile(families, duration, "0.99", "m1") == pytest.approx(4924.69)

    # Every activity entry feeds the quantiles: output_tokens=30 three times,
    # so all quantiles of a single distinct value equal the value itself.
    output = "llamaswap_model_request_output_tokens"
    for q in QUANTILE_LABELS:
        assert _quantile(families, output, q, "m1") == 30.0

    # The window samples gauge reports the number of entries per model.
    assert _window_sample(families, "m1") == 3.0


def test_json_to_quantiles_empty_input_returns_no_families(collector):
    assert collector.json_to_quantiles([]) == []


def test_json_to_quantiles_skips_non_numeric_and_malformed_entries(collector):
    items = [
        "not-a-dict",
        {"model": "m1", "tokens": {"output_tokens": "bad"}},
        {"model": "m1"},
    ]

    families = collector.json_to_quantiles(items)

    # No entry yields a numeric value for any quantile metric, so only the
    # window samples gauge is emitted (m1 has two window entries); the
    # non-dict and null-tokens entries are tolerated, not fatal.
    assert _family_names(families) == {WINDOW_SAMPLES_METRIC}
    assert _window_sample(families, "m1") == 2.0


def test_json_to_quantiles_null_field_yields_no_family_for_that_metric(collector):
    items = [
        make_activity_item("m1", duration_ms=None),
        make_activity_item("m1", duration_ms=None),
    ]

    families = collector.json_to_quantiles(items)

    # duration is absent, but the token-derived quantiles still populate.
    assert "llamaswap_model_request_duration_ms" not in _family_names(families)
    output = "llamaswap_model_request_output_tokens"
    for q in QUANTILE_LABELS:
        assert _quantile(families, output, q, "m1") == 30.0
    assert _window_sample(families, "m1") == 2.0


def test_latest_per_model_keeps_newest_and_skips_malformed():
    result = LlamaSwapCollector._latest_per_model(
        [
            "not-a-dict",
            {"model": "m1"},  # no timestamp -> ignored
            {"model": "m1", "timestamp": 9},
            {"model": "m1", "timestamp": 3},  # older -> does not replace
            {"model": "m2", "timestamp": 7},
        ]
    )

    by_model = {item["model"]: item["timestamp"] for item in result}
    assert by_model == {"m1": 9, "m2": 7}


def test_collect_exposes_activity_quantiles(collector, registry, clock, responses):
    items = [
        make_activity_item("m1", timestamp=1_700_000_000, duration_ms=100.0),
        make_activity_item("m1", timestamp=1_700_000_001, duration_ms=1000.0),
        make_activity_item("m2", timestamp=1_700_000_002, duration_ms=200.0),
    ]
    _register_scrape(responses, models=("m1", "m2"), activity_items=items)

    registry.register(collector)
    output = generate_latest(registry)

    # m1 has two samples, m2 one; per-model quantiles are exposed under their
    # distinct "request_" names, alongside the unchanged latest-value gauges.
    # p50 of 100 and 1000 interpolates to 550; a single sample repeats as-is.
    assert (
        b'llamaswap_model_request_duration_ms{model="m1",quantile="0.5"} 550.0'
        in output
    )
    assert (
        b'llamaswap_model_request_duration_ms{model="m2",quantile="0.99"} 200.0'
        in output
    )
    # The window samples gauge reports the per-model entry count.
    assert b'llamaswap_model_request_samples{model="m1"} 2.0' in output
    assert b'llamaswap_model_request_samples{model="m2"} 1.0' in output
    # The latest-value gauges for the same fields are still present and distinct.
    assert b'llamaswap_model_duration_ms{model="m1"} 1000.0' in output
    # default tokens_per_second is 50.25; all quantiles equal the sole sample.
    assert (
        b'llamaswap_model_request_tokens_per_second{model="m2",quantile="0.5"} 50.25'
        in output
    )
