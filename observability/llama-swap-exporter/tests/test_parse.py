"""Tests for LlamaSwapCollector.parse_model_metrics."""

import pytest
from conftest import SAMPLE_METRICS_TEXT
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from app import LlamaSwapCollector


@pytest.fixture
def parser() -> LlamaSwapCollector:
    return LlamaSwapCollector(None)


def test_parses_counters_gauges_and_model_label(parser):
    result = parser.parse_model_metrics("m1", SAMPLE_METRICS_TEXT)

    families = list(result.values())
    # CounterMetricFamily strips trailing _total from the name
    assert {family.name for family in families} == {
        "llamacpp_prompt_tokens",
        "llamacpp_tokens_predicted",
        "llamacpp_prompt_tokens_seconds",
        "llamacpp_requests_processing",
    }
    by_name = {family.name: family for family in families}

    counter = by_name["llamacpp_prompt_tokens"]
    assert isinstance(counter, CounterMetricFamily)
    assert counter.documentation == "Number of prompt tokens processed."
    # Sample objects are iterable as (name, labels, value)
    assert any(
        s.labels.get("model") == "m1" and s.value == 1234.0 for s in counter.samples
    )
    assert {"model"} == set(counter._labelnames)

    gauge = by_name["llamacpp_prompt_tokens_seconds"]
    assert isinstance(gauge, GaugeMetricFamily)
    assert any(s.labels.get("model") == "m1" and s.value == 88.5 for s in gauge.samples)


def test_parses_labels_and_multiple_series(parser):
    result = parser.parse_model_metrics("m1", SAMPLE_METRICS_TEXT)
    families = {family.name: family for family in result.values()}
    family = families["llamacpp_requests_processing"]

    # label keys are sorted alphabetically in parser
    assert family._labelnames == ("mode", "model", "server")
    # Verify both series are present
    samples = {(s.labels["mode"], s.labels["server"]): s.value for s in family.samples}
    assert samples[("generate", "srv1")] == 1.0
    assert samples[("chat", "srv1")] == 0.0


def test_empty_input(parser):
    assert parser.parse_model_metrics("m1", "") == {}


def test_invalid_text_returns_empty(parser):
    assert parser.parse_model_metrics("m1", "foo 1 bar\n") == {}


def test_ignores_blank_lines(parser):
    text = (
        "# HELP llamacpp:a_counter Help a.\n"
        "# TYPE llamacpp:a_counter counter\n"
        "llamacpp:a_counter 1\n"
        "\n"
        "# HELP llamacpp:b_gauge Help b.\n"
        "# TYPE llamacpp:b_gauge gauge\n"
        "llamacpp:b_gauge 2\n"
    )

    result = parser.parse_model_metrics("m1", text)
    # CounterMetricFamily strips trailing _total, but a_counter has no _total suffix
    assert {family.name for family in result.values()} == {
        "llamacpp_a_counter",
        "llamacpp_b_gauge",
    }


def test_unknown_metric_type_is_skipped(parser):
    text = (
        "# HELP llamacpp:a_counter Help a.\n"
        "# TYPE llamacpp:a_counter counter\n"
        "llamacpp:a_counter 1\n"
        "# HELP llamacpp:b_summary Help b.\n"
        "# TYPE llamacpp:b_summary summary\n"
        "llamacpp:b_summary 2\n"
    )

    result = parser.parse_model_metrics("m1", text)
    assert {family.name for family in result.values()} == {"llamacpp_a_counter"}


def test_metric_without_help_line_has_empty_help(parser):
    text = (
        "# HELP llamacpp:a_counter Help a.\n"
        "# TYPE llamacpp:a_counter counter\n"
        "llamacpp:a_counter 1\n"
        "# TYPE llamacpp:b_gauge gauge\n"
        "llamacpp:b_gauge 2\n"
    )

    result = parser.parse_model_metrics("m1", text)
    families = {family.name: family for family in result.values()}
    assert families["llamacpp_b_gauge"].documentation == ""
    assert families["llamacpp_a_counter"].documentation == "Help a."


def test_metric_without_type_line_does_not_inherit_previous_type(parser):
    text = (
        "# HELP llamacpp:a_counter Help a.\n"
        "# TYPE llamacpp:a_counter counter\n"
        "llamacpp:a_counter 1\n"
        "# HELP llamacpp:b_gauge Help b.\n"
        "llamacpp:b_gauge 2\n"
    )

    result = parser.parse_model_metrics("m1", text)
    families = {family.name: family for family in result.values()}
    assert isinstance(families["llamacpp_a_counter"], CounterMetricFamily)
    # b_gauge has no TYPE line; per the exposition spec it must not become a counter
    assert not isinstance(families["llamacpp_b_gauge"], CounterMetricFamily)


def test_first_metric_without_help_line(parser):
    text = "# TYPE llamacpp:b_gauge gauge\nllamacpp:b_gauge 2\n"

    result = parser.parse_model_metrics("m1", text)
    families = {family.name: family for family in result.values()}
    assert families["llamacpp_b_gauge"].documentation == ""


def test_label_value_containing_equals(parser):
    text = (
        "# HELP llamacpp:requests_processing Help.\n"
        "# TYPE llamacpp:requests_processing gauge\n"
        'llamacpp:requests_processing{code="a=b"} 1\n'
    )

    result = parser.parse_model_metrics("m1", text)
    families = {family.name: family for family in result.values()}
    family = families["llamacpp_requests_processing"]
    assert family._labelnames == ("code", "model")
    assert any(
        s.labels.get("code") == "a=b" and s.labels.get("model") == "m1"
        for s in family.samples
    )
