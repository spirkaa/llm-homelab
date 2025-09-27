import itertools
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from operator import itemgetter
from urllib.parse import urljoin

import requests
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector

LLAMA_SWAP_BASE_URL = os.getenv("LLAMA_SWAP_BASE_URL", "http://localhost:8080")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "15"))
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8081"))

METRIC_TYPES = {"counter": CounterMetricFamily, "gauge": GaugeMetricFamily}


class LlamaSwapApi:
    """Llama-swap API wrapper."""

    def __init__(self, base_url: str, timeout: int = 5):
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {"User-Agent": "llama-swap-exporter/0.1.0"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_running_models(self) -> list[dict]:
        url = urljoin(self.base_url, "/running")
        res = self.session.get(url, timeout=self.timeout)
        res.raise_for_status()
        return res.json().get("running", [])

    def get_ready_models(self) -> list[dict]:
        models = self.get_running_models()
        return [model for model in models if model["state"] == "ready"]

    def get_model_metrics(self, model_name: str) -> str:
        url = urljoin(self.base_url, f"/upstream/{model_name}/metrics")
        res = self.session.get(url, timeout=self.timeout)
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError:
            return ""
        return res.text

    def get_llama_swap_activity(self) -> list[dict]:
        url = urljoin(self.base_url, "/api/metrics")
        res = self.session.get(url, timeout=self.timeout)
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError:
            return []
        result = res.json()
        if not result:
            return []
        return result

    def get_llama_swap_metrics(self) -> list[dict]:
        activity = self.get_llama_swap_activity()
        data_sorted = sorted(activity, key=itemgetter("model"))
        groups = itertools.groupby(data_sorted, key=itemgetter("model"))
        return [max(group, key=itemgetter("timestamp")) for _, group in groups]


class LlamaSwapCollector(Collector):
    def __init__(self, client: LlamaSwapApi):
        self.client = client
        self.last_scrape = 0
        self.cached_metrics = []

    def _make_metric(self, name: str, documentation: str) -> GaugeMetricFamily:
        return GaugeMetricFamily(name, documentation, labels=["model"])

    def json_to_gauges(self, data: list[dict]) -> list[GaugeMetricFamily]:
        """Convert a list of JSON objects into GaugeMetricFamily objects."""
        cache_metric = self._make_metric(
            "llamaswap_model_cache_tokens", "Number of prompt tokens from cache."
        )
        input_metric = self._make_metric(
            "llamaswap_model_input_tokens", "Number of prompt tokens processed."
        )
        output_metric = self._make_metric(
            "llamaswap_model_output_tokens", "Number of generated tokens."
        )
        prompt_pps_metric = self._make_metric(
            "llamaswap_model_prompt_per_second", "Prompt tokens per second."
        )
        tokens_pps_metric = self._make_metric(
            "llamaswap_model_tokens_per_second", "Generated tokens per second."
        )
        duration_metric = self._make_metric(
            "llamaswap_model_duration_ms", "Duration of the request in milliseconds."
        )

        for entry in data:
            common_label = [entry["model"]]

            cache_metric.add_metric(common_label, float(entry["cache_tokens"]))
            input_metric.add_metric(common_label, float(entry["input_tokens"]))
            output_metric.add_metric(common_label, float(entry["output_tokens"]))
            prompt_pps_metric.add_metric(
                common_label, float(entry["prompt_per_second"])
            )
            tokens_pps_metric.add_metric(
                common_label, float(entry["tokens_per_second"])
            )
            duration_metric.add_metric(common_label, float(entry["duration_ms"]))

        return [
            cache_metric,
            input_metric,
            output_metric,
            prompt_pps_metric,
            tokens_pps_metric,
            duration_metric,
        ]

    def parse_model_metrics(self, model_name: str, metrics: str) -> dict:
        result = {}
        for line in metrics.splitlines():
            if line.startswith("# HELP"):
                # # HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
                metric_help = " ".join(line.split()[2:])
                continue
            if line.startswith("# TYPE"):
                # # TYPE llamacpp:prompt_tokens_total counter
                metric_type = METRIC_TYPES[line.split()[-1]]
                continue

            parts = line.split()
            name_and_labels = parts[0]
            metric_value = float(parts[1])

            if "{" in name_and_labels:
                metric_name = name_and_labels.split("{")[0]
                label_str = name_and_labels.split("{")[1].rstrip("}")
                labels = dict(item.split("=") for item in label_str.split(","))
                labels = {k: v.strip('"') for k, v in labels.items()}
            else:
                metric_name = name_and_labels
                labels = {}

            metric_name = metric_name.replace(":", "_")
            labels["model"] = model_name

            label_keys = tuple(sorted(labels.keys()))
            label_values = tuple(labels[k] for k in label_keys)
            metric_key = (metric_name, label_keys)
            if metric_key not in result:
                result[metric_key] = metric_type(
                    metric_name, metric_help, labels=label_keys
                )
            result[metric_key].add_metric(label_values, metric_value)
        return result

    def collect(self):
        current_time = time.time()
        if current_time - self.last_scrape < REFRESH_INTERVAL:
            for metric in self.cached_metrics:
                yield metric
            return

        self.cached_metrics.clear()
        self.last_scrape = current_time

        models = self.client.get_ready_models()
        for model in models:
            model_name = model["model"]
            model_metrics = self.client.get_model_metrics(model_name)
            result = self.parse_model_metrics(model_name, model_metrics)
            self.cached_metrics.extend(result.values())

        swap_metrics = self.client.get_llama_swap_metrics()
        swap_metrics = self.json_to_gauges(swap_metrics)
        self.cached_metrics.extend(swap_metrics)

        for metric in self.cached_metrics:
            yield metric


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics":
            try:
                output = generate_latest(REGISTRY)
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)
            except Exception as e:
                self.send_error(500, f"Failed to generate metrics: {e}")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Llama-swap Exporter</h1><p>See <a href='/metrics'>/metrics</a></p></body></html>"
            )


def signal_handler(*args) -> None:
    """Callback function to exit when signal received."""
    signal_name = signal.Signals(args[0]).name
    print(f"Terminating, {signal_name} signal received.")
    time.sleep(0)
    raise SystemExit


def run_http_server() -> None:
    server = HTTPServer(("", EXPORTER_PORT), MetricsHandler)
    print(f"Llama-swap Exporter listening on :{EXPORTER_PORT}")
    server.serve_forever()


def main() -> None:
    """Execute main function."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    client = LlamaSwapApi(LLAMA_SWAP_BASE_URL)
    REGISTRY.register(LlamaSwapCollector(client))

    run_http_server()


if __name__ == "__main__":
    main()
