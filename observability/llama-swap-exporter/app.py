"""Prometheus exporter for llama-swap."""

import itertools
import logging
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from operator import itemgetter
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    handlers=[logging.StreamHandler()],
)
logging.getLogger("urllib3").setLevel(logging.WARNING)

load_dotenv()

LLAMA_SWAP_BASE_URL = os.getenv("LLAMA_SWAP_BASE_URL", "http://localhost:8080")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "15"))
EXPORTER_PORT = int(os.getenv("EXPORTER_PORT", "8081"))

METRIC_TYPES = {"counter": CounterMetricFamily, "gauge": GaugeMetricFamily}


class LlamaSwapApi:
    """Llama-swap API wrapper."""

    def __init__(self, base_url: str, timeout: int = 5) -> None:
        """Initialize API client."""
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {"User-Agent": "llama-swap-exporter/0.1.0"}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_running_models(self) -> list[dict]:
        """Get running models."""
        url = urljoin(self.base_url, "/running")
        res = self.session.get(url, timeout=self.timeout)
        res.raise_for_status()
        return res.json().get("running", [])

    def get_ready_models(self) -> list[dict]:
        """Get ready models."""
        models = self.get_running_models()
        return [model for model in models if model["state"] == "ready"]

    def get_model_metrics(self, model_name: str) -> str:
        """Get model metrics."""
        url = urljoin(self.base_url, f"/upstream/{model_name}/metrics")
        res = self.session.get(url, timeout=self.timeout)
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.warning(
                "HTTP error fetching model metrics for %s: %s", model_name, e
            )
            return ""
        return res.text

    def get_llama_swap_activity(self) -> list[dict]:
        """Get llama-swap activity with pagination."""
        url = urljoin(self.base_url, "/api/metrics/activity")
        all_items = []
        page = 1
        limit = 100
        while True:
            params = {"page": page, "limit": limit}
            res = self.session.get(url, params=params, timeout=self.timeout)
            try:
                res.raise_for_status()
            except requests.exceptions.HTTPError as e:
                logger.warning(
                    "HTTP error fetching llama-swap activity page %s: %s", page, e
                )
                break
            result = res.json()
            data = result.get("data", [])
            if not data:
                break
            all_items.extend(data)
            total_pages = result.get("total_pages", 1)
            if page >= total_pages:
                break
            page += 1
        return all_items

    def get_llama_swap_metrics(self) -> list[dict]:
        """Get latest metrics per model."""
        activity = self.get_llama_swap_activity()
        data_sorted = sorted(activity, key=itemgetter("model"))
        groups = itertools.groupby(data_sorted, key=itemgetter("model"))
        return [max(group, key=itemgetter("timestamp")) for _, group in groups]


class LlamaSwapCollector(Collector):
    """Prometheus collector for llama-swap."""

    def __init__(self, client: LlamaSwapApi) -> None:
        """Initialize collector."""
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
        draft_tokens_metric = self._make_metric(
            "llamaswap_model_draft_tokens", "Number of draft tokens."
        )
        draft_acc_tokens_metric = self._make_metric(
            "llamaswap_model_draft_acc_tokens", "Number of draft accepted tokens."
        )

        for entry in data:
            common_label = [entry["model"]]

            tokens = entry.get("tokens", {})
            cache_metric.add_metric(common_label, float(tokens.get("cache_tokens", 0)))
            input_metric.add_metric(common_label, float(tokens.get("input_tokens", 0)))
            output_metric.add_metric(
                common_label, float(tokens.get("output_tokens", 0))
            )
            prompt_pps_metric.add_metric(
                common_label, float(tokens.get("prompt_per_second", 0))
            )
            tokens_pps_metric.add_metric(
                common_label, float(tokens.get("tokens_per_second", 0))
            )
            duration_metric.add_metric(common_label, float(entry.get("duration_ms", 0)))
            draft_tokens_metric.add_metric(
                common_label, float(tokens.get("draft_tokens", 0))
            )
            draft_acc_tokens_metric.add_metric(
                common_label, float(tokens.get("draft_acc_tokens", 0))
            )

        return [
            cache_metric,
            input_metric,
            output_metric,
            prompt_pps_metric,
            tokens_pps_metric,
            duration_metric,
            draft_tokens_metric,
            draft_acc_tokens_metric,
        ]

    def parse_model_metrics(self, model_name: str, metrics: str) -> dict:
        """Parse model metrics."""
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

    def collect(self) -> list:
        """Collect metrics."""
        current_time = time.time()
        if current_time - self.last_scrape < REFRESH_INTERVAL:
            return list(self.cached_metrics)

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

        return list(self.cached_metrics)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/metrics":
            try:
                output = generate_latest(REGISTRY)
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(output)))
                self.end_headers()
                self.wfile.write(output)
            except Exception as e:
                logger.exception("Failed to generate metrics")
                self.send_error(500, f"Failed to generate metrics: {e}")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Llama-swap Exporter</h1><p>See <a href='/metrics'>/metrics</a></p></body></html>"
            )


def signal_handler(signum: int, _frame: object | None = None) -> None:
    """Terminate on signal."""
    signal_name = signal.Signals(signum).name
    logger.info("Terminating, %s signal received.", signal_name)
    time.sleep(0)
    raise SystemExit


def run_http_server() -> None:
    """Run HTTP server."""
    server = HTTPServer(("", EXPORTER_PORT), MetricsHandler)
    logger.info("Llama-swap Exporter listening on :%s", EXPORTER_PORT)
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
