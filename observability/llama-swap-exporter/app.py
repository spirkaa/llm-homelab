"""Prometheus exporter for llama-swap."""

import logging
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urljoin, urlsplit

import requests
from dotenv import load_dotenv
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    Metric,
    UnknownMetricFamily,
)
from prometheus_client.parser import text_string_to_metric_families
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

ACTIVITY_PAGE_SIZE = 100
ACTIVITY_MAX_PAGES = 100

# metric name -> (field, field lives in the "tokens" sub-object, documentation)
GAUGE_SPECS: dict[str, tuple[str, bool, str]] = {
    "llamaswap_model_cache_tokens": (
        "cache_tokens",
        True,
        "Number of prompt tokens from cache.",
    ),
    "llamaswap_model_input_tokens": (
        "input_tokens",
        True,
        "Number of prompt tokens processed.",
    ),
    "llamaswap_model_output_tokens": (
        "output_tokens",
        True,
        "Number of generated tokens.",
    ),
    "llamaswap_model_prompt_per_second": (
        "prompt_per_second",
        True,
        "Prompt tokens per second.",
    ),
    "llamaswap_model_tokens_per_second": (
        "tokens_per_second",
        True,
        "Generated tokens per second.",
    ),
    "llamaswap_model_duration_ms": (
        "duration_ms",
        False,
        "Duration of the request in milliseconds.",
    ),
    "llamaswap_model_draft_tokens": (
        "draft_tokens",
        True,
        "Number of draft tokens.",
    ),
    "llamaswap_model_draft_acc_tokens": (
        "draft_acc_tokens",
        True,
        "Number of draft accepted tokens.",
    ),
}


class LlamaSwapApi:
    """Llama-swap API wrapper."""

    def __init__(self, base_url: str, timeout: int = 5) -> None:
        """Initialize API client."""
        # Normalize base URL to end with '/' for robust urljoin
        if not base_url.endswith("/"):
            base_url = base_url + "/"
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "llama-swap-exporter/0.1.0"})

    def get_running_models(self) -> list[dict]:
        """Get running models."""
        # HTTP errors intentionally propagate: a bad /running response fails
        # the whole scrape (500) so Prometheus can alert. Network errors are
        # treated as "no models" and swallowed.
        url = urljoin(self.base_url, "running")
        try:
            res = self.session.get(url, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
        except requests.exceptions.HTTPError:
            raise
        except requests.exceptions.JSONDecodeError as e:
            logger.warning("Invalid JSON fetching running models: %s", e)
            return []
        except requests.exceptions.RequestException as e:
            logger.warning("Network error fetching running models: %s", e)
            return []
        return data.get("running", [])

    def get_ready_models(self) -> list[dict]:
        """Get ready models."""
        models = self.get_running_models()
        return [model for model in models if model.get("state") == "ready"]

    def get_model_metrics(self, model_name: str) -> str:
        """Get model metrics."""
        url = urljoin(self.base_url, f"upstream/{model_name}/metrics")
        try:
            res = self.session.get(url, timeout=self.timeout)
            res.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(
                "HTTP error fetching model metrics for %s: %s", model_name, e
            )
            return ""
        return res.text

    def get_llama_swap_activity(self) -> list[dict]:
        """Get llama-swap activity with pagination."""
        url = urljoin(self.base_url, "api/metrics/activity")
        all_items = []
        page = 1
        while True:
            if page > ACTIVITY_MAX_PAGES:
                logger.warning("Pagination limit reached at page %s, stopping", page)
                break
            params = {"page": page, "limit": ACTIVITY_PAGE_SIZE}
            try:
                res = self.session.get(url, params=params, timeout=self.timeout)
                res.raise_for_status()
                result = res.json()
            except requests.exceptions.JSONDecodeError as e:
                logger.warning(
                    "Invalid JSON fetching llama-swap activity page %s: %s", page, e
                )
                break
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "HTTP error fetching llama-swap activity page %s: %s", page, e
                )
                break
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
        latest: dict[str, dict] = {}
        for item in self.get_llama_swap_activity():
            if not (isinstance(item, dict) and "model" in item and "timestamp" in item):
                continue
            model = item["model"]
            if model not in latest or item["timestamp"] > latest[model]["timestamp"]:
                latest[model] = item
        return list(latest.values())


class LlamaSwapCollector(Collector):
    """Prometheus collector for llama-swap."""

    def __init__(self, client: LlamaSwapApi, refresh_interval: int = 15) -> None:
        """Initialize collector."""
        self.client = client
        self.refresh_interval = refresh_interval
        self.last_scrape = 0
        self.cached_metrics = []

    def json_to_gauges(self, data: list[dict]) -> list[GaugeMetricFamily]:
        """Convert a list of JSON objects into GaugeMetricFamily objects."""
        families = {
            name: GaugeMetricFamily(name, documentation, labels=["model"])
            for name, (_, _, documentation) in GAUGE_SPECS.items()
        }
        for entry in data:
            model_name = entry.get("model")
            if not model_name:
                logger.warning("Skipping activity entry without 'model'")
                continue
            tokens = entry.get("tokens")
            if not isinstance(tokens, dict):
                tokens = {}
            for name, (field, in_tokens, _) in GAUGE_SPECS.items():
                source = tokens if in_tokens else entry
                families[name].add_metric([model_name], float(source.get(field) or 0))
        return list(families.values())

    def parse_model_metrics(self, model_name: str, metrics: str) -> dict[str, Metric]:
        """Parse model metrics."""
        result: dict[str, Metric] = {}
        try:
            for family in text_string_to_metric_families(metrics):
                name = family.name.replace(":", "_", 1)
                typ = family.type
                if typ in METRIC_TYPES:
                    metric_cls = METRIC_TYPES[typ]
                elif typ == "unknown":
                    metric_cls = UnknownMetricFamily
                else:
                    logger.warning(
                        "Skipping metric %s with unsupported type %s for model %s",
                        family.name,
                        typ,
                        model_name,
                    )
                    continue

                # Union of label names across all samples, plus our model label
                label_names = tuple(
                    sorted({ln for s in family.samples for ln in s.labels} | {"model"})
                )
                mf = metric_cls(name, family.documentation, labels=list(label_names))
                for s in family.samples:
                    labels = dict(s.labels)
                    labels["model"] = model_name
                    label_values = tuple(labels.get(ln, "") for ln in label_names)
                    mf.add_metric(label_values, s.value)
                result[name] = mf
        except Exception:
            logger.exception("Failed to parse model metrics for %s", model_name)
            return {}
        return result

    def collect(self) -> list[Metric]:
        """Collect metrics."""
        current_time = time.time()
        if current_time - self.last_scrape < self.refresh_interval:
            return list(self.cached_metrics)

        try:
            new_metrics = []
            models = self.client.get_ready_models()
            for model in models:
                model_name = model.get("model")
                if not model_name:
                    logger.warning("Skipping ready model entry without 'model' key")
                    continue
                model_metrics = self.client.get_model_metrics(model_name)
                result = self.parse_model_metrics(model_name, model_metrics)
                new_metrics.extend(result.values())

            swap_metrics = self.client.get_llama_swap_metrics()
            swap_metrics_families = self.json_to_gauges(swap_metrics)
            new_metrics.extend(swap_metrics_families)

            # Update cache on successful scrape
            self.cached_metrics = new_metrics
            self.last_scrape = current_time
            return list(self.cached_metrics)
        except Exception:
            # Keep last good cache, update last_scrape to throttle retries
            logger.exception("Failed to collect metrics, keeping last good cache")
            self.last_scrape = current_time
            raise


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for metrics and health endpoints."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        path = urlsplit(self.path).path
        if path == "/metrics":
            self._handle_metrics()
        elif path == "/health":
            self._handle_health()
        else:
            self.send_error(404, "Not Found")

    def _handle_metrics(self) -> None:
        """Serve the Prometheus metrics endpoint."""
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

    def _handle_health(self) -> None:
        """Serve the health endpoint."""
        body = b"OK"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        """Route request logs through the standard logger instead of stderr."""
        logger.info(fmt, *args)


def signal_handler(signum: int, _frame: object | None = None) -> None:
    """Terminate on signal."""
    signal_name = signal.Signals(signum).name
    logger.info("Terminating, %s signal received.", signal_name)
    raise SystemExit


def create_server() -> HTTPServer:
    """Create the metrics HTTP server."""
    return HTTPServer(("", EXPORTER_PORT), MetricsHandler)


def run_http_server() -> None:
    """Run HTTP server."""
    server = create_server()
    logger.info("Llama-swap Exporter listening on :%s", server.server_address[1])
    server.serve_forever()


def main() -> None:
    """Execute main function."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    client = LlamaSwapApi(LLAMA_SWAP_BASE_URL)
    collector = LlamaSwapCollector(client, refresh_interval=REFRESH_INTERVAL)
    REGISTRY.register(collector)

    run_http_server()


if __name__ == "__main__":
    main()
