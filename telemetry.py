"""Cloud-native observability helpers for traces, metrics, logs, and correlation IDs."""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from config import TelemetryConfig

try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    metrics = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment,misc]
    TracerProvider = None  # type: ignore[assignment,misc]
    BatchSpanProcessor = None  # type: ignore[assignment,misc]
    ConsoleSpanExporter = None  # type: ignore[assignment,misc]
    OTLPSpanExporter = None  # type: ignore[assignment,misc]
    MeterProvider = None  # type: ignore[assignment,misc]
    ConsoleMetricExporter = None  # type: ignore[assignment,misc]
    PeriodicExportingMetricReader = None  # type: ignore[assignment,misc]
    OTLPMetricExporter = None  # type: ignore[assignment,misc]
    OTEL_AVAILABLE = False


_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)
_tracer_provider_configured = False
_meter_provider_configured = False
_logging_configured = False
_metric_instruments: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class ExporterSelection:
    exporter: str
    endpoint: str | None


class JsonLogFormatter(logging.Formatter):
    """Small structured-log formatter with trace/span/correlation enrichment."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        span_context = get_current_span_context()
        if span_context:
            payload["trace_id"] = span_context["trace_id"]
            payload["span_id"] = span_context["span_id"]

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False)


def set_correlation_id(value: str) -> None:
    _correlation_id_var.set(value)


def reset_correlation_id() -> None:
    _correlation_id_var.set(None)


def get_correlation_id() -> str | None:
    return _correlation_id_var.get()


def get_or_create_correlation_id() -> str:
    existing = get_correlation_id()
    if existing:
        return existing
    created = str(uuid.uuid4())
    set_correlation_id(created)
    return created


def resolve_exporter_config(config: TelemetryConfig) -> ExporterSelection:
    exporter = (config.telemetry_exporter or "console").strip().lower()
    if exporter not in {"console", "otlp", "none"}:
        exporter = "console"
    endpoint = config.telemetry_otlp_endpoint if exporter == "otlp" else None
    return ExporterSelection(exporter=exporter, endpoint=endpoint)


def configure_observability(config: TelemetryConfig) -> None:
    """Configure tracing, metrics, and structured logs once per process."""

    if not config.telemetry_enabled:
        return

    configure_structured_logging(config)
    configure_tracer_provider(config)
    configure_meter_provider(config)


def configure_structured_logging(config: TelemetryConfig) -> None:
    """Enable JSON logs when configured. Existing handlers are reused."""

    global _logging_configured
    if _logging_configured or not config.structured_logs_enabled:
        return

    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler())
    for handler in root.handlers:
        handler.setFormatter(JsonLogFormatter())
    root.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    _logging_configured = True


def configure_tracer_provider(config: TelemetryConfig):
    """Configure OpenTelemetry traces with low-overhead batch exporting."""

    global _tracer_provider_configured
    if not config.telemetry_enabled or not OTEL_AVAILABLE or trace is None:
        return None

    if _tracer_provider_configured:
        return trace.get_tracer_provider()

    existing_provider = trace.get_tracer_provider()
    if isinstance(existing_provider, TracerProvider):
        _tracer_provider_configured = True
        return existing_provider

    selection = resolve_exporter_config(config)
    if selection.exporter == "none":
        return None

    provider = TracerProvider(resource=_resource(config))
    exporter = _build_span_exporter(selection)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer_provider_configured = True
    return provider


def configure_meter_provider(config: TelemetryConfig):
    """Configure OpenTelemetry metrics with periodic export."""

    global _meter_provider_configured
    if not config.telemetry_enabled or not config.metrics_enabled or not OTEL_AVAILABLE or metrics is None:
        return None

    if _meter_provider_configured:
        return metrics.get_meter_provider()

    selection = resolve_exporter_config(config)
    if selection.exporter == "none":
        return None

    exporter = _build_metric_exporter(selection)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=config.metric_export_interval_ms)
    provider = MeterProvider(resource=_resource(config), metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _meter_provider_configured = True
    return provider


def get_tracer(name: str = "rag-pipeline"):
    if not OTEL_AVAILABLE or trace is None:
        return None
    return trace.get_tracer(name)


def get_meter(name: str = "rag-pipeline"):
    if not OTEL_AVAILABLE or metrics is None:
        return None
    return metrics.get_meter(name)


def get_current_span_context() -> dict[str, str] | None:
    if not OTEL_AVAILABLE or trace is None:
        return None
    span = trace.get_current_span()
    if not span:
        return None
    context = span.get_span_context()
    if not context or not context.is_valid:
        return None
    return {
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
    }


@contextmanager
def start_span(name: str, attributes: dict[str, Any] | None = None, tracer_name: str = "rag-pipeline"):
    tracer = get_tracer(tracer_name)
    if not tracer:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            set_span_attributes(span, attributes)
        yield span


def set_span_attributes(span, attributes: dict[str, Any]) -> None:
    if not span:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def record_histogram(name: str, value: float, attributes: dict[str, Any] | None = None, unit: str = "1") -> None:
    instrument = _get_metric_instrument("histogram", name, unit)
    if instrument is not None:
        instrument.record(float(value), attributes or {})


def add_counter(name: str, value: int | float = 1, attributes: dict[str, Any] | None = None, unit: str = "1") -> None:
    instrument = _get_metric_instrument("counter", name, unit)
    if instrument is not None:
        instrument.add(value, attributes or {})


def observe_duration(name: str, start_time: float, attributes: dict[str, Any] | None = None) -> float:
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    record_histogram(name, elapsed_ms, attributes=attributes)
    return elapsed_ms


def span_context_or_null(name: str, attributes: dict[str, Any] | None = None, tracer_name: str = "rag-pipeline"):
    tracer = get_tracer(tracer_name)
    if tracer:
        return tracer.start_as_current_span(name)
    return nullcontext()


def _get_metric_instrument(kind: str, name: str, unit: str):
    meter = get_meter("rag.metrics")
    if not meter:
        return None

    key = (kind, name)
    if key in _metric_instruments:
        return _metric_instruments[key]

    if kind == "histogram":
        instrument = meter.create_histogram(name, unit=unit)
    elif kind == "counter":
        instrument = meter.create_counter(name, unit=unit)
    else:
        return None

    _metric_instruments[key] = instrument
    return instrument


def _resource(config: TelemetryConfig):
    attrs = {
        "service.name": config.telemetry_service_name,
        "service.version": config.service_version,
        "deployment.environment": config.environment,
    }
    return Resource.create(attrs)


def _build_span_exporter(selection: ExporterSelection):
    if selection.exporter == "otlp" and selection.endpoint:
        return OTLPSpanExporter(endpoint=_otlp_endpoint(selection.endpoint, "traces"))
    return ConsoleSpanExporter()


def _build_metric_exporter(selection: ExporterSelection):
    if selection.exporter == "otlp" and selection.endpoint:
        return OTLPMetricExporter(endpoint=_otlp_endpoint(selection.endpoint, "metrics"))
    return ConsoleMetricExporter()


def _otlp_endpoint(base: str, signal: str) -> str:
    base = base.rstrip("/")
    if base.endswith(f"/v1/{signal}"):
        return base
    if base.endswith("/v1/traces") or base.endswith("/v1/metrics"):
        return f"{base.rsplit('/v1/', 1)[0]}/v1/{signal}"
    return f"{base}/v1/{signal}"
