"""Telemetry helpers for correlation IDs and tracing bootstrap."""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass

from config import TelemetryConfig

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency path
    trace = None
    Resource = None
    TracerProvider = None
    ConsoleSpanExporter = None
    SimpleSpanProcessor = None
    OTLPSpanExporter = None
    OTEL_AVAILABLE = False


_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


@dataclass(frozen=True)
class ExporterSelection:
    exporter: str
    endpoint: str | None


def set_correlation_id(value: str) -> None:
    _correlation_id_var.set(value)


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
    if exporter not in {"console", "otlp"}:
        exporter = "console"
    endpoint = config.telemetry_otlp_endpoint if exporter == "otlp" else None
    return ExporterSelection(exporter=exporter, endpoint=endpoint)


def configure_tracer_provider(config: TelemetryConfig):
    """Configure OpenTelemetry tracer provider when dependencies are available."""

    if not config.telemetry_enabled or not OTEL_AVAILABLE:
        return None

    selection = resolve_exporter_config(config)
    provider = TracerProvider(
        resource=Resource.create({"service.name": config.telemetry_service_name})
    )

    if selection.exporter == "otlp" and selection.endpoint:
        exporter = OTLPSpanExporter(endpoint=selection.endpoint)
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "rag-pipeline"):
    if not OTEL_AVAILABLE or trace is None:
        return None
    return trace.get_tracer(name)
