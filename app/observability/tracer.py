"""OpenTelemetry distributed tracing instrumentation for PokéGrade AI.

Links end-to-end trace spans from the incoming collector request down through
the Coordinator, specialized Subagents (Inspector / Strategist), and individual
tool invocations (`analyze_card_condition_subgrades`, `query_tcg_market_pricing`,
`create_marketplace_listing`, `inspect_card_image_multimodal`).
"""

from __future__ import annotations

from contextlib import contextmanager
import functools
import inspect
import os
import threading
from typing import Any, Callable, Iterator, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

from app.observability.redact import redact_payload, redact_pii

F = TypeVar("F", bound=Callable[..., Any])


class _InMemorySpanCollector(SpanExporter):
    """Thread-safe in-memory OpenTelemetry SpanExporter for inspection and testing."""

    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Any) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


def build_cloud_trace_exporter(project_id: str | None = None) -> SpanExporter | None:
    """Instantiate a Google Cloud Trace exporter when `opentelemetry-exporter-gcp-trace` is installed.

    Args:
        project_id: Optional GCP project ID override.

    Returns:
        Configured `CloudTraceSpanExporter` instance, or None if unavailable.
    """
    try:
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter  # type: ignore[import-untyped]

        resolved_project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if resolved_project:
            return CloudTraceSpanExporter(project_id=resolved_project)
        return CloudTraceSpanExporter()
    except Exception:
        return None


_SPAN_COLLECTOR = _InMemorySpanCollector()
_PROVIDER = TracerProvider(
    resource=Resource.create(
        {
            "service.name": "pokegrade-ai-concierge",
            "service.version": "1.0.0",
            "deployment.environment": os.environ.get("ENVIRONMENT", "production"),
        }
    )
)
_PROVIDER.add_span_processor(SimpleSpanProcessor(_SPAN_COLLECTOR))

# Conditionally attach Google Cloud Trace exporter in production when enabled
if os.environ.get("ENABLE_CLOUD_TRACE_EXPORT", "false").lower() in ("1", "true", "yes"):
    _cloud_exporter = build_cloud_trace_exporter()
    if _cloud_exporter is not None:
        _PROVIDER.add_span_processor(BatchSpanProcessor(_cloud_exporter))

# Register as global tracer provider if not already overridden
try:
    trace.set_tracer_provider(_PROVIDER)
except Exception:
    pass

_TRACER = _PROVIDER.get_tracer("pokegrade_ai.tracer", "1.0.0")


@contextmanager
def start_trace_span(
    span_name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span]:
    """Start an OpenTelemetry span with automatic PII-scrubbed attributes.

    Args:
        span_name: Descriptive hierarchical span name (e.g. 'tool.analyze_card_condition_subgrades').
        attributes: Key-value metadata attached to the span after PII scrubbing.

    Yields:
        Active OpenTelemetry `Span` instance linked to the current context.
    """
    with _TRACER.start_as_current_span(span_name) as span:
        if attributes:
            sanitized = redact_payload(attributes)
            for key, value in sanitized.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(str(key), value)
                else:
                    span.set_attribute(str(key), str(value))
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, redact_pii(str(exc))))
            span.record_exception(exc)
            raise


def traced_operation(span_name: str, component: str = "agent") -> Callable[[F], F]:
    """Decorator wrapping synchronous or asynchronous functions in an OpenTelemetry span.

    Args:
        span_name: Operation identifier recorded on the span.
        component: Architectural layer ('coordinator', 'subagent', 'tool', 'memory').

    Returns:
        Decorated callable (sync or async) that emits linked OpenTelemetry spans on every call.
    """

    def decorator(func: F) -> F:
        span_attrs = {
            "pokegrade.component": component,
            "pokegrade.function": func.__name__,
        }
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with start_trace_span(span_name, attributes=span_attrs):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_trace_span(span_name, attributes=span_attrs):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def get_current_trace_id() -> str | None:
    """Return the 32-character hex OpenTelemetry trace_id of the active span, if any."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return None


def get_current_span_id() -> str | None:
    """Return the 16-character hex OpenTelemetry span_id of the active span, if any."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.span_id, "016x")
    return None


def get_recorded_spans() -> list[ReadableSpan]:
    """Return all finished OpenTelemetry spans captured in memory."""
    return _SPAN_COLLECTOR.get_finished_spans()


def clear_recorded_spans() -> None:
    """Clear all captured in-memory OpenTelemetry spans."""
    _SPAN_COLLECTOR.clear()
