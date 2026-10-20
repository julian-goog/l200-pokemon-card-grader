"""Observability package for PokéGrade AI (JSON logging, OpenTelemetry tracing, PII redaction)."""

from app.observability.logger import (
    StructuredJSONFormatter,
    clear_audit_trail,
    get_audit_trail,
    get_structured_logger,
    log_intent,
    log_outcome,
)
from app.observability.redact import redact_payload, redact_pii
from app.observability.tracer import (
    clear_recorded_spans,
    get_current_trace_id,
    get_recorded_spans,
    start_trace_span,
    traced_operation,
)

__all__ = [
    "StructuredJSONFormatter",
    "clear_audit_trail",
    "clear_recorded_spans",
    "get_audit_trail",
    "get_current_trace_id",
    "get_recorded_spans",
    "get_structured_logger",
    "log_intent",
    "log_outcome",
    "redact_payload",
    "redact_pii",
    "start_trace_span",
    "traced_operation",
]
