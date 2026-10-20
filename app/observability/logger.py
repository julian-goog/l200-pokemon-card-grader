"""Structured JSON logging and Intent-vs-Outcome auditing for PokéGrade AI.

Provides machine-readable JSON log formatting with automatic PII scrubbing and
dedicated `log_intent` / `log_outcome` capture functions to audit agent intent
before execution against verified outcomes after execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
import threading
import uuid
from typing import Any

from app.observability.redact import redact_payload, redact_pii
from app.observability.tracer import get_current_span_id, get_current_trace_id

# Thread-safe in-memory audit buffer for test inspection and compliance verification
_INTENT_OUTCOME_AUDIT_LOG: list[dict[str, Any]] = []
_AUDIT_LOCK = threading.Lock()


class StructuredJSONFormatter(logging.Formatter):
    """Formats Python LogRecords as single-line structured JSON with PII redaction."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a LogRecord into a machine-readable JSON string.

        Args:
            record: Standard Python logging.LogRecord instance.

        Returns:
            Single-line JSON string with redacted message and structured fields.
        """
        resolved_trace_id = getattr(record, "trace_id", None) or get_current_trace_id()
        resolved_span_id = getattr(record, "span_id", None) or get_current_span_id()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": redact_pii(record.getMessage()),
            "event_type": getattr(record, "event_type", "GENERAL"),
            "agent_name": getattr(record, "agent_name", "pokegrade_system"),
            "trace_id": resolved_trace_id,
            "span_id": resolved_span_id,
        }

        extra_data = getattr(record, "structured_data", None)
        if extra_data is not None:
            payload["data"] = redact_payload(extra_data)

        if record.exc_info:
            payload["exception"] = redact_pii(self.formatException(record.exc_info))

        return json.dumps(payload, default=str)


def get_structured_logger(name: str = "pokegrade_ai") -> logging.Logger:
    """Return a logger configured with `StructuredJSONFormatter`.

    Args:
        name: Logger namespace identifier.

    Returns:
        Configured `logging.Logger` instance emitting structured JSON.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_intent(
    agent_name: str,
    action: str,
    target_tool: str,
    arguments: dict[str, Any],
    rationale: str = "",
    trace_id: str | None = None,
    span_id: str | None = None,
) -> dict[str, Any]:
    """Explicitly log an agent's intended action prior to tool or subagent execution.

    Args:
        agent_name: Name of the coordinator or subagent initiating the action.
        action: High-level semantic description of the intended operation.
        target_tool: Specific tool function or subagent being invoked.
        arguments: Input arguments prepared for the invocation.
        rationale: Why the agent selected this action/tool.
        trace_id: Optional OpenTelemetry trace ID linking the turn.
        span_id: Optional OpenTelemetry span ID linking the active span.

    Returns:
        Structured intent audit record including the generated `intent_id`.
    """
    intent_id = f"intent-{uuid.uuid4().hex[:12]}"
    resolved_trace_id = trace_id or get_current_trace_id()
    resolved_span_id = span_id or get_current_span_id()
    sanitized_args = redact_payload(arguments)
    record = {
        "intent_id": intent_id,
        "phase": "PRE_EXECUTION_INTENT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_name": agent_name,
        "action": redact_pii(action),
        "target_tool": target_tool,
        "arguments": sanitized_args,
        "rationale": redact_pii(rationale),
        "trace_id": resolved_trace_id,
        "span_id": resolved_span_id,
    }
    with _AUDIT_LOCK:
        _INTENT_OUTCOME_AUDIT_LOG.append(record)

    logger = get_structured_logger("pokegrade_ai.audit")
    logger.info(
        f"Intent registered: {agent_name} -> {target_tool} ({action})",
        extra={
            "event_type": "AGENT_INTENT",
            "agent_name": agent_name,
            "trace_id": resolved_trace_id,
            "span_id": resolved_span_id,
            "structured_data": record,
        },
    )
    return record


def log_outcome(
    agent_name: str,
    action: str,
    target_tool: str,
    status: str,
    result_summary: dict[str, Any],
    latency_ms: float = 0.0,
    intent_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> dict[str, Any]:
    """Explicitly log the verified outcome after tool or subagent execution completes.

    Args:
        agent_name: Name of the coordinator or subagent that executed the action.
        action: Semantic description matching the prior intent.
        target_tool: Specific tool function or subagent that completed.
        status: Execution outcome status (e.g., 'SUCCESS', 'BLOCKED_HITL', 'GUIDED_ERROR').
        result_summary: Sanitized summary of the returned payload.
        latency_ms: Execution duration in milliseconds.
        intent_id: Correlation ID returned by `log_intent`.
        trace_id: Optional OpenTelemetry trace ID linking the turn.
        span_id: Optional OpenTelemetry span ID linking the active span.

    Returns:
        Structured outcome audit record linked to `intent_id`.
    """
    resolved_trace_id = trace_id or get_current_trace_id()
    resolved_span_id = span_id or get_current_span_id()
    sanitized_result = redact_payload(result_summary)
    record = {
        "outcome_id": f"outcome-{uuid.uuid4().hex[:12]}",
        "intent_id": intent_id,
        "phase": "POST_EXECUTION_OUTCOME",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_name": agent_name,
        "action": redact_pii(action),
        "target_tool": target_tool,
        "status": status,
        "latency_ms": round(latency_ms, 3),
        "result_summary": sanitized_result,
        "trace_id": resolved_trace_id,
        "span_id": resolved_span_id,
    }
    with _AUDIT_LOCK:
        _INTENT_OUTCOME_AUDIT_LOG.append(record)

    logger = get_structured_logger("pokegrade_ai.audit")
    logger.info(
        f"Outcome verified: {agent_name} <- {target_tool} [{status}]",
        extra={
            "event_type": "AGENT_OUTCOME",
            "agent_name": agent_name,
            "trace_id": resolved_trace_id,
            "span_id": resolved_span_id,
            "structured_data": record,
        },
    )
    return record


def get_audit_trail() -> list[dict[str, Any]]:
    """Return a copy of all captured intent and outcome audit records."""
    with _AUDIT_LOCK:
        return list(_INTENT_OUTCOME_AUDIT_LOG)


def clear_audit_trail() -> None:
    """Clear the in-memory intent/outcome audit buffer (used in test fixtures)."""
    with _AUDIT_LOCK:
        _INTENT_OUTCOME_AUDIT_LOG.clear()
