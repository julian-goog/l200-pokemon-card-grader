"""Non-blocking asynchronous memory consolidation worker for PokéGrade AI.

Uses `asyncio.create_task` to execute heavy memory operations—such as deterministic
embedding vector extraction, portfolio valuation indexing, and long-term collector
profile consolidation—in the background so the main agent response is never blocked.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
from typing import Any

from app.observability.logger import get_structured_logger
from app.observability.redact import redact_payload
from app.observability.tracer import start_trace_span

logger = get_structured_logger("pokegrade_ai.memory.async_worker")


class AsyncMemoryWorker:
    """Background asynchronous memory indexer and embedding consolidator."""

    def __init__(self, embedding_dim: int = 16) -> None:
        """Initialize the asynchronous memory worker.

        Args:
            embedding_dim: Dimensionality of the compact semantic card vector representation.
        """
        self.embedding_dim = embedding_dim
        self._indexed_records: list[dict[str, Any]] = []
        self._pending_tasks: set[asyncio.Task[dict[str, Any]]] = set()

    def schedule_card_indexing(
        self,
        session_id: str,
        card_evaluation: dict[str, Any],
    ) -> asyncio.Task[dict[str, Any]] | dict[str, Any]:
        """Schedule background indexing of an evaluated card without blocking the caller.

        When an asyncio event loop is running, dispatches `_consolidate_card_memory` via
        `asyncio.create_task` immediately and registers a cleanup callback. If called
        outside an active event loop, runs via `asyncio.run`.

        Args:
            session_id: Collector session ID associated with the evaluation.
            card_evaluation: Structured subgrade and pricing payload to index.

        Returns:
            `asyncio.Task` handle when an event loop is active, or the indexed record dict.
        """
        sanitized = redact_payload(card_evaluation)
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._consolidate_card_memory(session_id=session_id, card_evaluation=sanitized),
                name=f"pokegrade-memory-index-{session_id}",
            )
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            return task
        except RuntimeError:
            # No running event loop in synchronous caller context
            return asyncio.run(
                self._consolidate_card_memory(session_id=session_id, card_evaluation=sanitized)
            )

    async def _consolidate_card_memory(
        self,
        session_id: str,
        card_evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        """Asynchronously compute semantic card embeddings and update portfolio index.

        Args:
            session_id: Collector session identifier.
            card_evaluation: PII-scrubbed card evaluation dictionary.

        Returns:
            Consolidated vector memory record stored in the index.
        """
        with start_trace_span(
            "memory.async_consolidate_card",
            attributes={
                "session.id": session_id,
                "card.name": str(card_evaluation.get("card_name", "unknown")),
            },
        ):
            # Yield control to event loop so foreground chat turn returns immediately
            await asyncio.sleep(0.005)

            card_text = (
                f"{card_evaluation.get('card_name', '')}|"
                f"{card_evaluation.get('set_name', '')}|"
                f"{card_evaluation.get('condition_tier', '')}|"
                f"{card_evaluation.get('estimated_grade', '')}"
            )
            embedding = self._compute_embedding(card_text)

            record = {
                "session_id": session_id,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "card_name": card_evaluation.get("card_name", "Unknown Card"),
                "set_name": card_evaluation.get("set_name", "Unknown Set"),
                "estimated_grade": card_evaluation.get("estimated_grade"),
                "condition_tier": card_evaluation.get("condition_tier"),
                "net_grading_roi_usd": card_evaluation.get("net_grading_roi_usd"),
                "should_submit_for_grading": card_evaluation.get("should_submit_for_grading"),
                "embedding": embedding,
            }
            self._indexed_records.append(record)

            logger.info(
                f"Background memory consolidation completed for '{record['card_name']}'.",
                extra={
                    "event_type": "ASYNC_MEMORY_INDEXED",
                    "structured_data": {
                        "session_id": session_id,
                        "card_name": record["card_name"],
                        "estimated_grade": record["estimated_grade"],
                        "embedding_dim": len(embedding),
                    },
                },
            )
            return record

    async def flush_pending_tasks(self) -> list[dict[str, Any]]:
        """Await all currently scheduled background memory tasks (useful in tests/shutdown)."""
        if not self._pending_tasks:
            return []
        tasks = list(self._pending_tasks)
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    def get_indexed_records(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Return all background-indexed card records, optionally filtered by `session_id`."""
        if session_id is None:
            return list(self._indexed_records)
        return [r for r in self._indexed_records if r.get("session_id") == session_id]

    def _compute_embedding(self, text: str) -> list[float]:
        """Generate a normalized deterministic float vector from card attributes."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw_vals = [((b / 127.5) - 1.0) for b in digest[: self.embedding_dim]]
        norm = sum(v * v for v in raw_vals) ** 0.5 or 1.0
        return [round(v / norm, 4) for v in raw_vals]
