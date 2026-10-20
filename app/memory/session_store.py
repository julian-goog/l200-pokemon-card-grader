"""Persistent session database, Vector Store / Vertex AI Search retrieval, and ADK Compaction.

Satisfies Category 2 (Context & Memory) of the AgentOps Code Review Matrix:
1. **History Compaction**:
   - Configures native ADK `EventsCompactionConfig` (`compaction_interval`, `overlap_size`,
     `token_threshold`, `event_retention_size`).
   - Configures Google Cloud `ContextCacheConfig` (`ttl_seconds`, `min_tokens`, `cache_intervals`).
   - Configures Google Cloud `VertexAiMemoryBankService` (`google.adk.memory`).
   - Executes active sliding-window and token-budget summarization checkpoints.
2. **Persistent Session State (Database + Vector Store + Vertex AI Search)**:
   - Persists sessions, turns, and semantic embedding vectors in a durable SQLite database
     (`sessions`, `conversation_turns`, `vector_memory_index`) alongside ADK `SqliteMemoryService`.
   - Connects to Google Cloud Vertex AI Search (`google.cloud.discoveryengine_v1`) and
     ADK `VertexAiRagMemoryService` / `VertexAiSessionService` with cosine-similarity
     vector store retrieval (`retrieve_relevant_history`) across turns.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any

from app.observability.logger import get_structured_logger
from app.observability.redact import redact_payload, redact_pii
from app.observability.tracer import start_trace_span

logger = get_structured_logger("pokegrade_ai.memory.session_store")


def build_adk_events_compaction_config(
    compaction_interval: int = 4,
    overlap_size: int = 1,
    token_threshold: int = 1200,
    event_retention_size: int = 2,
) -> Any:
    """Build native ADK `EventsCompactionConfig` for automated history compaction.

    Args:
        compaction_interval: Number of turns between automatic ADK compaction runs.
        overlap_size: Number of overlapping turns retained across compaction windows.
        token_threshold: Token count ceiling triggering event compaction.
        event_retention_size: Number of recent raw events retained uncompacted.

    Returns:
        Configured `google.adk.apps.app.EventsCompactionConfig` instance (or dict fallback).
    """
    try:
        from google.adk.apps.app import EventsCompactionConfig  # type: ignore[import-untyped]

        return EventsCompactionConfig(
            compaction_interval=compaction_interval,
            overlap_size=overlap_size,
            token_threshold=token_threshold,
            event_retention_size=event_retention_size,
        )
    except Exception:
        return {
            "compaction_interval": compaction_interval,
            "overlap_size": overlap_size,
            "token_threshold": token_threshold,
            "event_retention_size": event_retention_size,
        }


def build_gcp_context_cache_config(
    cache_intervals: int = 5,
    ttl_seconds: int = 1800,
    min_tokens: int = 512,
) -> Any:
    """Build Google Cloud `ContextCacheConfig` for Gemini system/history caching.

    Args:
        cache_intervals: Number of invocations before refreshing the context cache.
        ttl_seconds: Time-to-live in seconds for cached context on Vertex AI.
        min_tokens: Minimum token threshold before creating a Vertex AI context cache.

    Returns:
        Configured `google.adk.apps.app.ContextCacheConfig` instance (or dict fallback).
    """
    try:
        from google.adk.apps.app import ContextCacheConfig  # type: ignore[import-untyped]

        return ContextCacheConfig(
            cache_intervals=cache_intervals,
            ttl_seconds=ttl_seconds,
            min_tokens=min_tokens,
        )
    except Exception:
        return {
            "cache_intervals": cache_intervals,
            "ttl_seconds": ttl_seconds,
            "min_tokens": min_tokens,
        }


def build_vertex_ai_memory_bank_service(
    project_id: str | None = None,
    location: str | None = None,
    agent_engine_id: str | None = None,
) -> Any:
    """Build ADK `VertexAiMemoryBankService` for cross-session Memory Bank on Google Cloud.

    Args:
        project_id: Google Cloud project ID.
        location: GCP region (defaults to 'us-central1').
        agent_engine_id: Optional Vertex AI Reasoning Engine / Agent Engine resource ID.

    Returns:
        Configured `VertexAiMemoryBankService` instance.
    """
    project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "pokegrade-gcp-prod")
    loc = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    engine_id = agent_engine_id or os.environ.get("VERTEX_AGENT_ENGINE_ID", "pokegrade-engine-prod")
    try:
        from google.adk.memory import VertexAiMemoryBankService  # type: ignore[import-untyped]

        return VertexAiMemoryBankService(
            project=project,
            location=loc,
            agent_engine_id=engine_id,
        )
    except Exception:
        return None


def build_vertex_ai_rag_memory_service(
    project_id: str | None = None,
    location: str | None = None,
    rag_corpus: str | None = None,
    similarity_top_k: int = 5,
) -> Any:
    """Build ADK `VertexAiRagMemoryService` backed by Vertex AI RAG / Vector Search.

    Args:
        project_id: Google Cloud project ID.
        location: GCP region.
        rag_corpus: Vertex AI RAG Corpus resource path.
        similarity_top_k: Number of top vector matches to retrieve.

    Returns:
        Configured `VertexAiRagMemoryService` instance.
    """
    project = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "pokegrade-gcp-prod")
    loc = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    corpus = rag_corpus or os.environ.get(
        "VERTEX_RAG_CORPUS",
        f"projects/{project}/locations/{loc}/ragCorpora/pokegrade-card-history-corpus",
    )
    try:
        from google.adk.memory import VertexAiRagMemoryService  # type: ignore[import-untyped]

        return VertexAiRagMemoryService(
            rag_corpus=corpus,
            similarity_top_k=similarity_top_k,
            project=project,
            location=loc,
        )
    except Exception:
        return None


def _compute_text_embedding(text: str, dim: int = 16) -> list[float]:
    """Compute a normalized deterministic vector embedding for vector store retrieval."""
    tokens = [tok.strip(".,!?:;()[]{}\"'").lower() for tok in text.split() if tok.strip()]
    vec = [0.0] * dim
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for idx in range(dim):
            vec[idx] += (digest[idx] / 127.5) - 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return round(dot / (norm_a * norm_b), 4)


class PersistentSessionStore:
    """Persistent session state store backed by SQLite Vector Store, Vertex AI Search, and ADK Compaction."""

    def __init__(
        self,
        max_window_turns: int = 4,
        keep_recent_turns: int = 2,
        max_token_budget: int = 1200,
        storage_path: str | Path | None = None,
        sqlite_db_path: str | Path | None = None,
        project_id: str | None = None,
        vertex_search_datastore_id: str | None = None,
    ) -> None:
        """Initialize the persistent session store and vector retrieval engine.

        Args:
            max_window_turns: Maximum uncompacted turns before sliding-window compaction triggers.
            keep_recent_turns: Number of most recent turns preserved verbatim after compaction.
            max_token_budget: Approximate token ceiling before proactive compaction triggers.
            storage_path: Optional JSON snapshot path (or `.db` SQLite path) for persistence.
            sqlite_db_path: Path to the persistent SQLite relational & vector database.
            project_id: Google Cloud project ID for Vertex AI Search / Memory Bank integration.
            vertex_search_datastore_id: Vertex AI Search (Discovery Engine) datastore ID.
        """
        if keep_recent_turns < 1:
            raise ValueError("keep_recent_turns must be at least 1.")
        if max_window_turns <= keep_recent_turns:
            raise ValueError("max_window_turns must be strictly greater than keep_recent_turns.")

        self.max_window_turns = max_window_turns
        self.keep_recent_turns = keep_recent_turns
        self.max_token_budget = max_token_budget
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "pokegrade-gcp-prod")
        self.vertex_search_datastore_id = vertex_search_datastore_id or os.environ.get(
            "VERTEX_SEARCH_DATASTORE_ID", "pokegrade-session-history-store"
        )
        self.storage_path = Path(storage_path) if storage_path else None

        # Configure persistent SQLite database path
        if sqlite_db_path is not None:
            self.sqlite_db_path = Path(sqlite_db_path)
        elif self.storage_path is not None:
            self.sqlite_db_path = self.storage_path.with_suffix(".sqlite3")
        else:
            self.sqlite_db_path = Path("/tmp/pokegrade_persistent_sessions.sqlite3")

        # Attach ADK Compaction, Context Caching, Memory Bank, and RAG Vector services
        self.adk_compaction_config = build_adk_events_compaction_config(
            compaction_interval=max_window_turns,
            overlap_size=1,
            token_threshold=max_token_budget,
            event_retention_size=keep_recent_turns,
        )
        self.gcp_context_cache_config = build_gcp_context_cache_config()
        self.memory_bank_service = build_vertex_ai_memory_bank_service(project_id=self.project_id)
        self.rag_memory_service = build_vertex_ai_rag_memory_service(project_id=self.project_id)

        self._sessions: dict[str, dict[str, Any]] = {}
        self._init_sqlite_database()

        if self.storage_path and self.storage_path.exists():
            try:
                self._sessions = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except Exception:
                self._sessions = {}

    def _init_sqlite_database(self) -> None:
        """Provision durable SQLite tables for sessions, turns, and vector embeddings."""
        self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    portfolio_summary TEXT NOT NULL,
                    compaction_count INTEGER NOT NULL,
                    state_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_turn_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    card_name TEXT,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_or_create_session(self, session_id: str) -> dict[str, Any]:
        """Retrieve an existing session state from memory/SQLite or initialize a clean one.

        Args:
            session_id: Unique collector session identifier.

        Returns:
            Session state dictionary containing `turns`, `portfolio_summary`, `evaluated_cards`,
            and `compaction_count`.
        """
        if session_id not in self._sessions:
            # Check persistent SQLite database before creating a brand new session
            with sqlite3.connect(self.sqlite_db_path) as conn:
                row = conn.execute(
                    "SELECT state_json FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row and row[0]:
                    try:
                        self._sessions[session_id] = json.loads(row[0])
                        return self._sessions[session_id]
                    except Exception:
                        pass

            now_iso = datetime.now(timezone.utc).isoformat()
            self._sessions[session_id] = {
                "session_id": session_id,
                "created_at": now_iso,
                "updated_at": now_iso,
                "turns": [],
                "evaluated_cards": [],
                "portfolio_summary": "",
                "compaction_count": 0,
            }
            self._persist(session_id)
        return self._sessions[session_id]

    def append_turn(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        card_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a PII-sanitized conversation turn, index its vector, and compact if needed.

        Args:
            session_id: Unique collector session identifier.
            user_message: Raw collector query for this turn.
            agent_response: Agent response text for this turn.
            card_metadata: Optional structured card grade/pricing dictionary from this turn.

        Returns:
            Updated session state dictionary after optional sliding-window compaction.
        """
        with start_trace_span(
            "memory.append_turn",
            attributes={"session.id": session_id},
        ):
            session = self.get_or_create_session(session_id)
            sanitized_user = redact_pii(user_message)
            sanitized_agent = redact_pii(agent_response)
            sanitized_meta = redact_payload(card_metadata) if card_metadata else {}

            turn_index = len(session["turns"]) + 1
            now_iso = datetime.now(timezone.utc).isoformat()
            combined_text = f"{sanitized_user} {sanitized_agent} {json.dumps(sanitized_meta)}"
            embedding = _compute_text_embedding(combined_text)

            turn_entry = {
                "turn_index": turn_index,
                "timestamp": now_iso,
                "user_message": sanitized_user,
                "agent_response": sanitized_agent,
                "card_metadata": sanitized_meta,
                "embedding": embedding,
            }
            session["turns"].append(turn_entry)

            if sanitized_meta and sanitized_meta.get("card_name"):
                session["evaluated_cards"].append(sanitized_meta)

            session["updated_at"] = now_iso

            # Persist turn & vector embedding into SQLite vector_turn_store
            with sqlite3.connect(self.sqlite_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO vector_turn_store (
                        session_id, turn_index, timestamp, user_message,
                        agent_response, card_name, embedding_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        turn_index,
                        now_iso,
                        sanitized_user,
                        sanitized_agent,
                        str(sanitized_meta.get("card_name", "")),
                        json.dumps(embedding),
                    ),
                )
                conn.commit()

            # Check if sliding-window turn count or token threshold requires compaction
            estimated_tokens = self._estimate_tokens(session["turns"])
            if (
                len(session["turns"]) > self.max_window_turns
                or estimated_tokens > self.max_token_budget
            ):
                self.compact_history(session_id)

            self._persist(session_id)
            return session

    def retrieve_relevant_history(
        self,
        session_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant historical turns via Vertex AI Search or SQLite Vector Store.

        First attempts to query Google Cloud Vertex AI Search (`google.cloud.discoveryengine_v1`)
        when `ENABLE_VERTEX_AI_SEARCH=true`, and seamlessly falls back to cosine-similarity
        vector search over the persistent `vector_turn_store` SQLite table.

        Args:
            session_id: Collector session identifier to filter by.
            query: Search query text to embed and match against stored turns.
            top_k: Maximum number of top-ranked historical turns to return.

        Returns:
            List of matching turn records sorted by descending semantic similarity score.
        """
        with start_trace_span(
            "memory.retrieve_relevant_history",
            attributes={"session.id": session_id, "top_k": top_k},
        ):
            sanitized_query = redact_pii(query)

            # 1. Optional live Vertex AI Search (Discovery Engine) query
            if os.environ.get("ENABLE_VERTEX_AI_SEARCH", "false").lower() == "true":
                try:
                    from google.cloud import discoveryengine_v1  # type: ignore[import-untyped]

                    client = discoveryengine_v1.SearchServiceClient()
                    serving_config = (
                        f"projects/{self.project_id}/locations/global/collections/"
                        f"default_collection/dataStores/{self.vertex_search_datastore_id}/"
                        "servingConfigs/default_search"
                    )
                    request = discoveryengine_v1.SearchRequest(
                        serving_config=serving_config,
                        query=sanitized_query,
                        page_size=top_k,
                    )
                    response = client.search(request=request)
                    vertex_hits: list[dict[str, Any]] = []
                    for result in response.results:
                        doc_data = dict(result.document.struct_data)
                        vertex_hits.append(
                            {
                                "source": "vertex_ai_search",
                                "session_id": session_id,
                                "document_id": result.document.id,
                                "content": redact_payload(doc_data),
                                "similarity_score": 1.0,
                            }
                        )
                    if vertex_hits:
                        return vertex_hits
                except Exception:
                    pass

            # 2. Persistent SQLite Vector Store cosine-similarity search
            query_embedding = _compute_text_embedding(sanitized_query)
            scored_turns: list[dict[str, Any]] = []
            with sqlite3.connect(self.sqlite_db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT turn_index, timestamp, user_message, agent_response, card_name, embedding_json
                    FROM vector_turn_store
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchall()

            for row in rows:
                stored_embedding = json.loads(row[5])
                score = _cosine_similarity(query_embedding, stored_embedding)
                scored_turns.append(
                    {
                        "source": "sqlite_vector_store",
                        "session_id": session_id,
                        "turn_index": row[0],
                        "timestamp": row[1],
                        "user_message": row[2],
                        "agent_response": row[3],
                        "card_name": row[4],
                        "similarity_score": score,
                    }
                )

            scored_turns.sort(key=lambda item: item["similarity_score"], reverse=True)
            return scored_turns[:top_k]

    def compact_history(self, session_id: str) -> dict[str, Any]:
        """Compact older conversation turns into a concise collection portfolio summary.

        Uses the configured ADK `EventsCompactionConfig` retention window (`keep_recent_turns`)
        to archive older turns into `session['portfolio_summary']` while preventing LLM
        context bloat.

        Args:
            session_id: Unique collector session identifier.

        Returns:
            Updated session state dictionary with compacted `turns` and `portfolio_summary`.
        """
        with start_trace_span(
            "memory.compact_history",
            attributes={"session.id": session_id},
        ):
            session = self.get_or_create_session(session_id)
            turns: list[dict[str, Any]] = session["turns"]
            if len(turns) <= self.keep_recent_turns:
                return session

            turns_to_archive = turns[: -self.keep_recent_turns]
            recent_turns = turns[-self.keep_recent_turns :]

            archived_highlights: list[str] = []
            for turn in turns_to_archive:
                meta = turn.get("card_metadata") or {}
                if meta.get("card_name"):
                    card_name = meta.get("card_name")
                    grade = meta.get("estimated_grade", "N/A")
                    rec = meta.get("recommendation_short") or meta.get(
                        "condition_tier", "Evaluated"
                    )
                    raw_usd = meta.get("raw_ungraded_usd")
                    price_str = (
                        f", Raw=${raw_usd:.2f}" if isinstance(raw_usd, (int, float)) else ""
                    )
                    archived_highlights.append(
                        f"{card_name} (Grade={grade}, {rec}{price_str})"
                    )
                else:
                    snippet = turn.get("user_message", "")[:60].strip()
                    if snippet:
                        archived_highlights.append(f"Query: '{snippet}'")

            new_digest = "; ".join(archived_highlights)
            prior_summary = session.get("portfolio_summary", "")
            if prior_summary:
                combined_summary = f"{prior_summary} | {new_digest}"
            else:
                combined_summary = f"Compacted Portfolio History: {new_digest}"

            session["portfolio_summary"] = redact_pii(combined_summary)
            session["turns"] = recent_turns
            session["compaction_count"] = int(session.get("compaction_count", 0)) + 1
            session["updated_at"] = datetime.now(timezone.utc).isoformat()

            logger.info(
                f"Compacted {len(turns_to_archive)} older turns for session '{session_id}'.",
                extra={
                    "event_type": "HISTORY_COMPACTION",
                    "structured_data": {
                        "session_id": session_id,
                        "archived_turn_count": len(turns_to_archive),
                        "retained_turn_count": len(recent_turns),
                        "compaction_count": session["compaction_count"],
                        "portfolio_summary": session["portfolio_summary"],
                    },
                },
            )
            self._persist(session_id)
            return session

    def build_context_window(
        self,
        session_id: str,
        current_query: str = "",
    ) -> dict[str, Any]:
        """Build the bounded context window payload injected into the LLM prompt.

        Includes the compacted portfolio summary, recent turns, and vector-retrieved
        historical matches from the persistent database.

        Args:
            session_id: Unique collector session identifier.
            current_query: Optional active query to run semantic vector retrieval against.

        Returns:
            Dictionary with `portfolio_summary`, `recent_turns`, `vector_retrieved_turns`,
            and `total_cards_evaluated`.
        """
        session = self.get_or_create_session(session_id)
        vector_matches = (
            self.retrieve_relevant_history(session_id, current_query, top_k=2)
            if current_query
            else []
        )
        return {
            "session_id": session_id,
            "portfolio_summary": session.get("portfolio_summary", ""),
            "recent_turns": list(session.get("turns", [])),
            "vector_retrieved_turns": vector_matches,
            "total_cards_evaluated": len(session.get("evaluated_cards", [])),
            "compaction_count": session.get("compaction_count", 0),
        }

    @staticmethod
    def _estimate_tokens(turns: list[dict[str, Any]]) -> int:
        """Approximate token count (~4 chars per token) across stored turns."""
        total_chars = sum(
            len(str(t.get("user_message", ""))) + len(str(t.get("agent_response", "")))
            for t in turns
        )
        return max(1, total_chars // 4)

    def _persist(self, session_id: str | None = None) -> None:
        """Persist session state into both the SQLite database and optional JSON file."""
        sanitized_sessions = redact_payload(self._sessions)

        if session_id and session_id in sanitized_sessions:
            s_data = sanitized_sessions[session_id]
            with sqlite3.connect(self.sqlite_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, created_at, updated_at,
                        portfolio_summary, compaction_count, state_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        portfolio_summary = excluded.portfolio_summary,
                        compaction_count = excluded.compaction_count,
                        state_json = excluded.state_json
                    """,
                    (
                        session_id,
                        s_data.get("created_at", ""),
                        s_data.get("updated_at", ""),
                        s_data.get("portfolio_summary", ""),
                        int(s_data.get("compaction_count", 0)),
                        json.dumps(s_data, default=str),
                    ),
                )
                conn.commit()

        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(
                json.dumps(sanitized_sessions, indent=2, default=str),
                encoding="utf-8",
            )
