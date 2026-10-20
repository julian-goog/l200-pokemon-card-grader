"""Comprehensive pytest unit test suite for PokéGrade AI (95-Point AgentOps Matrix).

Verifies all 19 criteria across:
1. Tool & Interface Design (docstrings, naming, Pydantic schemas, guided error recovery)
2. Context & Memory (constitution, sliding-window compaction, persistent state, async worker)
3. Orchestration & Logic (multi-agent coordinator, Flash/Pro routing, guardrails, HITL gate)
4. Observability & Tracing (JSON logging, intent/outcome audit, OpenTelemetry spans, PII redaction)
5. Infrastructure & CI/CD (Golden dataset evaluation suite, Terraform IaC, Secret Manager)
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from app.agent import (
    PokeGradeCoordinator,
    adk_app,
    root_agent,
    sequential_pipeline_agent,
)
from app.guardrails.policy_filter import (
    PolicyGuardrailPlugin,
    build_vertex_safety_settings,
)
from app.memory.async_memory import AsyncMemoryWorker
from app.memory.session_store import (
    PersistentSessionStore,
    build_adk_events_compaction_config,
    build_gcp_context_cache_config,
)
from app.observability.logger import (
    StructuredJSONFormatter,
    clear_audit_trail,
    get_audit_trail,
)
from app.observability.redact import CloudDLPRedactor, redact_payload, redact_pii
from app.observability.tracer import clear_recorded_spans, get_recorded_spans
from app.prompts.constitution import SYSTEM_CONSTITUTION
from app.subagents.inspector_agent import INSPECTOR_MODEL
from app.subagents.strategist_agent import STRATEGIST_MODEL
from app.tools.card_tools import (
    analyze_card_condition_subgrades,
    create_marketplace_listing,
    query_tcg_market_pricing,
)
from app.tools.schemas import Condition, GradingCompany, Rarity
from app.tools.secret_manager import SecretManagerService
from app.tools.vision_inspector import (
    generate_sample_card_image_bytes,
    inspect_card_image_multimodal,
)
from tests.eval.eval_harness import run_evaluation_suite


def test_tool_docstrings_and_descriptive_naming() -> None:
    """Criterion 1.1 & 1.2: Verify Google-style docstrings and unambiguous tool names."""
    tools = [
        analyze_card_condition_subgrades,
        query_tcg_market_pricing,
        create_marketplace_listing,
        inspect_card_image_multimodal,
    ]
    for tool_fn in tools:
        assert tool_fn.__doc__ is not None
        assert "Args:" in tool_fn.__doc__
        assert "Returns:" in tool_fn.__doc__
        assert "WHEN TO USE:" in tool_fn.__doc__
        assert "WHEN NOT TO USE:" in tool_fn.__doc__
        assert len(tool_fn.__name__) > 15


def test_pydantic_subgrade_calculation_and_conservative_cap() -> None:
    """Criterion 1.3: Verify strict Pydantic schemas and conservative limiting-subgrade cap."""
    # High centering/corners/edges (9.5), but surface is 6.0 -> grade must be capped at 6.5
    result = analyze_card_condition_subgrades(
        card_name="Charizard Holo",
        set_name="Base Set 1999",
        centering_score=9.5,
        corners_score=9.5,
        edges_score=9.5,
        surface_score=6.0,
        card_number="4/102",
        rarity=Rarity.HOLO_RARE,
        grading_company=GradingCompany.PSA,
    )
    assert result["status"] == "SUCCESS"
    assert result["estimated_grade"] == 6.5
    assert result["condition_tier"] == Condition.EXCELLENT_MINT.value
    assert "surface" in result["limiting_subgrade_factor"]


def test_guided_error_handling_on_invalid_tool_inputs() -> None:
    """Criterion 1.4: Verify defensive try/except returns structured guided_recovery instructions."""
    invalid_subgrades = analyze_card_condition_subgrades(
        card_name="Charizard Holo",
        set_name="Base Set 1999",
        centering_score=14.5,  # Invalid (> 10.0)
        corners_score=9.0,
        edges_score=9.0,
        surface_score=9.0,
    )
    assert invalid_subgrades["status"] == "ERROR"
    assert invalid_subgrades["error_code"] == "INVALID_SUBGRADE_PARAMETERS"
    assert "guided_recovery" in invalid_subgrades
    assert "1.0 and 10.0" in invalid_subgrades["guided_recovery"]

    invalid_pricing = query_tcg_market_pricing(
        card_name="Pikachu",
        set_name="Jungle",
        estimated_grade=0.2,  # Invalid (< 1.0)
    )
    assert invalid_pricing["status"] == "ERROR"
    assert "guided_recovery" in invalid_pricing


def test_system_constitution_and_sliding_window_compaction(tmp_path: Path) -> None:
    """Criterion 2.1, 2.2 & 2.3: Verify constitution, persistent vector DB, ADK compaction & context cache."""
    assert "Four-Pillar Subgrade Standard" in SYSTEM_CONSTITUTION
    assert "HUMAN-OK-LIST-" in SYSTEM_CONSTITUTION

    storage_file = tmp_path / "sessions.json"
    sqlite_db = tmp_path / "sessions.sqlite3"
    store = PersistentSessionStore(
        max_window_turns=3,
        keep_recent_turns=2,
        storage_path=storage_file,
        sqlite_db_path=sqlite_db,
    )

    for idx in range(1, 5):
        store.append_turn(
            session_id="collector-42",
            user_message=f"Evaluate card #{idx} Charizard Base Set",
            agent_response=f"Card #{idx} graded at 9.0",
            card_metadata={
                "card_name": f"Pokemon Card #{idx}",
                "estimated_grade": 9.0,
                "condition_tier": "MINT",
                "raw_ungraded_usd": 100.0,
            },
        )

    context = store.build_context_window("collector-42", current_query="Charizard Base Set")
    assert context["compaction_count"] >= 1
    assert len(context["recent_turns"]) == 2
    assert "Compacted Portfolio History:" in context["portfolio_summary"]
    assert "Pokemon Card #1" in context["portfolio_summary"]
    assert context["total_cards_evaluated"] == 4
    assert len(context["vector_retrieved_turns"]) >= 1
    assert context["vector_retrieved_turns"][0]["source"] == "sqlite_vector_store"
    assert storage_file.exists()
    assert sqlite_db.exists()

    # Verify ADK EventsCompactionConfig and Google Cloud ContextCacheConfig
    compaction_cfg = build_adk_events_compaction_config(compaction_interval=4, overlap_size=2)
    cache_cfg = build_gcp_context_cache_config(min_tokens=1024, ttl_seconds=1800)
    assert compaction_cfg is not None
    assert cache_cfg is not None


def test_async_memory_worker_non_blocking_indexing() -> None:
    """Criterion 2.4: Verify AsyncMemoryWorker uses asyncio.create_task in background."""

    async def _run_async_check() -> None:
        worker = AsyncMemoryWorker(embedding_dim=16)
        task = worker.schedule_card_indexing(
            session_id="async-session-1",
            card_evaluation={
                "card_name": "Lugia Neo Genesis Holo",
                "set_name": "Neo Genesis",
                "estimated_grade": 9.5,
                "condition_tier": "GEM_MINT",
                "net_grading_roi_usd": 420.0,
                "should_submit_for_grading": True,
            },
        )
        assert isinstance(task, asyncio.Task)
        flushed = await worker.flush_pending_tasks()
        assert len(flushed) == 1
        records = worker.get_indexed_records("async-session-1")
        assert len(records) == 1
        assert len(records[0]["embedding"]) == 16

    asyncio.run(_run_async_check())


def test_dynamic_model_routing_and_multi_agent_orchestration() -> None:
    """Criterion 3.1 & 3.2: Verify Coordinator, ADK SequentialAgent, and Flash/Pro routing."""
    coordinator = PokeGradeCoordinator()

    # Verify ADK SequentialAgent, Coordinator Agent (all 3 routing modes), and App instances
    assert sequential_pipeline_agent is not None
    assert sequential_pipeline_agent.name == "pokegrade_sequential_appraisal_pipeline"
    assert len(sequential_pipeline_agent.sub_agents) == 2
    assert root_agent is not None
    assert len(root_agent.sub_agents) == 3
    assert adk_app is not None

    # Verify modern Gemini 3.x model routing constants
    assert INSPECTOR_MODEL == "gemini-3.8-flash"
    assert STRATEGIST_MODEL == "gemini-3.1-pro-preview"

    # Triage-only query -> routes only to InspectorSubagent (gemini-3.8-flash)
    triage_route = coordinator.classify_and_route(
        user_query="Run a quick subgrade condition only triage on my card.",
        has_subgrades=True,
        request_market_or_listing=False,
    )
    assert triage_route["route_mode"] == "INSPECTOR_ONLY"
    assert triage_route["routed_models"] == [INSPECTOR_MODEL]

    # End-to-end query -> routes sequentially to Flash then Pro
    full_route = coordinator.classify_and_route(
        user_query="Inspect subgrades and calculate PSA grading ROI and market value.",
        has_subgrades=True,
        request_market_or_listing=True,
    )
    assert full_route["route_mode"] == "FULL_PIPELINE"
    assert full_route["routed_models"] == [INSPECTOR_MODEL, STRATEGIST_MODEL]


def test_guardrails_and_human_in_the_loop_checkpoint() -> None:
    """Criterion 3.3 & 3.4: Verify ADK BasePlugin, Vertex SafetySettings, reflection & HITL gate."""
    from google.adk.plugins import BasePlugin

    guardrails = PolicyGuardrailPlugin()
    assert isinstance(guardrails, BasePlugin)
    assert hasattr(guardrails, "before_tool_callback")
    assert hasattr(guardrails, "after_tool_callback")
    safety_settings = build_vertex_safety_settings()
    assert len(safety_settings) >= 3

    blocked = guardrails.validate_input_policy("Please list my counterfeit proxy Charizard")
    assert blocked["allowed"] is False
    assert blocked["policy_action"] == "BLOCK"

    # Negated authenticity assertion should be permitted
    allowed_negated = guardrails.validate_input_policy(
        "Grade my authentic 1999 Charizard — confirmed not a counterfeit and no trimmed edges."
    )
    assert allowed_negated["allowed"] is True
    assert allowed_negated["policy_action"] == "ALLOW"

    # Without approval token -> blocked by HITL gate
    unapproved = create_marketplace_listing(
        card_name="Blastoise Base Set Holo",
        set_name="Base Set 1999",
        condition=Condition.MINT,
        estimated_grade=9.0,
        listing_price_usd=250.0,
        seller_contact="ash@pokemon.com",
        human_approval_token=None,
    )
    assert unapproved["status"] == "BLOCKED_HITL_REQUIRED"
    assert unapproved["human_verified"] is False
    assert "HUMAN-OK-LIST-" in unapproved["guided_recovery"]

    # With valid approval token -> published
    approved = create_marketplace_listing(
        card_name="Blastoise Base Set Holo",
        set_name="Base Set 1999",
        condition=Condition.MINT,
        estimated_grade=9.0,
        listing_price_usd=250.0,
        seller_contact="ash@pokemon.com",
        human_approval_token="HUMAN-OK-LIST-77123",
    )
    assert approved["status"] == "PUBLISHED"
    assert approved["human_verified"] is True
    assert approved["listing_id"].startswith("LIST-")
    assert approved["sanitized_seller_contact"] == "[REDACTED_EMAIL_PII]"


def test_observability_json_logging_intent_outcome_tracing_and_pii_redaction() -> None:
    """Criterion 4.1, 4.2, 4.3 & 4.4: Verify JSON logs, intent/outcome, OTel spans, and Cloud DLP/regex PII scrubbing."""
    from app.observability.tracer import traced_operation

    clear_audit_trail()
    clear_recorded_spans()

    dlp_redactor = CloudDLPRedactor(project_id="pokegrade-test-project")
    assert dlp_redactor.project_id == "pokegrade-test-project"

    synthetic_test_token = "sk-" + "live9876543210abcdef1234"
    raw_sensitive = (
        "Ship to 1234 Elm Street, Suite 5B, PayPal buyer@collector.org, "
        f"Card 4532-0151-1283-0049, Token {synthetic_test_token}"
    )
    scrubbed = redact_pii(raw_sensitive)
    assert "[REDACTED_SHIPPING_ADDRESS]" in scrubbed
    assert "[REDACTED_EMAIL_PII]" in scrubbed
    assert "[REDACTED_CREDIT_CARD]" in scrubbed
    assert "[REDACTED_API_TOKEN]" in scrubbed
    assert "1234 Elm Street" not in scrubbed
    assert "buyer@collector.org" not in scrubbed

    formatter = StructuredJSONFormatter()
    record = logging.LogRecord(
        name="pokegrade_test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg=raw_sensitive,
        args=(),
        exc_info=None,
    )
    formatted_json = json.loads(formatter.format(record))
    assert formatted_json["severity"] == "INFO"
    assert "buyer@collector.org" not in formatted_json["message"]

    @traced_operation("custom.traced_sync_check", component="test")
    def _sample_traced_fn(val: int) -> int:
        return val * 2

    assert _sample_traced_fn(21) == 42

    coordinator = PokeGradeCoordinator()
    res = coordinator.handle_collector_request(
        session_id="obs-test-session",
        user_query="Grade my Charizard Holo and check PSA market ROI.",
        card_name="Charizard Holo",
        set_name="Base Set 1999",
        centering_score=9.5,
        corners_score=9.5,
        edges_score=9.0,
        surface_score=9.5,
    )
    assert res["status"] == "SUCCESS"

    audit_records = get_audit_trail()
    phases = {entry["phase"] for entry in audit_records}
    assert "PRE_EXECUTION_INTENT" in phases
    assert "POST_EXECUTION_OUTCOME" in phases
    assert all(entry.get("trace_id") is not None for entry in audit_records)
    assert all(entry.get("span_id") is not None for entry in audit_records)

    spans = get_recorded_spans()
    span_names = {s.name for s in spans}
    assert "custom.traced_sync_check" in span_names
    assert "coordinator.handle_collector_request" in span_names
    assert "subagent.card_inspector" in span_names
    assert "tool.analyze_card_condition_subgrades" in span_names
    assert "tool.query_tcg_market_pricing" in span_names


def test_secret_manager_terraform_agents_cli_and_golden_evaluation_suite(monkeypatch) -> None:
    """Criterion 5.1, 5.2 & 5.3: Verify Secret Manager, Terraform, agents-cli docs, and Golden Eval Suite."""
    synthetic_tcg_key = "tcg_live_" + "secret_token_99887766"
    monkeypatch.setenv("TCG_MARKET_API_KEY", synthetic_tcg_key)
    sm = SecretManagerService(project_id="pokegrade-test-project")
    resolved = sm.get_secret("TCG_MARKET_API_KEY")
    assert resolved == synthetic_tcg_key
    assert redact_payload({"api_key": resolved})["api_key"] == "[REDACTED_API_TOKEN]"

    # Verify Terraform files exist and define required resources (including Vertex AI Search & Vector Search & DLP)
    repo_root = Path(__file__).resolve().parent.parent
    tf_main = repo_root / "terraform" / "main.tf"
    assert tf_main.exists()
    tf_content = tf_main.read_text(encoding="utf-8")
    assert "google_service_account" in tf_content
    assert "google_storage_bucket" in tf_content
    assert "google_secret_manager_secret" in tf_content
    assert "google_discovery_engine_data_store" in tf_content
    assert "google_vertex_ai_index" in tf_content
    assert "dlp.googleapis.com" in tf_content

    # Verify agents-cli usage is documented in README.md
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "agents-cli" in readme_text
    assert "agents-cli deploy" in readme_text
    assert "agents-cli eval run" in readme_text

    # Run Golden Dataset Evaluation Harness
    eval_summary = run_evaluation_suite()
    assert eval_summary["total_cases"] == 7
    assert eval_summary["pass_rate"] == 1.0
    assert eval_summary["routing_accuracy"] == 1.0
    assert eval_summary["grading_accuracy"] == 1.0


def test_multimodal_vision_inspector_and_psa_10_ceiling() -> None:
    """Verify picture-only multimodal card extraction and PSA 10 ceiling pricing."""
    sample_png = generate_sample_card_image_bytes("Charizard Base Set Holo (Gem Mint)")
    assert len(sample_png) > 500

    vision_res = inspect_card_image_multimodal(
        image_bytes=sample_png,
        mime_type="image/png",
        filename_hint="Charizard Base Set Holo (Gem Mint).png",
        use_live_vertex_vision=False,
    )
    assert vision_res["status"] == "SUCCESS"
    assert vision_res["card_name"] == "Charizard Holo"
    assert 1.0 <= vision_res["centering_score"] <= 10.0
    assert 1.0 <= vision_res["estimated_psa_grade"] <= 10.0

    # Verify guided recovery when empty image bytes are supplied
    empty_res = inspect_card_image_multimodal(image_bytes=b"", mime_type="image/png")
    assert empty_res["status"] == "ERROR"
    assert "guided_recovery" in empty_res

    # Verify PSA 10 ceiling calculation in query_tcg_market_pricing
    pricing_res = query_tcg_market_pricing(
        card_name="Charizard Holo",
        set_name="Base Set 1999",
        estimated_grade=9.0,
        rarity=Rarity.HOLO_RARE,
        grading_company=GradingCompany.PSA,
    )
    assert pricing_res["status"] == "SUCCESS"
    assert pricing_res["psa_10_ceiling_usd"] > pricing_res["graded_value_usd"]

    # Verify multi-thousand-dollar Gold Star / Grail card pricing (zero manual input required)
    grail_pricing = query_tcg_market_pricing(
        card_name="Gyarados Star Delta Species",
        set_name="EX Holon Phantoms",
        estimated_grade=6.5,
        rarity=Rarity.ULTRA_RARE,
        grading_company=GradingCompany.PSA,
    )
    assert grail_pricing["status"] == "SUCCESS"
    assert grail_pricing["raw_ungraded_usd"] >= 1500.0
    assert grail_pricing["graded_value_usd"] >= 2000.0
    assert grail_pricing["psa_10_ceiling_usd"] >= 10000.0
    assert grail_pricing["should_submit_for_grading"] is True

    # Verify negated crease notes (e.g. "no distinct surface creases") do NOT cap grade at 5.5
    no_crease_subgrades = analyze_card_condition_subgrades(
        card_name="Gyarados Star Delta Species",
        set_name="EX Holon Phantoms",
        centering_score=9.0,
        corners_score=8.5,
        edges_score=8.5,
        surface_score=8.0,
        surface_notes="Glossy finish inside clear sleeve showing no distinct surface creases or severe scratching.",
    )
    assert no_crease_subgrades["status"] == "SUCCESS"
    assert no_crease_subgrades["estimated_grade"] >= 8.0
    assert "Structural surface defect" not in no_crease_subgrades["inspector_notes"]

    # Verify drag-and-drop Base64 data URI resolution and sequential camera capture digests
    import base64
    import hashlib
    from app.ui import _resolve_dropped_or_url_image_payload

    b64_uri = b"data:image/png;base64," + base64.b64encode(sample_png)
    resolved_bytes, _ = _resolve_dropped_or_url_image_payload(b64_uri, "dropped_uri.png")
    assert resolved_bytes == sample_png

    cam_shot_1 = generate_sample_card_image_bytes("Charizard Holo", "Base Set 1999")
    cam_shot_2 = generate_sample_card_image_bytes("Umbreon VMAX Alt Art", "Evolving Skies")
    assert hashlib.sha256(cam_shot_1).hexdigest() != hashlib.sha256(cam_shot_2).hexdigest()
