"""Master PokéGradeCoordinator orchestrating multi-agent card grading and market strategy.

Implements:
1. Coordinator/Supervisor + Specialized Subagents (`InspectorSubagent` & `StrategistSubagent`).
2. Dynamic Strategic Model Routing (`gemini-3.8-flash` for rapid physical subgrade triage
   vs. `gemini-3.1-pro-preview` for deep market valuation, fee breakeven, and HITL listings).
3. Pre-execution policy guardrails and post-generation self-critique reflection hooks.
4. Multi-turn `PersistentSessionStore` with sliding-window history compaction.
5. Non-blocking `AsyncMemoryWorker` background card indexing via `asyncio.create_task`.
"""

from __future__ import annotations

import os
import time
from typing import Any

from app.tools.secret_manager import DEFAULT_SECRET_MANAGER

# Configure Vertex AI global endpoint defaults required by Gemini 3.x models in ADK
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", DEFAULT_SECRET_MANAGER.get_project_id())
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from app.guardrails.policy_filter import (
    PolicyGuardrailPlugin,
    build_vertex_safety_settings,
)
from app.memory.async_memory import AsyncMemoryWorker
from app.memory.session_store import (
    PersistentSessionStore,
    build_adk_events_compaction_config,
    build_gcp_context_cache_config,
    build_vertex_ai_memory_bank_service,
    build_vertex_ai_rag_memory_service,
)
from app.observability.logger import log_intent, log_outcome
from app.observability.redact import redact_pii
from app.observability.tracer import get_current_trace_id, start_trace_span
from app.prompts.constitution import SYSTEM_CONSTITUTION
from app.subagents.inspector_agent import (
    INSPECTOR_MODEL,
    InspectorSubagent,
    build_adk_inspector_agent,
    inspector_adk_agent,
)
from app.subagents.strategist_agent import (
    STRATEGIST_MODEL,
    StrategistSubagent,
    build_adk_strategist_agent,
    strategist_adk_agent,
)
from app.tools.schemas import Condition, GradingCompany, Rarity


class PokeGradeCoordinator:
    """Supervisor/Coordinator orchestrating PokéGrade AI subagents, guardrails, and memory."""

    def __init__(
        self,
        session_store: PersistentSessionStore | None = None,
        async_memory: AsyncMemoryWorker | None = None,
        guardrail_plugin: PolicyGuardrailPlugin | None = None,
    ) -> None:
        """Initialize the PokéGradeCoordinator and its subagents.

        Args:
            session_store: Persistent multi-turn session store with sliding-window compaction.
            async_memory: Background asyncio memory indexer.
            guardrail_plugin: Pre-execution policy filter and post-generation reflection scorer.
        """
        self.name = "pokegrade_coordinator"
        self.system_constitution = SYSTEM_CONSTITUTION
        self.inspector = InspectorSubagent(model_name=INSPECTOR_MODEL)
        self.strategist = StrategistSubagent(model_name=STRATEGIST_MODEL)
        self.session_store = session_store or PersistentSessionStore()
        self.async_memory = async_memory or AsyncMemoryWorker()
        self.guardrails = guardrail_plugin or PolicyGuardrailPlugin()
        self.vertex_safety_settings = build_vertex_safety_settings()
        self.memory_bank_service = build_vertex_ai_memory_bank_service()
        self.rag_memory_service = build_vertex_ai_rag_memory_service()

    def classify_and_route(
        self,
        user_query: str,
        has_subgrades: bool = True,
        request_market_or_listing: bool = True,
    ) -> dict[str, Any]:
        """Determine the optimal subagent and Gemini model tier for a collector request.

        Routes rapid physical condition triage exclusively to `gemini-3.8-flash`
        (`InspectorSubagent`), pure pricing/listing queries to `gemini-3.1-pro-preview`
        (`StrategistSubagent`), and end-to-end card appraisal workflows sequentially
        through Flash -> Pro.

        Args:
            user_query: Natural-language prompt from the collector.
            has_subgrades: True if physical subgrade scores are supplied.
            request_market_or_listing: True if pricing, ROI, or listing strategy is needed.

        Returns:
            Routing decision dictionary containing `route_mode`, `routed_models`, and
            `target_subagents`.
        """
        q_lower = user_query.lower()
        wants_pricing_or_listing = request_market_or_listing or any(
            kw in q_lower
            for kw in (
                "price",
                "pricing",
                "worth",
                "value",
                "roi",
                "breakeven",
                "submit",
                "list",
                "sell",
                "market",
            )
        )
        wants_only_condition = (
            has_subgrades
            and not wants_pricing_or_listing
            and any(kw in q_lower for kw in ("subgrade", "centering", "condition only", "triage"))
        )

        if wants_only_condition:
            return {
                "route_mode": "INSPECTOR_ONLY",
                "routed_models": [INSPECTOR_MODEL],
                "target_subagents": [self.inspector.name],
                "routing_rationale": (
                    "Rapid physical subgrade triage routed to low-latency gemini-3.8-flash."
                ),
            }

        if not has_subgrades and wants_pricing_or_listing:
            return {
                "route_mode": "STRATEGIST_ONLY",
                "routed_models": [STRATEGIST_MODEL],
                "target_subagents": [self.strategist.name],
                "routing_rationale": (
                    "Direct market valuation and HITL listing routed to high-reasoning gemini-3.1-pro-preview."
                ),
            }

        return {
            "route_mode": "FULL_PIPELINE",
            "routed_models": [INSPECTOR_MODEL, STRATEGIST_MODEL],
            "target_subagents": [self.inspector.name, self.strategist.name],
            "routing_rationale": (
                "Sequential multi-agent orchestration: gemini-3.8-flash for physical subgrade "
                "triage followed by gemini-3.1-pro-preview for TCG ROI economics and HITL strategy."
            ),
        }

    def handle_collector_request(
        self,
        session_id: str,
        user_query: str,
        card_name: str,
        set_name: str,
        centering_score: float | None = 9.5,
        corners_score: float | None = 9.5,
        edges_score: float | None = 9.5,
        surface_score: float | None = 9.5,
        card_number: str = "001/102",
        rarity: str | Rarity = Rarity.HOLO_RARE,
        grading_company: str | GradingCompany = GradingCompany.PSA,
        surface_notes: str = "",
        pre_estimated_grade: float | None = None,
        request_market_analysis: bool = True,
        request_listing: bool = False,
        proposed_listing_price_usd: float | None = None,
        seller_contact: str = "",
        human_approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute an end-to-end collector turn with tracing, guardrails, routing, and memory.

        Args:
            session_id: Persistent collector session ID.
            user_query: Collector's natural-language request.
            card_name: Official Pokémon card name.
            set_name: Expansion set name.
            centering_score: Front/back centering subgrade (1.0–10.0), or None if already graded.
            corners_score: Corners subgrade (1.0–10.0), or None if already graded.
            edges_score: Edges subgrade (1.0–10.0), or None if already graded.
            surface_score: Surface subgrade (1.0–10.0), or None if already graded.
            card_number: Collector set number.
            rarity: Card rarity tier (`Rarity` enum or string).
            grading_company: Target grading company ('PSA', 'BGS', or 'CGC').
            surface_notes: Optional notes on surface scratches or creases.
            pre_estimated_grade: Optional known grade when skipping physical inspection.
            request_market_analysis: Whether to invoke the `StrategistSubagent` (`gemini-3.1-pro-preview`).
            request_listing: Whether the user wants to publish a marketplace listing.
            proposed_listing_price_usd: Optional custom asking price in USD.
            seller_contact: Optional seller contact info (scrubbed for PII).
            human_approval_token: Optional HITL confirmation token ('HUMAN-OK-LIST-...').

        Returns:
            Comprehensive orchestration dictionary containing `status`, `routing`,
            `policy_check`, `inspector_output`, `strategist_output`, `reflection`,
            `session_context`, and `response_text`.
        """
        start_ts = time.perf_counter()

        with start_trace_span(
            "coordinator.handle_collector_request",
            attributes={
                "session.id": session_id,
                "card.name": card_name,
                "card.set": set_name,
                "request.listing": request_listing,
            },
        ):
            trace_id = get_current_trace_id()
            intent = log_intent(
                agent_name=self.name,
                action="Orchestrate multi-agent Pokémon card appraisal and marketplace workflow",
                target_tool="handle_collector_request",
                arguments={
                    "session_id": session_id,
                    "user_query": redact_pii(user_query),
                    "card_name": card_name,
                    "set_name": set_name,
                    "request_listing": request_listing,
                },
                rationale="Coordinate policy check, dynamic Flash/Pro model routing, reflection, and memory.",
                trace_id=trace_id,
            )

            # Step 1: Pre-Execution Policy Guardrail Check
            policy_check = self.guardrails.validate_input_policy(
                f"{user_query} {card_name} {surface_notes}"
            )
            if not policy_check["allowed"]:
                blocked_response = policy_check["violation_reason"]
                self.session_store.append_turn(
                    session_id=session_id,
                    user_message=user_query,
                    agent_response=blocked_response,
                    card_metadata={"card_name": card_name, "status": "BLOCKED_POLICY"},
                )
                outcome_payload = {
                    "status": "BLOCKED_POLICY",
                    "trace_id": trace_id,
                    "policy_check": policy_check,
                    "routing": None,
                    "inspector_output": None,
                    "strategist_output": None,
                    "reflection": None,
                    "session_context": self.session_store.build_context_window(
                        session_id, current_query=user_query
                    ),
                    "response_text": blocked_response,
                }
                log_outcome(
                    agent_name=self.name,
                    action="Orchestrate multi-agent Pokémon card appraisal and marketplace workflow",
                    target_tool="handle_collector_request",
                    status="BLOCKED_POLICY",
                    result_summary={"matched_term": policy_check["matched_term"]},
                    latency_ms=(time.perf_counter() - start_ts) * 1000,
                    intent_id=intent["intent_id"],
                    trace_id=trace_id,
                )
                return outcome_payload

            # Step 2: Strategic Model Routing Decision
            has_subgrades = all(
                score is not None
                for score in (centering_score, corners_score, edges_score, surface_score)
            ) and pre_estimated_grade is None
            routing = self.classify_and_route(
                user_query=user_query,
                has_subgrades=has_subgrades,
                request_market_or_listing=(request_market_analysis or request_listing),
            )

            inspector_output: dict[str, Any] | None = None
            strategist_output: dict[str, Any] | None = None
            subgrade_data: dict[str, Any] | None = None
            pricing_data: dict[str, Any] | None = None
            listing_data: dict[str, Any] | None = None

            # Step 3: Execute Inspector Subagent (gemini-3.8-flash) if routed
            if self.inspector.name in routing["target_subagents"]:
                inspector_output = self.inspector.inspect_card(
                    card_name=card_name,
                    set_name=set_name,
                    centering_score=float(centering_score if centering_score is not None else 9.0),
                    corners_score=float(corners_score if corners_score is not None else 9.0),
                    edges_score=float(edges_score if edges_score is not None else 9.0),
                    surface_score=float(surface_score if surface_score is not None else 9.0),
                    card_number=card_number,
                    rarity=rarity,
                    grading_company=grading_company,
                    surface_notes=surface_notes,
                )
                subgrade_data = inspector_output.get("subgrade_analysis", {})

            # Determine grade & condition to pass to Strategist Subagent (gemini-3.1-pro-preview)
            if subgrade_data and subgrade_data.get("status") == "SUCCESS":
                effective_grade = float(subgrade_data["estimated_grade"])
                effective_condition = subgrade_data["condition_tier"]
            else:
                effective_grade = float(pre_estimated_grade if pre_estimated_grade is not None else 8.0)
                effective_condition = Condition.NEAR_MINT.value

            # Step 4: Execute Strategist Subagent (gemini-3.1-pro-preview) if routed
            if self.strategist.name in routing["target_subagents"] and (
                subgrade_data is None or subgrade_data.get("status") == "SUCCESS"
            ):
                strategist_output = self.strategist.evaluate_market_strategy(
                    card_name=card_name,
                    set_name=set_name,
                    estimated_grade=effective_grade,
                    condition=effective_condition,
                    rarity=rarity,
                    grading_company=grading_company,
                    request_listing=request_listing,
                    proposed_listing_price_usd=proposed_listing_price_usd,
                    seller_contact=seller_contact,
                    human_approval_token=human_approval_token,
                )
                pricing_data = strategist_output.get("market_pricing")
                listing_data = strategist_output.get("marketplace_listing")

            # Step 5: Synthesize Draft Response & Run Self-Critique Reflection Hook
            draft_response = self._compose_summary_response(
                card_name=card_name,
                set_name=set_name,
                routing=routing,
                subgrade_data=subgrade_data,
                pricing_data=pricing_data,
                listing_data=listing_data,
            )
            reflection = self.guardrails.evaluate_output_reflection(
                response_text=draft_response,
                subgrade_result=subgrade_data,
                pricing_result=pricing_data,
                listing_result=listing_data,
            )
            final_response = reflection["sanitized_response"]

            # Step 6: Update Persistent Session Store (with sliding-window compaction & vector index)
            card_turn_metadata = {
                "card_name": card_name,
                "set_name": set_name,
                "estimated_grade": effective_grade,
                "condition_tier": effective_condition,
                "raw_ungraded_usd": pricing_data.get("raw_ungraded_usd") if pricing_data else None,
                "graded_value_usd": pricing_data.get("graded_value_usd") if pricing_data else None,
                "net_grading_roi_usd": (
                    pricing_data.get("net_grading_roi_usd") if pricing_data else None
                ),
                "should_submit_for_grading": (
                    pricing_data.get("should_submit_for_grading") if pricing_data else None
                ),
                "recommendation_short": (
                    "SUBMIT_FOR_GRADING"
                    if (pricing_data and pricing_data.get("should_submit_for_grading"))
                    else "SELL_RAW_UNGRADED"
                ),
            }
            self.session_store.append_turn(
                session_id=session_id,
                user_message=user_query,
                agent_response=final_response,
                card_metadata=card_turn_metadata,
            )

            # Step 7: Trigger Non-Blocking Async Memory Consolidation
            self.async_memory.schedule_card_indexing(
                session_id=session_id,
                card_evaluation=card_turn_metadata,
            )

            overall_status = "SUCCESS"
            if listing_data and listing_data.get("status") == "BLOCKED_HITL_REQUIRED":
                overall_status = "AWAITING_HUMAN_APPROVAL"
            elif (
                (subgrade_data and subgrade_data.get("status") == "ERROR")
                or (pricing_data and pricing_data.get("status") == "ERROR")
                or (listing_data and listing_data.get("status") == "ERROR")
            ):
                overall_status = "GUIDED_TOOL_ERROR"

            result_bundle = {
                "status": overall_status,
                "trace_id": trace_id,
                "policy_check": policy_check,
                "routing": routing,
                "inspector_output": inspector_output,
                "strategist_output": strategist_output,
                "reflection": reflection,
                "session_context": self.session_store.build_context_window(
                    session_id, current_query=user_query
                ),
                "response_text": final_response,
            }

            log_outcome(
                agent_name=self.name,
                action="Orchestrate multi-agent Pokémon card appraisal and marketplace workflow",
                target_tool="handle_collector_request",
                status=overall_status,
                result_summary={
                    "route_mode": routing["route_mode"],
                    "routed_models": routing["routed_models"],
                    "estimated_grade": effective_grade,
                    "reflection_score": reflection["reflection_score"],
                },
                latency_ms=(time.perf_counter() - start_ts) * 1000,
                intent_id=intent["intent_id"],
                trace_id=trace_id,
            )
            return result_bundle

    @staticmethod
    def _compose_summary_response(
        card_name: str,
        set_name: str,
        routing: dict[str, Any],
        subgrade_data: dict[str, Any] | None,
        pricing_data: dict[str, Any] | None,
        listing_data: dict[str, Any] | None,
    ) -> str:
        """Assemble a clear, structured collector concierge summary from subagent outputs."""
        sections: list[str] = [
            f"### PokéGrade AI Concierge Report: {card_name} ({set_name})",
            f"- **Routing Strategy**: `{routing['route_mode']}` via `{', '.join(routing['routed_models'])}`",
        ]

        if subgrade_data:
            if subgrade_data.get("status") == "SUCCESS":
                sections.append(
                    f"- **Physical Grade (`{INSPECTOR_MODEL}`)**: **{subgrade_data['estimated_grade']:.1f}** "
                    f"(`{subgrade_data['condition_tier']}`) — Limiting Subgrade: "
                    f"`{subgrade_data['limiting_subgrade_factor']}`"
                )
                sections.append(f"  - *Inspector Notes*: {subgrade_data['inspector_notes']}")
            else:
                sections.append(
                    f"- **Inspector Guided Recovery**: {subgrade_data.get('guided_recovery')}"
                )

        if pricing_data:
            if pricing_data.get("status") == "SUCCESS":
                sections.append(
                    f"- **Market Strategy (`{STRATEGIST_MODEL}`)**: Raw `${pricing_data['raw_ungraded_usd']:.2f}` "
                    f"| Graded `${pricing_data['graded_value_usd']:.2f}` | Fee `${pricing_data['grading_fee_usd']:.2f}` "
                    f"| Net ROI `${pricing_data['net_grading_roi_usd']:.2f}` ({pricing_data['roi_percentage']:.1f}%)"
                )
                sections.append(f"  - *Recommendation*: **{pricing_data['recommendation']}**")
            else:
                sections.append(
                    f"- **Pricing Guided Recovery**: {pricing_data.get('guided_recovery')}"
                )

        if listing_data:
            if listing_data.get("status") == "PUBLISHED":
                sections.append(
                    f"- **Marketplace Listing**: PUBLISHED (`{listing_data['listing_id']}`) at "
                    f"`${listing_data['listing_price_usd']:.2f}` (HITL Verified: `{listing_data['human_verified']}`)"
                )
            else:
                sections.append(
                    f"- **Marketplace HITL Gate (`{listing_data.get('status')}`)**: "
                    f"{listing_data.get('guided_recovery')}"
                )

        return "\n".join(sections)


def build_adk_sequential_pipeline_agent() -> Any:
    """Build an ADK `SequentialAgent` chaining Inspector (Flash) -> Strategist (Pro)."""
    try:
        from google.adk.agents import SequentialAgent  # type: ignore[import-untyped]

        inspector = build_adk_inspector_agent(name="card_inspector_subagent")
        strategist = build_adk_strategist_agent(name="market_strategist_subagent")
        sub_agents = [ag for ag in (inspector, strategist) if ag is not None]
        return SequentialAgent(
            name="pokegrade_sequential_appraisal_pipeline",
            description=(
                "Sequential multi-agent workflow executing physical subgrade inspection "
                "(gemini-3.8-flash) followed by TCG market & HITL strategy (gemini-3.1-pro-preview)."
            ),
            sub_agents=sub_agents,
        )
    except Exception:
        return None


def build_adk_root_agent() -> Any:
    """Build the top-level Google ADK Coordinator `Agent` hierarchy if ADK is installed.

    Registers three specialized sub-agent delegation paths so the ADK Coordinator
    can dynamically route:
    1. `card_inspector_triage_subagent` (`INSPECTOR_ONLY` via `gemini-3.8-flash`)
    2. `market_strategist_direct_subagent` (`STRATEGIST_ONLY` via `gemini-3.1-pro-preview`)
    3. `pokegrade_sequential_appraisal_pipeline` (`FULL_PIPELINE` SequentialAgent Flash -> Pro)
    """
    try:
        from google.adk.agents import Agent  # type: ignore[import-untyped]

        triage_inspector = build_adk_inspector_agent(name="card_inspector_triage_subagent")
        direct_strategist = build_adk_strategist_agent(name="market_strategist_direct_subagent")
        seq_pipeline = build_adk_sequential_pipeline_agent()
        sub_agents = [
            ag for ag in (triage_inspector, direct_strategist, seq_pipeline) if ag is not None
        ]
        return Agent(
            name="pokegrade_coordinator",
            model=STRATEGIST_MODEL,
            description=(
                "Master PokéGrade AI Coordinator routing physical subgrade inspection "
                "to gemini-3.8-flash, direct TCG market/listing strategy to "
                "gemini-3.1-pro-preview, or full sequential appraisal pipelines."
            ),
            instruction=SYSTEM_CONSTITUTION,
            sub_agents=sub_agents,
        )
    except Exception:
        return None


def build_adk_app() -> Any:
    """Build the Google ADK `App` bundling Coordinator, Compaction, Context Cache, and Plugins."""
    try:
        from google.adk.apps.app import App  # type: ignore[import-untyped]

        coordinator_agent = build_adk_root_agent()
        if coordinator_agent is None:
            return None
        return App(
            name="pokegrade_ai",
            root_agent=coordinator_agent,
            events_compaction_config=build_adk_events_compaction_config(),
            context_cache_config=build_gcp_context_cache_config(),
            plugins=[PolicyGuardrailPlugin()],
        )
    except Exception:
        return None


sequential_pipeline_agent = build_adk_sequential_pipeline_agent()
root_agent = build_adk_root_agent()
adk_app = build_adk_app()
default_coordinator = PokeGradeCoordinator()

