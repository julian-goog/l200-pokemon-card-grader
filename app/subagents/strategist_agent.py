"""Market Strategy Subagent (`gemini-3.1-pro-preview`) for TCG pricing, ROI, and HITL listings.

Specializes in complex financial analysis, grading fee breakeven calculations,
and Human-in-the-Loop (HITL) gated marketplace listing execution.
"""

from __future__ import annotations

import time
from typing import Any

from app.observability.logger import log_intent, log_outcome
from app.observability.tracer import get_current_trace_id, start_trace_span
from app.prompts.constitution import STRATEGIST_SUBAGENT_PROMPT
from app.tools.card_tools import create_marketplace_listing, query_tcg_market_pricing
from app.tools.schemas import Condition, GradingCompany, Rarity

STRATEGIST_MODEL = "gemini-3.1-pro-preview"


class StrategistSubagent:
    """Specialized subagent routed to `gemini-3.1-pro-preview` for market ROI and listing strategy."""

    def __init__(self, model_name: str = STRATEGIST_MODEL) -> None:
        """Initialize the Market Strategy Subagent.

        Args:
            model_name: Gemini model identifier (defaults to `gemini-3.1-pro-preview`).
        """
        self.name = "market_strategist_subagent"
        self.model_name = model_name
        self.system_instruction = STRATEGIST_SUBAGENT_PROMPT
        self.tools = [query_tcg_market_pricing, create_marketplace_listing]

    def evaluate_market_strategy(
        self,
        card_name: str,
        set_name: str,
        estimated_grade: float,
        condition: str | Condition,
        rarity: str | Rarity = Rarity.HOLO_RARE,
        grading_company: str | GradingCompany = GradingCompany.PSA,
        request_listing: bool = False,
        proposed_listing_price_usd: float | None = None,
        seller_contact: str = "",
        human_approval_token: str | None = None,
    ) -> dict[str, Any]:
        """Run deep market valuation, fee breakeven calculation, and optional HITL listing.

        Args:
            card_name: Official Pokémon card name.
            set_name: Expansion set name.
            estimated_grade: Projected numerical grade (1.0–10.0) from Inspector Subagent.
            condition: Standardized `Condition` enum value.
            rarity: Official card rarity tier.
            grading_company: Target grading authority ('PSA', 'BGS', 'CGC').
            request_listing: Whether the user requested creating a marketplace listing.
            proposed_listing_price_usd: Optional custom asking price in USD.
            seller_contact: Optional seller contact notes (scrubbed for PII).
            human_approval_token: Optional HITL confirmation token ('HUMAN-OK-LIST-...').

        Returns:
            Dictionary containing `subagent`, `routed_model`, `market_pricing`, and
            optional `marketplace_listing` outcome.
        """
        start_ts = time.perf_counter()
        with start_trace_span(
            "subagent.market_strategist",
            attributes={
                "subagent.name": self.name,
                "subagent.model": self.model_name,
                "card.name": card_name,
                "request.listing": request_listing,
            },
        ):
            trace_id = get_current_trace_id()
            intent = log_intent(
                agent_name=self.name,
                action="Evaluate TCG market pricing, grading ROI, and marketplace strategy",
                target_tool="query_tcg_market_pricing",
                arguments={
                    "card_name": card_name,
                    "set_name": set_name,
                    "estimated_grade": estimated_grade,
                    "request_listing": request_listing,
                    "model": self.model_name,
                },
                rationale="Route multi-variable ROI economics and HITL listing to gemini-3.1-pro-preview.",
                trace_id=trace_id,
            )

            pricing = query_tcg_market_pricing(
                card_name=card_name,
                set_name=set_name,
                estimated_grade=estimated_grade,
                rarity=rarity,
                grading_company=grading_company,
            )

            listing_outcome: dict[str, Any] | None = None
            if request_listing and pricing.get("status") == "SUCCESS":
                default_price = (
                    pricing.get("graded_value_usd", 50.0)
                    if pricing.get("should_submit_for_grading")
                    else pricing.get("raw_ungraded_usd", 25.0)
                )
                listing_price = (
                    proposed_listing_price_usd
                    if proposed_listing_price_usd is not None
                    else float(default_price)
                )
                listing_outcome = create_marketplace_listing(
                    card_name=card_name,
                    set_name=set_name,
                    condition=condition,
                    estimated_grade=estimated_grade,
                    listing_price_usd=listing_price,
                    grading_company=grading_company,
                    seller_contact=seller_contact,
                    human_approval_token=human_approval_token,
                )

            payload = {
                "subagent": self.name,
                "routed_model": self.model_name,
                "market_pricing": pricing,
                "marketplace_listing": listing_outcome,
            }
            log_outcome(
                agent_name=self.name,
                action="Evaluate TCG market pricing, grading ROI, and marketplace strategy",
                target_tool="query_tcg_market_pricing",
                status=pricing.get("status", "UNKNOWN"),
                result_summary={
                    "routed_model": self.model_name,
                    "net_grading_roi_usd": pricing.get("net_grading_roi_usd"),
                    "should_submit_for_grading": pricing.get("should_submit_for_grading"),
                    "listing_status": (
                        listing_outcome.get("status") if listing_outcome else "NOT_REQUESTED"
                    ),
                },
                latency_ms=(time.perf_counter() - start_ts) * 1000,
                intent_id=intent["intent_id"],
                trace_id=trace_id,
            )
            return payload


def build_adk_strategist_agent(name: str = "market_strategist_subagent") -> Any:
    """Build a Google ADK `Agent` instance for the Market Strategy Subagent if ADK is installed."""
    try:
        from google.adk.agents import Agent  # type: ignore[import-untyped]
        from google.adk.tools import FunctionTool  # type: ignore[import-untyped]

        adk_tools = [
            FunctionTool(query_tcg_market_pricing),
            FunctionTool(create_marketplace_listing),
        ]
        return Agent(
            name=name,
            model=STRATEGIST_MODEL,
            description=(
                "Deep Pokémon TCG market strategist using gemini-3.1-pro-preview to compute "
                "grading fee breakeven, net ROI, and HITL-verified marketplace listings."
            ),
            instruction=STRATEGIST_SUBAGENT_PROMPT,
            tools=adk_tools,
        )
    except Exception:
        return None


strategist_adk_agent = build_adk_strategist_agent()
