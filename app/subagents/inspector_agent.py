"""Card Inspector Subagent (`gemini-3.8-flash`) for rapid physical subgrade triage.

Specializes in fast, low-latency inspection of the 4 physical card dimensions
(Centering, Corners, Edges, Surface) and computes a conservative composite grade
via `analyze_card_condition_subgrades`.
"""

from __future__ import annotations

import time
from typing import Any

from app.observability.logger import log_intent, log_outcome
from app.observability.tracer import get_current_trace_id, start_trace_span
from app.prompts.constitution import INSPECTOR_SUBAGENT_PROMPT
from app.tools.card_tools import analyze_card_condition_subgrades
from app.tools.schemas import GradingCompany, Rarity
from app.tools.vision_inspector import inspect_card_image_multimodal

INSPECTOR_MODEL = "gemini-3.8-flash"


class InspectorSubagent:
    """Specialized subagent routed to `gemini-3.8-flash` for physical condition triage."""

    def __init__(self, model_name: str = INSPECTOR_MODEL) -> None:
        """Initialize the Card Inspector Subagent.

        Args:
            model_name: Gemini model identifier (defaults to `gemini-3.8-flash`).
        """
        self.name = "card_inspector_subagent"
        self.model_name = model_name
        self.system_instruction = INSPECTOR_SUBAGENT_PROMPT
        self.tools = [analyze_card_condition_subgrades, inspect_card_image_multimodal]

    def inspect_card(
        self,
        card_name: str,
        set_name: str,
        centering_score: float,
        corners_score: float,
        edges_score: float,
        surface_score: float,
        card_number: str = "001/102",
        rarity: str | Rarity = Rarity.HOLO_RARE,
        grading_company: str | GradingCompany = GradingCompany.PSA,
        surface_notes: str = "",
    ) -> dict[str, Any]:
        """Run physical subgrade triage on `gemini-3.8-flash`.

        Args:
            card_name: Official Pokémon card name.
            set_name: Expansion set name.
            centering_score: Border alignment score (1.0–10.0).
            corners_score: Corner sharpness score (1.0–10.0).
            edges_score: Edge smoothness score (1.0–10.0).
            surface_score: Surface gloss and defect score (1.0–10.0).
            card_number: Collector number.
            rarity: Official card rarity tier.
            grading_company: Target grading authority ('PSA', 'BGS', 'CGC').
            surface_notes: Optional notes on scratches, print lines, or creases.

        Returns:
            Dictionary containing subagent metadata (`routed_model`, `subagent`) and
            the structured `subgrade_analysis` result.
        """
        start_ts = time.perf_counter()
        with start_trace_span(
            "subagent.card_inspector",
            attributes={
                "subagent.name": self.name,
                "subagent.model": self.model_name,
                "card.name": card_name,
            },
        ):
            trace_id = get_current_trace_id()
            intent = log_intent(
                agent_name=self.name,
                action="Execute physical card subgrade triage",
                target_tool="analyze_card_condition_subgrades",
                arguments={
                    "card_name": card_name,
                    "set_name": set_name,
                    "model": self.model_name,
                },
                rationale="Route rapid physical subgrade inspection to low-latency gemini-3.8-flash.",
                trace_id=trace_id,
            )

            analysis = analyze_card_condition_subgrades(
                card_name=card_name,
                set_name=set_name,
                centering_score=centering_score,
                corners_score=corners_score,
                edges_score=edges_score,
                surface_score=surface_score,
                card_number=card_number,
                rarity=rarity,
                grading_company=grading_company,
                surface_notes=surface_notes,
            )

            payload = {
                "subagent": self.name,
                "routed_model": self.model_name,
                "subgrade_analysis": analysis,
            }
            log_outcome(
                agent_name=self.name,
                action="Execute physical card subgrade triage",
                target_tool="analyze_card_condition_subgrades",
                status=analysis.get("status", "UNKNOWN"),
                result_summary={
                    "estimated_grade": analysis.get("estimated_grade"),
                    "condition_tier": analysis.get("condition_tier"),
                    "routed_model": self.model_name,
                },
                latency_ms=(time.perf_counter() - start_ts) * 1000,
                intent_id=intent["intent_id"],
                trace_id=trace_id,
            )
            return payload


def build_adk_inspector_agent(name: str = "card_inspector_subagent") -> Any:
    """Build a Google ADK `Agent` instance for the Card Inspector Subagent if ADK is installed."""
    try:
        from google.adk.agents import Agent  # type: ignore[import-untyped]
        from google.adk.tools import FunctionTool  # type: ignore[import-untyped]

        adk_tools = [
            FunctionTool(analyze_card_condition_subgrades),
            FunctionTool(inspect_card_image_multimodal),
        ]
        return Agent(
            name=name,
            model=INSPECTOR_MODEL,
            description=(
                "Rapid physical Pokémon card condition inspector using gemini-3.8-flash "
                "to evaluate Centering, Corners, Edges, and Surface subgrades or card images."
            ),
            instruction=INSPECTOR_SUBAGENT_PROMPT,
            tools=adk_tools,
        )
    except Exception:
        return None


inspector_adk_agent = build_adk_inspector_agent()
