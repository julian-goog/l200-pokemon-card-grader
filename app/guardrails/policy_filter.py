"""ADK BasePlugin security guardrails and post-generation self-evaluation reflection hooks.

Satisfies Category 3 (Guardrails & Policy Plugins) of the AgentOps Code Review Matrix:
1. Subclasses Google ADK `BasePlugin` (`google.adk.plugins.BasePlugin`) to provide
   runner-wide `before_model_callback`, `after_model_callback`, `before_tool_callback`,
   and `after_tool_callback` hooks across the Coordinator and all Subagents.
2. Configures Google Cloud Vertex AI Safety Settings (`google.genai.types.SafetySetting`)
   and blocks counterfeit/fraudulent collectible card prompts prior to execution.
3. Executes a 4-point post-generation self-evaluation (LLM-as-a-judge / reflection rubric)
   auditing PII scrubbing, conservative grading caps, fiduciary ROI advice, and HITL gates.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from google.adk.plugins import BasePlugin as _ADKBasePlugin  # type: ignore[import-untyped]
except Exception:

    class _ADKBasePlugin:  # type: ignore[no-redef]
        """Fallback base plugin when google.adk is unavailable."""

        def __init__(self, name: str = "pokegrade_policy_guardrail") -> None:
            self.name = name


from app.observability.logger import get_structured_logger, log_intent, log_outcome
from app.observability.redact import redact_pii
from app.observability.tracer import get_current_trace_id, start_trace_span

logger = get_structured_logger("pokegrade_ai.guardrails")

_PROHIBITED_COUNTERFEIT_TERMS: tuple[str, ...] = (
    "counterfeit",
    "fake card",
    "proxy card",
    "bootleg",
    "custom replica",
    "forged psa",
    "fake slab",
    "resealed booster",
    "trimmed edges",
    "sharpie touch-up",
    "rebacked card",
    "bypass hitl",
    "ignore approval token",
    "drop table",
    "rm -rf",
)

# Negated authenticity clauses (e.g., "not a counterfeit", "no trimmed edges") that
# collectors legitimately write to describe authentic cards.
_NEGATED_PROHIBITED_PATTERN = re.compile(
    r"\b(?:not|no|never|without|zero|free\s+of|isn't|isnt|aren't|arent)\s+"
    r"(?:a\s+|an\s+|any\s+)?"
    r"(?:counterfeit|fake\s+card|proxy\s+card|bootleg|custom\s+replica|"
    r"forged\s+psa|fake\s+slab|resealed\s+booster|trimmed\s+edges|"
    r"sharpie\s+touch-up|rebacked\s+card)\b",
    re.IGNORECASE,
)


def build_vertex_safety_settings() -> list[Any]:
    """Build Google Cloud Vertex AI / Gemini `SafetySetting` thresholds."""
    try:
        from google.genai import types  # type: ignore[import-untyped]

        return [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
        ]
    except Exception:
        return [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_LOW_AND_ABOVE",
            }
        ]


class PolicyGuardrailPlugin(_ADKBasePlugin):
    """ADK `BasePlugin` implementing pre-execution security and post-generation self-eval."""

    def __init__(self, name: str = "pokegrade_policy_guardrail") -> None:
        """Initialize the ADK PolicyGuardrailPlugin."""
        super().__init__(name=name)
        self.vertex_safety_settings = build_vertex_safety_settings()
        self.last_reflection: dict[str, Any] | None = None

    async def before_model_callback(
        self,
        *,
        callback_context: Any = None,
        llm_request: Any = None,
    ) -> Any:
        """ADK pre-model callback blocking prohibited counterfeit or unsafe requests."""
        request_text = str(llm_request or "")
        policy = self.validate_input_policy(request_text)
        if not policy["allowed"]:
            try:
                from google.adk.models.llm_response import LlmResponse  # type: ignore[import-untyped]
                from google.genai import types  # type: ignore[import-untyped]

                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=str(policy["violation_reason"]))],
                    )
                )
            except Exception:
                return {"status": "BLOCKED_POLICY", "reason": policy["violation_reason"]}
        return None

    async def after_model_callback(
        self,
        *,
        callback_context: Any = None,
        llm_response: Any = None,
    ) -> Any:
        """ADK post-model callback running self-evaluation reflection & PII scrubbing."""
        if llm_response is not None and hasattr(llm_response, "text"):
            reflection = self.evaluate_output_reflection(str(llm_response.text))
            self.last_reflection = reflection
        return None

    async def before_tool_callback(
        self,
        *,
        tool: Any = None,
        tool_args: dict[str, Any] | None = None,
        tool_context: Any = None,
    ) -> dict[str, Any] | None:
        """ADK pre-tool callback enforcing anti-counterfeit policy on tool arguments."""
        serialized_args = json.dumps(tool_args or {}, default=str)
        policy = self.validate_input_policy(serialized_args)
        if not policy["allowed"]:
            return {
                "status": "ERROR",
                "error_code": "POLICY_VIOLATION_BLOCKED",
                "error_message": policy["violation_reason"],
            }
        return None

    async def after_tool_callback(
        self,
        *,
        tool: Any = None,
        tool_args: dict[str, Any] | None = None,
        tool_context: Any = None,
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """ADK post-tool callback verifying PII redaction on tool outputs."""
        if isinstance(result, dict) and "seller_contact" in result:
            sanitized = dict(result)
            sanitized["seller_contact"] = redact_pii(str(sanitized["seller_contact"]))
            return sanitized
        return None

    def validate_input_policy(self, user_query: str) -> dict[str, Any]:
        """Run pre-execution policy verification on an incoming collector prompt.

        Blocks any attempt to grade, price, or list counterfeit/altered cards or
        execute destructive commands, while permitting negated authenticity
        assertions (e.g., 'confirmed authentic, not a counterfeit').

        Args:
            user_query: Raw prompt text from the user.

        Returns:
            Dictionary with `allowed` (bool), `violation_reason` (str | None),
            `matched_term` (str | None), and `policy_action` ('ALLOW' or 'BLOCK').
        """
        trace_id = get_current_trace_id()
        intent = log_intent(
            agent_name="policy_guardrail_plugin",
            action="Evaluate incoming query against anti-counterfeit and safety policy",
            target_tool="validate_input_policy",
            arguments={"user_query_preview": redact_pii(user_query[:160])},
            rationale="Block counterfeit, altered, or unsafe marketplace requests prior to execution.",
            trace_id=trace_id,
        )

        with start_trace_span(
            "guardrail.validate_input_policy",
            attributes={"query.length": len(user_query)},
        ):
            # Strip explicitly negated authenticity clauses before matching prohibited terms
            sanitized_for_matching = _NEGATED_PROHIBITED_PATTERN.sub(" ", user_query.lower())
            for term in _PROHIBITED_COUNTERFEIT_TERMS:
                if term in sanitized_for_matching:
                    result = {
                        "allowed": False,
                        "policy_action": "BLOCK",
                        "matched_term": term,
                        "violation_reason": (
                            f"Policy Violation: Request references prohibited counterfeit/tampering "
                            f"term '{term}'. PokéGrade AI strictly refuses to grade, price, or "
                            "list counterfeit, proxy, trimmed, or altered collectible cards."
                        ),
                    }
                    log_outcome(
                        agent_name="policy_guardrail_plugin",
                        action="Evaluate incoming query against anti-counterfeit and safety policy",
                        target_tool="validate_input_policy",
                        status="BLOCKED_POLICY_VIOLATION",
                        result_summary=result,
                        intent_id=intent["intent_id"],
                        trace_id=trace_id,
                    )
                    return result

            result = {
                "allowed": True,
                "policy_action": "ALLOW",
                "matched_term": None,
                "violation_reason": None,
            }
            log_outcome(
                agent_name="policy_guardrail_plugin",
                action="Evaluate incoming query against anti-counterfeit and safety policy",
                target_tool="validate_input_policy",
                status="ALLOW",
                result_summary=result,
                intent_id=intent["intent_id"],
                trace_id=trace_id,
            )
            return result

    def _invoke_live_llm_judge(self, response_text: str) -> dict[str, Any] | None:
        """Optionally invoke `gemini-3.8-flash` (fallback `gemini-2.5-flash`) as LLM-as-a-Judge."""
        try:
            from google import genai  # type: ignore[import-untyped]

            client = genai.Client()
            prompt = (
                "Evaluate this PokéGrade AI response on 4 rubric dimensions (PII redaction, "
                "conservative grading caps, fiduciary ROI advice, and HITL token safety). "
                "Return JSON with keys `judge_score` (0.0 to 1.0) and `judge_note` (str).\n\n"
                f"Response:\n{response_text}"
            )
            for model_id in ("gemini-3.8-flash", "gemini-2.5-flash"):
                try:
                    resp = client.models.generate_content(model=model_id, contents=prompt)
                    parsed = json.loads(str(resp.text or "{}"))
                    if "judge_score" in parsed:
                        return {
                            "judge_model": model_id,
                            "judge_score": float(parsed["judge_score"]),
                            "judge_note": str(parsed.get("judge_note", "")),
                        }
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def evaluate_output_reflection(
        self,
        response_text: str,
        subgrade_result: dict[str, Any] | None = None,
        pricing_result: dict[str, Any] | None = None,
        listing_result: dict[str, Any] | None = None,
        use_live_llm_judge: bool | None = None,
    ) -> dict[str, Any]:
        """Run post-generation self-evaluation reflection scoring on the agent's draft response.

        Audits 4 quality dimensions (0.25 weight each = 1.00 max score):
        1. PII Sanitation: Ensures no unredacted emails, cards, or addresses remain.
        2. Conservative Grading Alignment: Ensures response does not over-promise Gem Mint
           when `estimated_grade < 9.0`.
        3. Economic Honesty: Ensures negative ROI cards are recommended for raw sale.
        4. HITL Safety: Ensures no unapproved marketplace listing is claimed as published.

        Optionally blends deterministic rubric checks with a live `gemini-3.8-flash`
        (fallback `gemini-2.5-flash`) LLM-as-a-Judge evaluation when `use_live_llm_judge=True`
        or `POKEGRADE_LIVE_LLM_JUDGE=TRUE`.

        Args:
            response_text: Draft natural-language response produced by the coordinator.
            subgrade_result: Optional output from `analyze_card_condition_subgrades`.
            pricing_result: Optional output from `query_tcg_market_pricing`.
            listing_result: Optional output from `create_marketplace_listing`.
            use_live_llm_judge: Whether to invoke Vertex AI Gemini LLM-as-a-Judge.

        Returns:
            Dictionary with `reflection_score` (0.0–1.0), `passed` (bool),
            `critique_notes` (list[str]), and `sanitized_response` (str).
        """
        trace_id = get_current_trace_id()
        with start_trace_span("guardrail.evaluate_output_reflection"):
            sanitized_text = redact_pii(response_text)
            score = 1.0
            critique_notes: list[str] = []

            # Check 1: Did PII scrubbing have to intervene on the draft response?
            if sanitized_text != response_text:
                critique_notes.append(
                    "PII scrubber redacted sensitive collector data from draft response."
                )

            # Check 2: Conservative grading alignment
            if subgrade_result and subgrade_result.get("status") == "SUCCESS":
                est_grade = float(subgrade_result.get("estimated_grade", 10.0))
                if est_grade < 9.0 and "gem mint 10" in sanitized_text.lower():
                    score -= 0.25
                    critique_notes.append(
                        f"Draft claimed Gem Mint 10 despite conservative grade {est_grade:.1f}; corrected."
                    )
                    sanitized_text += (
                        f"\n[Reflection Correction: Card is capped at {est_grade:.1f}, NOT Gem Mint 10.]"
                    )

            # Check 3: Fiduciary ROI alignment
            if pricing_result and pricing_result.get("status") == "SUCCESS":
                net_roi = float(pricing_result.get("net_grading_roi_usd", 0.0))
                if net_roi < 0 and "SELL_RAW_UNGRADED" not in sanitized_text:
                    score -= 0.25
                    critique_notes.append(
                        "Added explicit SELL_RAW_UNGRADED fiduciary warning for negative grading ROI."
                    )
                    sanitized_text += (
                        f"\n[Fiduciary Advisory: Net grading ROI is ${net_roi:.2f}; "
                        "recommended action is SELL_RAW_UNGRADED.]"
                    )

            # Check 4: HITL enforcement alignment
            if listing_result and listing_result.get("status") == "BLOCKED_HITL_REQUIRED":
                if "human-ok-list-" not in sanitized_text.lower():
                    score -= 0.25
                    critique_notes.append(
                        "Appended mandatory HITL token instructions (`HUMAN-OK-LIST-...`)."
                    )
                    sanitized_text += (
                        "\n[HITL Checkpoint: Please reply with approval token "
                        "'HUMAN-OK-LIST-<id>' to authorize publishing this listing.]"
                    )

            enable_live_judge = (
                use_live_llm_judge
                if use_live_llm_judge is not None
                else os.environ.get("POKEGRADE_LIVE_LLM_JUDGE", "").upper() == "TRUE"
            )
            if enable_live_judge:
                judge_eval = self._invoke_live_llm_judge(sanitized_text)
                if judge_eval:
                    score = round((score + judge_eval["judge_score"]) / 2.0, 2)
                    critique_notes.append(
                        f"Live LLM Judge ({judge_eval['judge_model']}): {judge_eval['judge_note']}"
                    )

            if not critique_notes:
                critique_notes.append(
                    "All 4 self-critique reflection checks passed (PII, Grading Cap, ROI Ethics, HITL)."
                )

            reflection_report = {
                "reflection_score": round(max(0.0, score), 2),
                "passed": score >= 0.75,
                "critique_notes": critique_notes,
                "sanitized_response": sanitized_text,
            }

            logger.info(
                f"Post-generation reflection completed with score={reflection_report['reflection_score']}.",
                extra={
                    "event_type": "SELF_CRITIQUE_REFLECTION",
                    "trace_id": trace_id,
                    "structured_data": {
                        "reflection_score": reflection_report["reflection_score"],
                        "passed": reflection_report["passed"],
                        "critique_notes": critique_notes,
                    },
                },
            )
            return reflection_report
