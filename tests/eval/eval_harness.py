"""Automated Golden Dataset Evaluation Harness for PokéGrade AI.

Statically measures:
1. Strategic Model Routing Accuracy (`gemini-3.8-flash` vs `gemini-3.1-pro-preview`)
2. Conservative Physical Grading Accuracy (subgrade cap & structural crease rules)
3. Economic ROI & Submission Recommendation Accuracy
4. Pre-Execution Guardrail & Human-in-the-Loop (HITL) Enforcement
5. Zero PII Leakage across agent responses and structured logs
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from app.agent import PokeGradeCoordinator

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> list[dict[str, Any]]:
    """Load the golden evaluation dataset from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def run_evaluation_suite(
    dataset_path: Path = GOLDEN_DATASET_PATH,
) -> dict[str, Any]:
    """Execute all golden dataset cases against `PokeGradeCoordinator` and score accuracy.

    Args:
        dataset_path: Path to `golden_dataset.json`.

    Returns:
        Evaluation summary dictionary containing `total_cases`, `passed_cases`,
        `pass_rate`, `routing_accuracy`, `grading_accuracy`, and `case_results`.
    """
    cases = load_golden_dataset(dataset_path)
    coordinator = PokeGradeCoordinator()

    case_results: list[dict[str, Any]] = []
    routing_hits = 0
    grading_hits = 0
    graded_case_count = 0

    for case in cases:
        expected = case["expected"]
        result = coordinator.handle_collector_request(
            session_id=case["session_id"],
            user_query=case["user_query"],
            card_name=case["card_name"],
            set_name=case["set_name"],
            card_number=case["card_number"],
            rarity=case["rarity"],
            grading_company=case["grading_company"],
            centering_score=case["centering_score"],
            corners_score=case["corners_score"],
            edges_score=case["edges_score"],
            surface_score=case["surface_score"],
            surface_notes=case["surface_notes"],
            request_market_analysis=case["request_market_analysis"],
            request_listing=case["request_listing"],
            seller_contact=case["seller_contact"],
            human_approval_token=case["human_approval_token"],
        )

        checks: dict[str, bool] = {}

        # 1. Overall workflow status match
        checks["status_match"] = result["status"] == expected["status"]

        # 2. Routing accuracy check
        if expected["route_mode"] is None:
            checks["routing_match"] = result["routing"] is None
        else:
            routing = result["routing"] or {}
            checks["routing_match"] = (
                routing.get("route_mode") == expected["route_mode"]
                and routing.get("routed_models") == expected["routed_models"]
            )
        if checks["routing_match"]:
            routing_hits += 1

        # 3. Grade bounds check
        if expected["min_grade"] is not None and expected["max_grade"] is not None:
            graded_case_count += 1
            subgrade_analysis = (result.get("inspector_output") or {}).get(
                "subgrade_analysis", {}
            )
            est_grade = subgrade_analysis.get("estimated_grade")
            checks["grade_in_bounds"] = (
                isinstance(est_grade, (int, float))
                and expected["min_grade"] <= float(est_grade) <= expected["max_grade"]
            )
            if checks["grade_in_bounds"]:
                grading_hits += 1
        else:
            checks["grade_in_bounds"] = True

        # 4. Market ROI recommendation check
        if expected["should_submit_for_grading"] is not None:
            pricing = (result.get("strategist_output") or {}).get("market_pricing", {})
            checks["roi_decision_match"] = (
                pricing.get("should_submit_for_grading")
                == expected["should_submit_for_grading"]
            )
        else:
            checks["roi_decision_match"] = True

        # 5. Zero raw PII leakage check
        response_text = result.get("response_text", "")
        checks["pii_scrubbed"] = (
            "collector.ash@pallet-mail.com" not in response_text
            and "742 Evergreen Terrace" not in response_text
        )

        case_passed = all(checks.values())
        case_results.append(
            {
                "case_id": case["case_id"],
                "description": case["description"],
                "passed": case_passed,
                "checks": checks,
                "actual_status": result["status"],
                "expected_status": expected["status"],
            }
        )

    total = len(cases)
    passed = sum(1 for c in case_results if c["passed"])
    summary = {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "routing_accuracy": round(routing_hits / total, 4) if total else 0.0,
        "grading_accuracy": (
            round(grading_hits / graded_case_count, 4) if graded_case_count else 1.0
        ),
        "case_results": case_results,
    }
    return summary


def main() -> int:
    """CLI entrypoint for running the PokéGrade AI evaluation harness."""
    summary = run_evaluation_suite()
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
