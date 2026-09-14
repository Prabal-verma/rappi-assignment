"""Evaluation runner.

    python -m evals.runner                     # whole suite, configured provider
    python -m evals.runner --provider gemini   # against a live model
    python -m evals.runner --case SC2X         # one case
    python -m evals.runner --compare           # rules engine vs model, side by side

Every case runs against a freshly reset and reseeded database, so cases
cannot contaminate each other and the suite is order-independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import run_case  # noqa: E402
from app.db.models import AgentRun  # noqa: E402
from app.db.seed import SCENARIOS, seed_all  # noqa: E402
from app.db.session import SessionLocal, reset_db  # noqa: E402
from evals.rubric import DIMENSIONS, CaseResult, grade_case  # noqa: E402

SUITE_PATH = Path(__file__).parent / "scenarios.yaml"
REPORT_DIR = Path(__file__).parent / "reports"


def load_suite() -> list[dict[str, Any]]:
    with SUITE_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)["cases"]


def run_one(case: dict[str, Any], provider: str | None, model: str | None) -> CaseResult:
    """Reset the world, run the scenario, grade what it left behind."""
    reset_db()
    with SessionLocal() as session:
        seed_all(session)
        session.commit()

    scenario = SCENARIOS[case["scenario_id"]]
    with SessionLocal() as session:
        run_id = run_case(session, case["scenario_id"], dict(scenario), provider, model)

    with SessionLocal() as session:
        run = session.get(AgentRun, run_id)
        return grade_case(session, case, run)


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────


def _bar(score: float, width: int = 12) -> str:
    filled = int(round(score * width))
    return "#" * filled + "." * (width - filled)


def print_report(results: list[CaseResult], provider_label: str) -> None:
    print()
    print("=" * 84)
    print(f"  AI PURCHASING AGENT — EVALUATION  ({provider_label})")
    print("=" * 84)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print()
        print(f"  [{status}]  {r.case_id}   score {r.score:.2f}  {_bar(r.score)}")
        print(
            f"           decision={r.decision_type or '-'}  units={r.units if r.units is not None else '-'}  "
            f"outcome={r.outcome}  replans={r.attempts}  {r.duration_ms}ms"
        )
        for dim, score in r.dimension_scores.items():
            print(f"             {dim:<13} {score:.2f}  {_bar(score, 10)}")
        for check in r.checks:
            if not check.passed:
                mark = "CRITICAL" if check.critical else "fail    "
                print(f"             ! {mark}  {check.name}: {check.detail}")
        for note in r.notes:
            print(f"             ~ note: {note}")

    print()
    print("-" * 84)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    critical = sum(len(r.critical_failures) for r in results)
    overall = sum(r.score for r in results) / total if total else 0.0

    print(f"  Cases passed:        {passed}/{total}")
    print(f"  Critical failures:   {critical}")
    print(f"  Mean score:          {overall:.2f}  {_bar(overall, 20)}")

    print()
    print("  By dimension:")
    for dim in DIMENSIONS:
        scores = [r.dimension_scores[dim] for r in results if dim in r.dimension_scores]
        if scores:
            mean = sum(scores) / len(scores)
            print(f"    {dim:<14} {mean:.2f}  {_bar(mean, 20)}   ({len(scores)} case(s))")
    print("-" * 84)
    print()


def write_report(results: list[CaseResult], provider_label: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"eval-{stamp}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider_label,
        "cases_total": len(results),
        "cases_passed": sum(1 for r in results if r.passed),
        "critical_failures": sum(len(r.critical_failures) for r in results),
        "mean_score": round(sum(r.score for r in results) / len(results), 3) if results else 0.0,
        "dimension_means": {
            dim: round(
                sum(r.dimension_scores[dim] for r in results if dim in r.dimension_scores)
                / max(1, sum(1 for r in results if dim in r.dimension_scores)),
                3,
            )
            for dim in DIMENSIONS
            if any(dim in r.dimension_scores for r in results)
        },
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the purchasing-agent evaluation suite.")
    parser.add_argument("--provider", help="gemini | anthropic | openai | deterministic")
    parser.add_argument("--model", help="Override the model id.")
    parser.add_argument("--case", help="Run a single case by id or scenario id.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run the suite twice — rules engine, then the configured model — and compare.",
    )
    args = parser.parse_args()

    cases = load_suite()
    if args.case:
        needle = args.case.lower()
        cases = [c for c in cases if needle in c["id"].lower() or needle == c["scenario_id"].lower()]
        if not cases:
            print(f"No case matching '{args.case}'.")
            return 2

    if args.compare:
        from app.config import settings

        baseline = [run_one(c, "deterministic", None) for c in cases]
        print_report(baseline, "deterministic (rules baseline)")
        write_report(baseline, "deterministic")

        model_results = [run_one(c, settings.llm_provider, args.model) for c in cases]
        print_report(model_results, f"{settings.llm_provider}:{args.model or settings.llm_model}")
        write_report(model_results, settings.llm_provider)

        print("=" * 84)
        print("  COMPARISON — rules baseline vs model")
        print("=" * 84)
        print(f"  {'case':<28} {'rules':>8} {'model':>8}   delta")
        for base, model in zip(baseline, model_results):
            delta = model.score - base.score
            arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else "=")
            print(f"  {base.case_id:<28} {base.score:>8.2f} {model.score:>8.2f}   {arrow}{abs(delta):.2f}")
        print()
        return 0 if all(r.passed for r in model_results) else 1

    from app.config import settings

    provider = args.provider or settings.llm_provider
    results = [run_one(c, provider, args.model) for c in cases]
    label = f"{provider}:{args.model or settings.llm_model}" if provider != "deterministic" else "deterministic (rules baseline)"
    print_report(results, label)
    path = write_report(results, provider)
    print(f"  Report written to {path}")
    print()

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
