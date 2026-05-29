"""CLI: python -m mltgnt.kpi <audit.jsonl>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mltgnt.kpi import KPIReport, compute_kpis
from mltgnt.kpi._parser import parse_date_arg


def _format_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _format_text(report: KPIReport) -> str:
    failed, total = report.response_failure_detail
    retried, threads = report.re_question_detail
    lines = [
        "=== mltgnt KPI Report ===",
        f"Period: {report.period_start} ~ {report.period_end}",
        "",
        f"Response Failure Rate:  {_format_pct(report.response_failure_rate)}  "
        f"({failed} failed / {total} total)",
        f"Re-question Rate:      {_format_pct(report.re_question_rate)}  "
        f"({retried} retried / {threads} threads)",
        "Skill Resolution Rate:  N/A   (requires skill audit events)",
    ]
    return "\n".join(lines) + "\n"


def _format_json(report: KPIReport) -> str:
    payload = {
        "period_start": report.period_start,
        "period_end": report.period_end,
        "response_failure_rate": report.response_failure_rate,
        "response_failure_detail": {
            "failed": report.response_failure_detail[0],
            "total": report.response_failure_detail[1],
        },
        "re_question_rate": report.re_question_rate,
        "re_question_detail": {
            "retried": report.re_question_detail[0],
            "total_threads": report.re_question_detail[1],
        },
        "skill_resolution_rate": report.skill_resolution_rate,
    }
    return json.dumps(payload, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mltgnt.kpi", description="Compute KPIs from audit.jsonl")
    parser.add_argument("audit_path", type=Path, help="Path to audit.jsonl")
    parser.add_argument("--since", type=parse_date_arg, default=None, help="Include events on/after YYYY-MM-DD")
    parser.add_argument("--until", type=parse_date_arg, default=None, help="Include events on/before YYYY-MM-DD")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    args = parser.parse_args(argv)

    try:
        report = compute_kpis(args.audit_path, since=args.since, until=args.until)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output_format == "json":
        sys.stdout.write(_format_json(report))
    else:
        sys.stdout.write(_format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
