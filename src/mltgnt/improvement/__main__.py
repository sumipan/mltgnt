from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mltgnt.improvement.loop import run_improvement_cycle
from mltgnt.improvement.reporter import format_cycle_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--persona-dir", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--since", type=int, default=7)
    args = parser.parse_args()

    try:
        result = run_improvement_cycle(
            args.audit,
            args.persona_dir,
            args.skills_dir,
            since_days=args.since,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc

    print(format_cycle_report(result))


if __name__ == "__main__":
    main()
