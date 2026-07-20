"""Regenerate the HTML report from stored run records.

    python -m rl_card_lib.report.cli --results-dir ./results

Deliberately separate from the training sweep: iterating on the page should
never cost a retrain. Everything the report needs is already on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from rl_card_lib.report.html_report import HtmlReport
from rl_card_lib.report.run_record import RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rl_card_lib.report.cli", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", default="./results",
        help="Store to read (default: ./results)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output file (default: <results-dir>/index.html)",
    )
    parser.add_argument(
        "--no-embed", action="store_true",
        help="Reference figures on disk instead of embedding them, for a "
             "much smaller file that needs its figures/ directory alongside",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip chart rendering and emit tables only",
    )
    parser.add_argument(
        "--formats", default="png,svg",
        help="Figure formats to write to disk (default: png,svg)",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    store = RunStore(args.results_dir)
    runs = store.load_runs()
    if not runs:
        print(
            f"No run records under {Path(args.results_dir).resolve()}.\n"
            "Train something first: "
            "python packages/examples/scripts/run_sweep.py --episodes 200",
            file=sys.stderr,
        )
        return 1

    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    report = HtmlReport.build(
        store,
        embed=not args.no_embed,
        formats=formats,
        with_figures=not args.no_figures,
        command=f"python -m rl_card_lib.report.cli --results-dir {args.results_dir}",
    )

    out = Path(args.out) if args.out else Path(args.results_dir) / "index.html"
    written = report.write(out)
    size_mb = written.stat().st_size / 1_048_576

    print(f"Wrote {written} ({size_mb:.1f} MB) from {len(runs)} run(s)")
    for record in runs:
        figures = len(report.run_figures.get(record.run_id) or [])
        print(f"  {record.run_id:24s} {record.episode_count:6,d} episodes  "
              f"{figures} figures")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
