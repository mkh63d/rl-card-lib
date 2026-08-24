"""Create one GitHub issue per finding, from thesis_notes/issues/*.md.

Each markdown file is the issue body; the title and labels live in the table
below so the bodies stay clean. Grouping rule: one issue per *fix* — findings
that a single change resolves are one issue, cross-referenced rather than split.

Usage:
    python thesis_notes/scripts/create_issues.py --dry-run
    python thesis_notes/scripts/create_issues.py
    python thesis_notes/scripts/create_issues.py --only 02 06
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ISSUES = os.path.join(HERE, "..", "issues")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

SPEC = [
    ("01-unseeded-deals.md",
     "Training and evaluation deals are unseeded: the \"fixed-deal\" protocol "
     "is not fixed",
     ["bug"]),
    ("02-truncation-treated-as-terminal.md",
     "A time-limit truncation is learned as a terminal state (bootstrap zeroed "
     "on 100% of Klondike episodes)",
     ["bug"]),
    ("03-illegal-action-livelock.md",
     "CardGameEnv.step() can never end an episode made of illegal actions",
     ["bug"]),
    ("04-not-a-gymnasium-env.md",
     "Environments do not subclass gymnasium.Env: check_env, wrappers and "
     "Stable-Baselines3 all reject them",
     ["bug"]),
    ("05-evaluation-consumes-epsilon.md",
     "Evaluation mutates the exploration schedule (recorded epsilon_start is "
     "0.8647, not 1.0)",
     ["bug"]),
    ("06-repeated-position-penalty-never-enabled.md",
     "repeated_position_penalty is implemented but never enabled; trained "
     "greedy policies loop on 80% of steps",
     ["bug"]),
    ("07-klondike-loss-reward-unreachable.md",
     "Klondike LOSS_REWARD is unreachable with the default max_passes=None",
     ["bug"]),
    ("08-target-update-freq-not-comparable.md",
     "target_update_freq=500 means 1.7 episodes on Klondike but 10.9 on Macao",
     ["enhancement"]),
    ("09-mcts-sweep-docstring-numbers.md",
     "sweep_mcts_budget.py docstring anchors contradict its own committed CSV",
     ["documentation"]),
    ("10-greedy-eval-discards-learned-policy.md",
     "Greedy evaluation discards the learned policy: PPO scores 7.5 cards by "
     "argmax and 22.5 by sampling the same weights",
     ["bug"]),
]

FOOTER = """

---

*Filed from the measurement pass in `thesis_notes/`. Every number above is
reproducible with the scripts in `thesis_notes/scripts/`; raw measurements are
in `thesis_notes/raw/`.*
"""


def body(filename: str) -> str:
    with open(os.path.join(ISSUES, filename), "r", encoding="utf-8") as handle:
        return handle.read().rstrip() + FOOTER


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only these numeric prefixes, e.g. --only 02 06")
    args = parser.parse_args(argv)

    created = []
    for filename, title, labels in SPEC:
        prefix = filename.split("-", 1)[0]
        if args.only and prefix not in args.only:
            continue
        text = body(filename)
        if args.dry_run:
            print(f"[dry-run] {prefix}  {title}")
            print(f"          labels={','.join(labels)}  "
                  f"body={len(text)} chars")
            continue
        cmd = ["gh", "issue", "create", "--title", title, "--body", text]
        for label in labels:
            cmd += ["--label", label]
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAILED {prefix}: {result.stderr.strip()}", file=sys.stderr)
            return 1
        url = result.stdout.strip().splitlines()[-1]
        created.append({"file": filename, "title": title, "url": url})
        print(f"created {prefix}  {url}", flush=True)

    if created:
        path = os.path.join(ISSUES, "created.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(created, handle, indent=2)
        print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
