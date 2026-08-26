"""Line counts, classes and the seven-method contract, parsed rather than counted.

`facts.md` §1 quotes the size and shape of both games and asserts that each
implements all seven `CardGame` contract methods. Those numbers came from a
one-off script that was never committed, so `raw/code_facts.json` could not be
regenerated -- and PR #30 changed `klondike.py`, which made the committed counts
wrong with no way to correct them. This is that missing producer.

Everything here is read with `ast.parse`, never with a regex or by hand: a
docstring that happens to contain the word `def` does not become a method, and
the contract check asks the abstract base class which methods it declares rather
than trusting a hardcoded list.

Writes thesis_notes/raw/code_facts.json.

    python thesis_notes/scripts/probe_code_facts.py
"""

from __future__ import annotations

import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(HERE, "..", "raw")

#: The files facts.md §1 describes, keyed by the short name it uses.
FILES = {
    "klondike.py": "packages/examples/src/rl_card_lib/games/klondike.py",
    "macao.py": "packages/examples/src/rl_card_lib/games/macao.py",
    "klondike_solver.py": "packages/examples/src/rl_card_lib/games/klondike_solver.py",
    "heuristics.py": "packages/examples/src/rl_card_lib/games/heuristics.py",
    "game.py": "packages/core/src/rl_card_lib/core/game.py",
    "card_game.py": "packages/cardgames/src/rl_card_lib/cardgames/card_game.py",
}

#: The contract both games must satisfy. Derived below from whichever methods
#: the base classes mark `@abstractmethod`, so it cannot drift from the code;
#: this is only the fallback if neither base parses.
CONTRACT_FALLBACK = [
    "get_action_space_size", "get_legal_actions", "get_observation",
    "get_observation_shape", "is_game_over", "reset", "step",
]


def code_lines(source: str) -> int:
    """Non-blank, non-comment lines. Docstrings count as code, as facts.md says."""
    total = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            total += 1
    return total


def method_names(node: ast.ClassDef) -> list[str]:
    return [n.name for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def is_abstract(node) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "abstractmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
        for d in getattr(node, "decorator_list", [])
    )


def describe(path: str) -> dict:
    with open(os.path.join(REPO, path), "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)

    classes, names, abstract = [], {}, {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = method_names(node)
            classes.append([node.name, len(methods)])
            names[node.name] = methods
            found = sorted(
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and is_abstract(n)
            )
            if found:
                abstract[node.name] = found

    out = {
        "path": path,
        "total_lines": len(source.splitlines()),
        "code_lines": code_lines(source),
        "classes": classes,
        "methods_in_classes": sum(count for _, count in classes),
        "module_level_functions": sum(
            1 for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
    }
    if names:
        out["method_names"] = names
    if abstract:
        out["abstract_methods"] = abstract
    return out


def contract_from(report: dict) -> list[str]:
    """The seven methods, taken from whichever base class declares them.

    `core/game.py` and `cardgames/card_game.py` are asserted in facts.md to
    carry an identical abstract list; deriving it here means that claim is
    checked on every run instead of being restated.
    """
    declared = []
    for name in ("game.py", "card_game.py"):
        for methods in report.get(name, {}).get("abstract_methods", {}).values():
            declared.append(sorted(methods))
    if not declared:
        return list(CONTRACT_FALLBACK)
    common = set(declared[0])
    for other in declared[1:]:
        common &= set(other)
    return sorted(common)


def main() -> int:
    report = {name: describe(path) for name, path in FILES.items()}

    contract = contract_from(report)
    report["_contract"] = {
        "methods": contract,
        "count": len(contract),
        "bases_agree": len({
            tuple(sorted(m))
            for name in ("game.py", "card_game.py")
            for m in report[name].get("abstract_methods", {}).values()
        }) <= 1,
    }

    for name in ("klondike.py", "macao.py"):
        game_class = report[name]["classes"][0][0]
        implemented = set(report[name]["method_names"][game_class])
        report[name]["contract_implemented"] = sorted(
            m for m in contract if m in implemented)
        report[name]["contract_missing"] = sorted(
            m for m in contract if m not in implemented)
        report[name]["own_methods"] = report[name]["methods_in_classes"]

    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, "code_facts.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"contract: {len(contract)} methods, bases agree: "
          f"{report['_contract']['bases_agree']}")
    for name in FILES:
        row = report[name]
        extra = ""
        if "contract_missing" in row:
            extra = (f"  contract {len(row['contract_implemented'])}/"
                     f"{len(contract)}"
                     + (f"  MISSING {row['contract_missing']}"
                        if row["contract_missing"] else ""))
        print(f"  {name:22s} {row['total_lines']:4d} lines "
              f"({row['code_lines']:4d} code)  "
              f"{row['methods_in_classes']:2d} methods{extra}")
    print(f"Wrote {os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
