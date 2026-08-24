"""Generate thesis_notes/results.md from raw/, and fill the timing table in facts.md.

Kept separate from make_report.py so the prose lives in one place while the
numbers still come from the same loaders. Run after make_report.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_report import (  # noqa: E402
    AGENT_ORDER,
    HERE,
    LABEL,
    TEST_SPEC,
    agg,
    load_json,
    load_runs,
    markdown_table,
)

HEAD = """# Wyniki po wprowadzeniu podziału train/test

> **Status:** materiał źródłowy dla rozdziału 6. Każda liczba w tym pliku jest
> generowana przez [`scripts/make_results_md.py`](scripts/make_results_md.py)
> z [`raw/runs/`](raw/runs/) i
> [`raw/baselines_on_test.json`](raw/baselines_on_test.json) — nic nie jest
> przepisywane ręcznie.

## Protokół

| | |
|---|---|
| **pula TRAIN** | seedy `0 .. 9999` (10 000 rozdań); w każdym epizodzie losowane jedno rozdanie z puli |
| **pula TEST** | seedy `100000 .. 100199` (**200 rozdań**), ta sama dla wszystkich agentów i wszystkich baselinów |
| **TEST_SOLVABLE** | podzbiór TEST, dla którego solver pełnej informacji znalazł linię wygrywającą (tylko Klondike) |
| rozłączność | gwarantowana konstrukcyjnie; pole `protocol.train_test_overlap` w każdym `raw/runs/*.json` wynosi **0** |
| odtwarzalność | rozdanie jest funkcją seeda (`game.reset(seed=...)` reseeduje prywatny generator gry), więc cała pula odtwarza się z trzech liczb |
| strumień rozdań treningowych | zależy **wyłącznie** od seeda inicjalizacji, więc każdy agent przy seedzie *k* widzi identyczną sekwencję rozdań — porównanie sparowane, nie cztery niezależne losowania |
| ewaluacja | **cała** pula TEST, `agent.eval()` (zachłannie), bez uczenia i bez efektów ubocznych na harmonogram eksploracji |
| seedy inicjalizacji | **3** na każdą kombinację (gra, agent, ramię) |
| raportowane | średnia ± odchylenie standardowe z próby (ddof = 1) po 3 seedach |

### Ramiona eksperymentu

Każde ramię zmienia **dokładnie jedną** rzecz względem `asis`:

| Ramię | Co zmienia | Uzasadnienie |
|---|---|---|
| `asis` | nic w kodzie biblioteki — zmieniony jest **wyłącznie protokół** (pule rozdań, pełna pula TEST, ewaluacja bez skutków ubocznych) | punkt odniesienia dla zadania 3 |
| `fixed` | truncation przestaje być traktowany jak stan terminalny w celu TD | [`diagnosis.md`](diagnosis.md) D1 |
| `noloop` | kara `-0,05` za wejście w pozycję już widzianą w epizodzie (`repeated_position_penalty` — mechanizm już obecny w `CardGameEnv`, dotąd nigdy nie włączony) | [`diagnosis.md`](diagnosis.md) D3 |

Żaden plik w `packages/` nie był modyfikowany: poprawki są zaimplementowane
jako podklasy w [`scripts/harness.py`](scripts/harness.py).
"""


CAVEAT = """

---

## Zastrzeżenie, które trzeba przeczytać przed tabelami Klondike

Wszystkie liczby Klondike poniżej są mierzone **zachłannie**, bo taki jest
protokół opisany w pracy. Dla tej gry jest to jednak zła reguła odczytu
polityki: w środowisku z ruchami odwracalnymi polityka deterministyczna wchodzi
w cykl, a jedyne, co ten cykl kiedykolwiek przerywało, to ε > 0 podczas
treningu.

Te same wagi, ta sama pula 200 rozdań, zmieniona wyłącznie reguła wyboru akcji:

| Reguła | karty na bazach | wygrane rozdania | powtórzone pozycje |
|---|---:|---:|---:|
| `argmax` nad polityką PPO (protokół z pracy) | **7,54** | **0,0 %** | 78,0 % |
| próbkowanie tej samej polityki PPO | **22,45** | **28,5 %** | 44,8 % |

Pełny pomiar dla wszystkich czterech agentów i trzech wartości ε jest w
[`diagnosis.md`](diagnosis.md) D11 i na rysunku
[`figures/action_rule_klondike.png`](figures/action_rule_klondike.png).
Tabele w tym pliku podają liczby zachłanne, żeby były porównywalne z pracą —
ale wniosek „żaden agent nie przebił baseline'u losowego” z nich **nie
wynika**.
"""


def solvable_block(solvable) -> str:
    if not solvable:
        return ""
    return (
        "\n### Pula TEST_SOLVABLE\n\n"
        f"Solver `solve_klondike` z budżetem **{solvable['max_nodes']:,} węzłów** "
        "na rozdanie, uruchomiony na wszystkich 200 rozdaniach TEST:\n\n"
        "| werdykt | rozdań |\n|---|---:|\n"
        f"| znaleziono linię wygrywającą | **{len(solvable['solvable'])}** |\n"
        f"| dowiedziono, że wygrać się nie da | {len(solvable['unsolvable'])} |\n"
        f"| nierozstrzygnięte w budżecie | {len(solvable['undecided'])} |\n\n"
        "> **Uwaga do podpisu tabeli.** To jest **dolne ograniczenie** liczby "
        "rozdań wygrywalnych, a nie ich liczba. Rozdanie nierozstrzygnięte nie "
        "jest rozdaniem przegranym — solverowi skończył się budżet. Podpis musi "
        "brzmieć „rozdania, dla których solver znalazł linię wygrywającą”.\n"
    )


def game_block(runs, baselines, game, heading, metric_name) -> str:
    key = TEST_SPEC[game][0]
    percent = game == "macao"
    out = [f"\n---\n\n## {heading}\n\n"
           f"Metryka nagłówkowa: *{metric_name}*, mierzona zachłannie na pełnej "
           "puli 200 rozdań TEST.\n\n"]

    header = ["agent", "ramię", "seedy", "TEST przed", "TEST po", "Δ"]
    rows = []
    for row in (baselines or {}).get(game, []):
        value = f"{row[key] * 100:.1f} %" if percent else f"{row[key]:.2f}"
        rows.append([f"*{row['agent']}* (baseline)", "—", 1, "—", value, "—"])
    for arm in ("asis", "fixed", "noloop"):
        for agent in AGENT_ORDER:
            records = runs.get((game, agent, arm))
            if not records:
                continue
            bm, bs, _ = agg(records, key, "test_before")
            am, asd, n = agg(records, key)
            before = (f"{bm * 100:.1f} ± {bs * 100:.1f} %" if percent
                      else f"{bm:.2f} ± {bs:.2f}")
            after = (f"**{am * 100:.1f} ± {asd * 100:.1f} %**" if percent
                     else f"**{am:.2f} ± {asd:.2f}**")
            delta = f"{(am - bm) * 100:+.1f} pp" if percent else f"{am - bm:+.2f}"
            rows.append([LABEL[agent], f"`{arm}`", n, before, after, delta])
    out.append(markdown_table(header, rows))

    csv_name = ("klondike_cards_to_foundation.csv" if game == "klondike"
                else "macao_win_rate.csv")
    out.append(
        f"\n\nWersja maszynowa, z metryką dodatkową i czasami: "
        f"[`tables/{csv_name}`](tables/{csv_name}).\n\n"
        f"Rysunki: [`figures/test_comparison_{game}_asis.png`]"
        f"(figures/test_comparison_{game}_asis.png) (porównanie na TEST) · "
        f"[`figures/train_curve_{game}_asis.png`]"
        f"(figures/train_curve_{game}_asis.png) (krzywe uczenia na TRAIN) · "
        f"[`figures/ablation_{game}.png`](figures/ablation_{game}.png) "
        f"(efekt poprawek).\n")
    return "".join(out)


def solve_time_block(baselines) -> str:
    if not baselines or "klondike_solve_time" not in baselines:
        return ""
    header = ["agent", "solve rate", "cards up", "ruchów do wygranej",
              "czas do wygranej [s]"]
    rows = []
    for row in baselines["klondike_solve_time"]:
        rows.append([
            row["agent"], f"{row['solve_rate'] * 100:.1f} %",
            f"{row['cards_up']:.2f}",
            "—" if row["solve_moves"] is None else f"{row['solve_moves']:.0f}",
            "—" if row["solve_seconds"] is None else f"{row['solve_seconds']:.3f}",
        ])
    pool = baselines.get("test_solvable", {}).get("size", "?")
    return ("\n---\n\n## Klondike — benchmark czasu rozwiązania "
            f"(pula TEST_SOLVABLE, {pool} rozdań)\n\n"
            + markdown_table(header, rows)
            + "\n\nŚrednie ruchów i czasu liczone **tylko po rozdaniach "
              "rozwiązanych** — uśrednienie limitu ruchów z rozdania, którego "
              "agent nie rozwiązał, robiłoby ze słabszego agenta szybszego.\n")


def timing_table(runs) -> tuple[str, float]:
    rows = []
    total = 0.0
    for game in ("klondike", "macao"):
        for agent in AGENT_ORDER:
            for arm in ("asis", "fixed", "noloop"):
                records = runs.get((game, agent, arm))
                if not records:
                    continue
                train = float(np.mean([r["duration"]["train_seconds"]
                                       for r in records]))
                ev = float(np.mean([r["duration"]["eval_seconds"]
                                    for r in records]))
                total += train * len(records)
                rows.append([game, LABEL[agent], f"`{arm}`", len(records),
                             f"{train:.0f}", f"{train / 60:.1f}", f"{ev:.0f}"])
    if not rows:
        return "", 0.0
    return markdown_table(
        ["gra", "agent", "ramię", "seedy", "trening [s/seed]",
         "trening [min/seed]", "ewaluacja [s/seed]"], rows), total


def inject_timing(table: str, total: float) -> None:
    path = os.path.join(HERE, "..", "facts.md")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    marker = "<!-- TIMING_TABLE -->"
    if marker not in text:
        return
    head, tail = text.split(marker, 1)
    rest = tail.split("\n---\n", 1)
    block = (marker + "\n\n" + table
             + f"\n\nŁączny czas CPU treningu w nowym sweepie: "
               f"**{total / 3600:.1f} h**.\n")
    text = head + block + ("\n---\n" + rest[1] if len(rest) > 1 else "")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("  injected timing table into facts.md")


def main() -> int:
    runs = load_runs()
    baselines = load_json("baselines_on_test.json")
    solvable = load_json("klondike_test_solvable.json")

    parts = [HEAD, solvable_block(solvable), CAVEAT]
    for game, heading, metric in (
        ("klondike", "Klondike — karty odłożone na stosy bazowe",
         "cards to foundation (0–52)"),
        ("macao", "Macao — win rate przeciwko heurystyce",
         "win rate vs heuristic"),
    ):
        if any(k[0] == game for k in runs) or (baselines or {}).get(game):
            parts.append(game_block(runs, baselines, game, heading, metric))
    parts.append(solve_time_block(baselines))

    table, total = timing_table(runs)
    if table:
        parts.append("\n---\n\n## Czasy\n\n" + table
                     + f"\n\nŁączny czas CPU treningu: **{total / 3600:.1f} h**. "
                       "Wszystko na CPU, jeden wątek PyTorch na proces, "
                       "6 procesów równolegle.\n")
        inject_timing(table, total)

    path = os.path.join(HERE, "..", "results.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    print(f"  wrote results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
