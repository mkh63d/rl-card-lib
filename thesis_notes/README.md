# thesis_notes — fakty, dane i wykresy do rozdziału 6

Katalog z materiałem źródłowym dla pracy. **Nie jest to tekst pracy** i nic
w `packages/` nie zostało tu zmienione — wszystkie poprawki z
[`diagnosis.md`](diagnosis.md) są zaimplementowane jako podklasy w
[`scripts/harness.py`](scripts/harness.py), żeby biblioteka nadal zachowywała
się dokładnie tak, jak praca ją opisuje.

> **Uwaga po PR [#29](https://github.com/mkh63d/rl-card-lib/pull/29).** To
> zdanie opisuje, jak powstały liczby w tym katalogu, i pozostaje prawdziwe.
> Ale PR #29 **zmienia `packages/`**: włącza `repeated_position_penalty =
> -0,05` dla Klondike w środowisku treningowym. Po jego scaleniu domyślne
> środowisko biblioteki **nie jest już** ramieniem `asis` — powtórzenie
> pomiarów bez jawnego `repeated_position_penalty=0.0` da inne liczby niż
> tabele w tym katalogu. Sam pomiar mówi, że kara i tak nie usuwa zapętlenia
> ([`diagnosis.md`](diagnosis.md) D3).

## Co gdzie jest

| Plik | Odpowiada na |
|---|---|
| [`gymnasium.md`](gymnasium.md) | Co pochodzi z Gymnasium, a co jest napisane od zera. Wynik `check_env`. Przykład ze Stable-Baselines3. |
| [`protocol.md`](protocol.md) | Pięć pytań promotora o protokół: jedno rozdanie czy wiele, czym jest epizod, z kim gra agent w Macao, rozłączność train/test, czym są krzywe 6.2/6.3. |
| [`diagnosis.md`](diagnosis.md) | Dziesięć znalezisk w formacie objaw → przyczyna → poprawka → wynik. |
| [`results.md`](results.md) | Nowy protokół (TRAIN/TEST/TEST_SOLVABLE) i wyniki po 3 seedy na agenta. |
| [`facts.md`](facts.md) | Karta faktów: linie kodu, rozmiary przestrzeni, wersje, hashe, sprzęt, czasy. |
| [`chapter6_obsolete.md`](chapter6_obsolete.md) | Lista zdań z obecnego rozdziału 6, które przestały być prawdziwe. |
| [`tables/`](tables/) | Tabele wynikowe w CSV, ze średnią ± odchyleniem standardowym. |
| [`figures/`](figures/) | PNG (300 dpi) i SVG, czcionka ≥ 10 pt, podpisy osi po angielsku. |
| [`raw/`](raw/) | Surowe wyniki pomiarów w JSON — źródło każdej liczby powyżej. |
| [`logs/`](logs/) | Logi uruchomień. |
| [`scripts/`](scripts/) | Kod, który to wszystko wyprodukował. |
| [`issues/`](issues/) | Treści zgłoszeń GitHub — jedno na *poprawkę*, nie na objaw. Tworzone przez [`scripts/create_issues.py`](scripts/create_issues.py); utworzone adresy lądują w `issues/created.json`. |

## Jak odtworzyć

```bash
# 1. podział rozdań i klasyfikacja solverem (~40 min)
python thesis_notes/scripts/split.py

# 2. fakty o kontrakcie Gymnasium + test ze Stable-Baselines3
python thesis_notes/scripts/probe_gymnasium.py

# 3. fakty o protokole (epizody, epsilon, truncation, ...)
python thesis_notes/scripts/probe_protocol.py

# 4. pełny sweep: 4 agentów x 2 gry x 3 seedy x 3 ramiona (~20 CPU-godzin)
python thesis_notes/scripts/run_sweep_all.py --workers 6

# 5. baseliny na tej samej puli TEST (MCTS jest wolny, ~1.5 h)
python thesis_notes/scripts/baselines_on_test.py

# 6. tabele i wykresy
python thesis_notes/scripts/make_report.py
python thesis_notes/scripts/make_results_md.py
python thesis_notes/scripts/figures_concept.py

# 6b. pomiar wstępny ramienia noloop (DQN, 2 seedy, ~1 h)
python thesis_notes/scripts/probe_noloop_preliminary.py

# 7. zgłoszenia na GitHubie (najpierw --dry-run)
python thesis_notes/scripts/create_issues.py --dry-run
python thesis_notes/scripts/create_issues.py
```

Każdy przebieg zapisuje własny JSON do `raw/runs/`, a `run_sweep_all.py`
pomija te, które już istnieją — przerwany sweep wznawia się przez ponowne
uruchomienie tego samego polecenia.

## Rysunki

| Plik | Do którego rozdziału |
|---|---|
| `concept_agent_env_loop` | rozdz. 2 — pętla agent–środowisko z podpisami *s*, *a*, *r*, na konkretnej pozycji Macao |
| `concept_action_masking` | rozdz. 2/4 — 65 wyjść sieci → maska → argmax, na konkretnej ręce |
| `concept_mcts_phases` | rozdz. 2 — cztery fazy MCTS z determinizacją ukrytych kart |
| `concept_dueling` | rozdz. 2/4 — architektura duelling: wspólny trzon → V i A → agregacja |
| `train_curve_klondike_*`, `train_curve_macao_*` | rozdz. 6 — krzywe uczenia na TRAIN |
| `test_comparison_klondike_*`, `test_comparison_macao_*` | rozdz. 6 — porównanie agentów na TEST |
| `mcts_budget_sweep` | rozdz. 6 (Fig. 6.1) — MCTS win rate vs budżet symulacji |
| `ablation_klondike`, `ablation_macao` | rozdz. 6 — efekt każdej poprawki |
| `epsilon_schedule` | rozdz. 6 — faktycznie zrealizowany harmonogram eksploracji |

Każdy rysunek istnieje jako `.png` (300 dpi, do Worda) i `.svg`
(wektorowo, do LaTeX-a).
