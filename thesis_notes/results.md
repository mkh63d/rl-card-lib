# Wyniki po wprowadzeniu podziału train/test

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

### Pula TEST_SOLVABLE

Solver `solve_klondike` z budżetem **20,000 węzłów** na rozdanie, uruchomiony na wszystkich 200 rozdaniach TEST:

| werdykt | rozdań |
|---|---:|
| znaleziono linię wygrywającą | **102** |
| dowiedziono, że wygrać się nie da | 0 |
| nierozstrzygnięte w budżecie | 98 |

> **Uwaga do podpisu tabeli.** To jest **dolne ograniczenie** liczby rozdań wygrywalnych, a nie ich liczba. Rozdanie nierozstrzygnięte nie jest rozdaniem przegranym — solverowi skończył się budżet. Podpis musi brzmieć „rozdania, dla których solver znalazł linię wygrywającą”.


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

---

## Klondike — karty odłożone na stosy bazowe

Metryka nagłówkowa: *cards to foundation (0–52)*, mierzona zachłannie na pełnej puli 200 rozdań TEST.

| agent | ramię | seedy | TEST przed | TEST po | Δ |
|---|---|---|---|---|---|
| *Random* (baseline) | — | 1 | — | 11.59 | — |
| *Heuristic* (baseline) | — | 1 | — | 28.74 | — |
| *GreedyLookahead(1)* (baseline) | — | 1 | — | 9.22 | — |
| *MCTS(20)* (baseline) | — | 1 | — | 26.80 | — |
| PPO | `asis` | 3 | 2.24 ± 0.43 | **7.07 ± 0.25** | +4.83 |
| Double DQN | `asis` | 3 | 1.90 ± 1.53 | **5.88 ± 0.30** | +3.98 |
| DQN | `asis` | 3 | 3.28 ± 0.93 | **5.67 ± 0.41** | +2.39 |
| Q-learning | `asis` | 3 | 11.31 ± 0.26 | **11.33 ± 0.20** | +0.02 |
| Double DQN | `fixed` | 3 | 1.90 ± 1.53 | **5.38 ± 0.44** | +3.48 |
| DQN | `fixed` | 3 | 3.28 ± 0.93 | **5.77 ± 0.07** | +2.49 |

Wersja maszynowa, z metryką dodatkową i czasami: [`tables/klondike_cards_to_foundation.csv`](tables/klondike_cards_to_foundation.csv).

Rysunki: [`figures/test_comparison_klondike_asis.png`](figures/test_comparison_klondike_asis.png) (porównanie na TEST) · [`figures/train_curve_klondike_asis.png`](figures/train_curve_klondike_asis.png) (krzywe uczenia na TRAIN) · [`figures/ablation_klondike.png`](figures/ablation_klondike.png) (efekt poprawek).

---

## Macao — win rate przeciwko heurystyce

Metryka nagłówkowa: *win rate vs heuristic*, mierzona zachłannie na pełnej puli 200 rozdań TEST.

| agent | ramię | seedy | TEST przed | TEST po | Δ |
|---|---|---|---|---|---|
| *Random* (baseline) | — | 1 | — | 2.5 % | — |
| *Heuristic* (baseline) | — | 1 | — | 54.0 % | — |
| *GreedyLookahead(1)* (baseline) | — | 1 | — | 23.5 % | — |
| *MCTS(40)* (baseline) | — | 1 | — | 32.0 % | — |
| PPO | `asis` | 3 | 3.2 ± 2.9 % | **35.5 ± 1.7 %** | +32.3 pp |
| Double DQN | `asis` | 3 | 0.0 ± 0.0 % | **7.5 ± 1.0 %** | +7.5 pp |
| DQN | `asis` | 3 | 4.3 ± 7.5 % | **7.8 ± 0.6 %** | +3.5 pp |
| Q-learning | `asis` | 3 | 2.2 ± 0.3 % | **2.2 ± 1.4 %** | +0.0 pp |
| Double DQN | `fixed` | 3 | 0.0 ± 0.0 % | **8.7 ± 0.6 %** | +8.7 pp |
| DQN | `fixed` | 3 | 4.3 ± 7.5 % | **7.8 ± 1.0 %** | +3.5 pp |

Wersja maszynowa, z metryką dodatkową i czasami: [`tables/macao_win_rate.csv`](tables/macao_win_rate.csv).

Rysunki: [`figures/test_comparison_macao_asis.png`](figures/test_comparison_macao_asis.png) (porównanie na TEST) · [`figures/train_curve_macao_asis.png`](figures/train_curve_macao_asis.png) (krzywe uczenia na TRAIN) · [`figures/ablation_macao.png`](figures/ablation_macao.png) (efekt poprawek).

---

## Klondike — benchmark czasu rozwiązania (pula TEST_SOLVABLE, 102 rozdań)

| agent | solve rate | cards up | ruchów do wygranej | czas do wygranej [s] |
|---|---|---|---|---|
| Random | 0.0 % | 14.25 | — | — |
| Heuristic | 74.5 % | 41.17 | 156 | 0.051 |
| GreedyLookahead(1) | 5.9 % | 12.61 | 170 | 0.310 |
| MCTS(20) | 52.9 % | 34.64 | 217 | 16.732 |

Średnie ruchów i czasu liczone **tylko po rozdaniach rozwiązanych** — uśrednienie limitu ruchów z rozdania, którego agent nie rozwiązał, robiłoby ze słabszego agenta szybszego.

---

## Czasy

| gra | agent | ramię | seedy | trening [s/seed] | trening [min/seed] | ewaluacja [s/seed] |
|---|---|---|---|---|---|---|
| klondike | PPO | `asis` | 3 | 2243 | 37.4 | 60 |
| klondike | Double DQN | `asis` | 3 | 12226 | 203.8 | 70 |
| klondike | Double DQN | `fixed` | 3 | 12198 | 203.3 | 69 |
| klondike | DQN | `asis` | 3 | 8884 | 148.1 | 61 |
| klondike | DQN | `fixed` | 3 | 8883 | 148.0 | 63 |
| klondike | Q-learning | `asis` | 3 | 878 | 14.6 | 64 |
| macao | PPO | `asis` | 3 | 266 | 4.4 | 0 |
| macao | Double DQN | `asis` | 3 | 703 | 11.7 | 0 |
| macao | Double DQN | `fixed` | 3 | 692 | 11.5 | 0 |
| macao | DQN | `asis` | 3 | 491 | 8.2 | 0 |
| macao | DQN | `fixed` | 3 | 477 | 7.9 | 0 |
| macao | Q-learning | `asis` | 3 | 58 | 1.0 | 0 |

Łączny czas CPU treningu: **40.0 h**. Wszystko na CPU, jeden wątek PyTorch na proces, 6 procesów równolegle.
