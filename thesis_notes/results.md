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

**To się zmieniło.** Kiedy te ramiona powstawały, poprawki mieszkały wyłącznie
w [`scripts/harness.py`](scripts/harness.py), a `asis` znaczyło „biblioteka bez
zmian”. PR-y #24–#34 wprowadziły je do `packages/`, więc role się odwróciły:
`fixed` to dziś biblioteka **bez żadnych modyfikacji**, a `asis` jest
*odtwarzane* na niej przez cofnięcie czterech rzeczy. Wszystkie ramiona liczone
są na **tym samym commicie** i tym samym protokole ewaluacji.

| Ramię | Co ustawia | Uzasadnienie |
|---|---|---|
| `asis` | truncation znów liczony jak stan terminalny w celu TD; Klondike bez limitu przejść przez stos; `target_update_freq` 500 w obu grach; PPO oceniane przez argmax | stan sprzed PR-ów, punkt odniesienia dla zadania 3 |
| `fixed` | dzisiejsza biblioteka, nic nie zmienione | — |
| `noloop` | `fixed` + kara `-0,05` za wejście w pozycję już widzianą w epizodzie (`repeated_position_penalty`, tylko Klondike) | [`diagnosis.md`](diagnosis.md) D3 |

> **Czego ta tabela nie mówi.** `fixed` różni się od `asis` **czterema**
> rzeczami naraz, a nie jedną: obsługą truncation ([D1](diagnosis.md)),
> skończonym `max_passes` ([D4](diagnosis.md)), kadencją `target_update_freq`
> per gra ([D5](diagnosis.md)) i regułą wyboru akcji PPO w ewaluacji
> ([D11](diagnosis.md)). W Klondike `asis` i `fixed` to w dodatku **inne
> reguły gry** — bez limitu przejść przegrana jest nieosiągalna, więc każdy
> epizod `asis` kończy się limitem kroków. Różnica `asis → fixed` jest więc
> porównaniem „przed i po PR-ach”, a nie ablacją jednego czynnika; pojedynczy
> czynnik izoluje tylko `fixed → noloop`.

Ramię `asis` jest zaimplementowane jako podklasy w
[`scripts/harness.py`](scripts/harness.py); konkretne wartości, których użył
dany przebieg, są zapisane w polu `arm_config` każdego `raw/runs/*.json`.

### Pula TEST_SOLVABLE

Solver `solve_klondike` z budżetem **20,000 węzłów** na rozdanie, uruchomiony na wszystkich 200 rozdaniach TEST:

| werdykt | rozdań |
|---|---:|
| znaleziono linię wygrywającą | **91** |
| dowiedziono, że wygrać się nie da | 0 |
| nierozstrzygnięte w budżecie | 109 |

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
| *Random* (baseline) | — | 1 | — | 9.79 | — |
| *Heuristic* (baseline) | — | 1 | — | 25.84 | — |
| *GreedyLookahead(1)* (baseline) | — | 1 | — | 9.22 | — |
| *MCTS(20)* (baseline) | — | 1 | — | 20.34 | — |
| PPO | `asis` | 3 | 2.24 ± 0.43 | **7.07 ± 0.25** | +4.83 |
| Double DQN | `asis` | 3 | 1.90 ± 1.53 | **5.88 ± 0.30** | +3.98 |
| DQN | `asis` | 3 | 3.28 ± 0.93 | **5.67 ± 0.41** | +2.39 |
| Q-learning | `asis` | 3 | 11.31 ± 0.26 | **11.33 ± 0.20** | +0.02 |
| PPO | `fixed` | 3 | 9.07 ± 0.02 | **17.09 ± 2.12** | +8.02 |
| Double DQN | `fixed` | 3 | 2.55 ± 1.22 | **6.44 ± 0.08** | +3.88 |
| DQN | `fixed` | 3 | 3.36 ± 1.07 | **6.14 ± 0.59** | +2.77 |
| Q-learning | `fixed` | 3 | 9.33 ± 0.41 | **9.37 ± 0.24** | +0.04 |
| PPO | `noloop` | 3 | 9.07 ± 0.02 | **9.73 ± 0.98** | +0.66 |
| Double DQN | `noloop` | 3 | 2.55 ± 1.22 | **6.22 ± 0.10** | +3.66 |
| DQN | `noloop` | 3 | 3.36 ± 1.07 | **6.41 ± 0.17** | +3.05 |
| Q-learning | `noloop` | 3 | 9.33 ± 0.41 | **9.48 ± 0.23** | +0.14 |

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
| PPO | `fixed` | 3 | 1.7 ± 0.3 % | **37.7 ± 1.5 %** | +36.0 pp |
| Double DQN | `fixed` | 3 | 0.0 ± 0.0 % | **12.7 ± 0.6 %** | +12.7 pp |
| DQN | `fixed` | 3 | 4.3 ± 7.5 % | **7.7 ± 4.0 %** | +3.3 pp |
| Q-learning | `fixed` | 3 | 2.2 ± 0.3 % | **2.2 ± 1.4 %** | +0.0 pp |

Wersja maszynowa, z metryką dodatkową i czasami: [`tables/macao_win_rate.csv`](tables/macao_win_rate.csv).

Rysunki: [`figures/test_comparison_macao_asis.png`](figures/test_comparison_macao_asis.png) (porównanie na TEST) · [`figures/train_curve_macao_asis.png`](figures/train_curve_macao_asis.png) (krzywe uczenia na TRAIN) · [`figures/ablation_macao.png`](figures/ablation_macao.png) (efekt poprawek).

---

## Klondike — benchmark czasu rozwiązania (pula TEST_SOLVABLE, 91 rozdań)

| agent | seedy | solve rate | cards up | ruchów do wygranej | czas do wygranej [s] |
|---|---|---|---|---|---|
| Random | 1 | 0.0 % | 11.18 | — | — |
| Heuristic | 1 | 71.4 % | 39.93 | 146 | 0.053 |
| GreedyLookahead(1) | 1 | 4.4 % | 12.24 | 169 | 0.317 |
| MCTS(20) | 1 | 38.5 % | 29.38 | 198 | 12.031 |
| ppo (asis) | 3 | 0.4 ± 0.6 % | 8.07 ± 0.26 | 116 | — |
| ppo (asis / sampled) | 3 | 46.5 ± 4.6 % | 30.62 ± 2.45 | 178 | — |
| double_dqn (asis) | 3 | 0.0 ± 0.0 % | 6.52 ± 0.45 | — | — |
| dqn (asis) | 3 | 0.0 ± 0.0 % | 6.00 ± 0.60 | — | — |
| q_learning (asis) | 3 | 0.0 ± 0.0 % | 13.80 ± 0.34 | — | — |
| ppo (fixed) | 3 | 27.5 ± 2.9 % | 23.05 ± 1.48 | 160 | — |
| ppo (fixed / argmax) | 3 | 0.7 ± 0.6 % | 8.77 ± 0.23 | 124 | — |
| double_dqn (fixed) | 3 | 0.0 ± 0.0 % | 6.75 ± 0.06 | — | — |
| dqn (fixed) | 3 | 0.0 ± 0.0 % | 6.18 ± 0.76 | — | — |
| q_learning (fixed) | 3 | 0.0 ± 0.0 % | 11.87 ± 1.14 | — | — |
| ppo (noloop) | 3 | 0.4 ± 0.6 % | 11.72 ± 2.05 | 172 | — |
| ppo (noloop / argmax) | 3 | 0.0 ± 0.0 % | 7.46 ± 0.16 | — | — |
| double_dqn (noloop) | 3 | 0.0 ± 0.0 % | 6.41 ± 0.17 | — | — |
| dqn (noloop) | 3 | 0.0 ± 0.0 % | 6.58 ± 0.28 | — | — |
| q_learning (noloop) | 3 | 0.0 ± 0.0 % | 11.87 ± 1.14 | — | — |

Średnie ruchów i czasu liczone **tylko po rozdaniach rozwiązanych** — uśrednienie limitu ruchów z rozdania, którego agent nie rozwiązał, robiłoby ze słabszego agenta szybszego. Wiersze baselinów to jeden deterministyczny pomiar; wiersze agentów uczących się to średnia ± odchylenie po 3 seedach. Czasu do wygranej nie mierzymy dla agentów uczących się — jeden przebieg sieci jest tu tańszy niż narzut pomiaru.

---

## Ablacja: czy głowica duelling na siebie zarabia? (#42)

To ablacja **architektury**, nie czwarte ramię. Ramię jest dźwignią środowiska lub protokołu, trzymaną identycznie dla każdego agenta; głowicę duelling ma tylko Double DQN. Poza nią nie zmienia się nic — ta sama funkcja straty, ten sam optymalizator, te same `hidden_sizes`, to samo ramię `fixed`, ten sam strumień rozdań TRAIN dla danego seeda i ta sama pula 200 rozdań TEST.

> **Czego sparować się nie da:** wag początkowych. Płaska i duellingowa głowica mają różne liczby parametrów, więc ten sam seed losuje inną sieć. Stąd 3 seedy i podawane odchylenie — przy n=3 nakładające się przedziały znaczą „brak wykrywalnej różnicy”, a nie „tyle samo”.

### Klondike — karty na bazach

| wariant | seedy | karty na bazach | średnie Q (legalne) | średni rozstęp Q | rozstęp jako % średniej |
|---|---|---|---|---|---|
| Double DQN (głowica duelling, domyślna) | 3 | 6.438 ± 0.079 | 1.076 | 0.0583 | 5.4 % |
| Double DQN (płaska głowica Q, `dueling=False`) | 3 | 6.095 ± 0.320 | 0.960 | 0.2969 | 38.1 % |

### Macao — win rate vs heurystyka

| wariant | seedy | win rate vs heurystyka | średnie Q (legalne) | średni rozstęp Q | rozstęp jako % średniej |
|---|---|---|---|---|---|
| Double DQN (głowica duelling, domyślna) | 3 | 0.127 ± 0.006 | 5.576 | 1.1325 | 20.3 % |
| Double DQN (płaska głowica Q, `dueling=False`) | 3 | 0.080 ± 0.028 | 5.496 | 1.1376 | 20.7 % |

Wynik pochodzi z rekordów przebiegów (`test_after`), a rozstęp Q — z zachłannej powtórki checkpointów, tą samą ścieżką kodu dla obu wariantów. Dane: [`raw/ablation_dueling.json`](raw/ablation_dueling.json); skrypt: [`scripts/ablate_dueling.py`](scripts/ablate_dueling.py); omówienie: [`diagnosis.md`](diagnosis.md) D2.

---

## Czasy

| gra | agent | ramię | seedy | trening [s/seed] | trening [min/seed] | ewaluacja [s/seed] |
|---|---|---|---|---|---|---|
| klondike | PPO | `asis` | 3 | 1860 | 31.0 | 65 |
| klondike | PPO | `fixed` | 3 | 1459 | 24.3 | 61 |
| klondike | PPO | `noloop` | 3 | 1417 | 23.6 | 62 |
| klondike | Double DQN | `asis` | 3 | 15354 | 255.9 | 87 |
| klondike | Double DQN | `fixed` | 3 | 14923 | 248.7 | 79 |
| klondike | Double DQN | `noloop` | 3 | 14743 | 245.7 | 77 |
| klondike | DQN | `asis` | 3 | 10712 | 178.5 | 65 |
| klondike | DQN | `fixed` | 3 | 9736 | 162.3 | 52 |
| klondike | DQN | `noloop` | 3 | 9436 | 157.3 | 50 |
| klondike | Q-learning | `asis` | 3 | 716 | 11.9 | 52 |
| klondike | Q-learning | `fixed` | 3 | 509 | 8.5 | 38 |
| klondike | Q-learning | `noloop` | 3 | 524 | 8.7 | 37 |
| macao | PPO | `asis` | 3 | 133 | 2.2 | 0 |
| macao | PPO | `fixed` | 3 | 104 | 1.7 | 0 |
| macao | Double DQN | `asis` | 3 | 883 | 14.7 | 0 |
| macao | Double DQN | `fixed` | 3 | 843 | 14.0 | 0 |
| macao | DQN | `asis` | 3 | 586 | 9.8 | 0 |
| macao | DQN | `fixed` | 3 | 589 | 9.8 | 0 |
| macao | Q-learning | `asis` | 3 | 36 | 0.6 | 0 |
| macao | Q-learning | `fixed` | 3 | 30 | 0.5 | 0 |

Łączny czas CPU treningu: **70.5 h**. Wszystko na CPU, jeden wątek PyTorch na proces, 6 procesów równolegle.
