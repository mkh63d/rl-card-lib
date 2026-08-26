# Karta faktów — zgodność tekstu pracy z kodem

> **Status:** wszystkie liczby wygenerowane z repozytorium, nie przepisane.
> Źródła: [`raw/code_facts.json`](raw/code_facts.json),
> [`raw/host.json`](raw/host.json), [`raw/protocol_probe.json`](raw/protocol_probe.json),
> [`raw/runs/`](raw/runs/), `git log`.
> Data wygenerowania: 2026-08-24.
> Wersję maszynową tabel znajdziesz w [`tables/`](tables/).

---

## 1. Rozmiar i kształt obu gier

Zliczone przez parser AST (`ast.parse`), nie ręcznie.

| Metryka | Klondike | Macao | Uwaga |
|---|---:|---:|---|
| plik | `games/klondike.py` | `games/macao.py` | |
| wiersze pliku (z komentarzami i docstringami) | **665** | **693** | Klondike urósł z 640 w PR #30 (skończone `max_passes`) |
| wiersze niepuste, niekomentarzowe | 440 | 493 | docstringi liczone jako kod |
| klasy | 1 (`KlondikeSolitaire`) | 1 (`Macao`) | |
| metody klasy gry | **21** | **17** | |
| funkcje modułu | 1 (`_clone_cards`) | 1 (`_clone_cards`) | |
| metody kontraktu `CardGame` zaimplementowane | **7 / 7** | **7 / 7** | |
| dodatkowo: solver | `klondike_solver.py`, **132** wiersze, 3 funkcje | – | opcjonalny, poza kontraktem |
| dodatkowo: heurystyka | `KlondikeHeuristicAgent`, 8 metod | `MacaoHeuristicAgent`, 3 metody | wspólny plik `heuristics.py`, 299 wierszy |

**Kontrakt siedmiu metod** — potwierdzony jako `@abstractmethod` w dwóch
miejscach o identycznej liście:
[`core/game.py`](../packages/core/src/rl_card_lib/core/game.py) (163 wiersze,
19 metod) i [`cardgames/card_game.py`](../packages/cardgames/src/rl_card_lib/cardgames/card_game.py)
(63 wiersze, 13 metod):

```
reset · step · get_legal_actions · get_observation ·
get_action_space_size · get_observation_shape · is_game_over
```

Obie gry implementują wszystkie siedem i **żadnej nie brakuje**.

> **Uwaga do §6.6 i do wniosków rozdz. 7.** Praca dwukrotnie przypisuje łatwość
> dodania gry „shared rule helpers”. W kodzie obie gry importują z
> `rl_card_lib.cardgames` wyłącznie obiekty domenowe
> (`Card, Suit, Rank, Deck, Player, CardGame`) i **nie wołają nic z**
> `cardgames/rules.py`. Logikę układania/kolorów mają własną, inline. Zdanie
> o „shared rule predicates” trzeba więc albo usunąć, albo przenieść do dalszych
> prac — jest to już zaproponowane w `thesis_paste_ready.md` §7.

---

## 2. Przestrzenie obserwacji i akcji

Odczytane z żywych obiektów `CardGameEnv`, nie z komentarzy.

| | Klondike | Macao (2 graczy) |
|---|---|---|
| `observation_space` | `Box(0.0, 1.0, (221,), float32)` | `Box(0.0, high, (126,), float32)`, gdzie `high` = 1.0 poza cechą rozmiaru ręki przeciwnika: 52/15 ≈ 3.467 |
| `action_space` | `Discrete(68)` | `Discrete(65)` |
| rozbicie obserwacji | 52 × 4 (lokalizacja i widoczność każdej karty) + 4 (wierzchołki baz, znormalizowane) + 7 (rozmiary kolumn / 19) + 1 (kupka odkryta / 24) + 1 (talia / 24) = **221** | 52 (ręka, binarnie) + 52 (wierzchołek stosu, one-hot) + 4 (żądany kolor) + 13 (żądana figura) + 2 (faza deklaracji) + 1 (kara dobrania / 15) + 1 (ręka przeciwnika / 15) + 1 (talia / 52) = **126** |
| rozbicie akcji | 0 dobierz/przełóż · 1–7 z kupki na kolumnę · 8–11 z kupki na bazę · 12–18 wierzch kolumny na bazę · 19–67 kolumna→kolumna (`19 + from*7 + to`) = **68** (7 pozycji `from == to` jest martwych) | 0–51 zagraj tę kartę · 52 dobierz · 53 pas · 54–57 zadeklaruj kolor · 58–64 zadeklaruj figurę = **65**, wszystkie używane |
| typowa liczba legalnych akcji | zmienna, rzędu 3–10 | 2–4 (np. rozdanie `seed=100003`: 4 z 65) |
| `MaskedCardGameEnv` | `Dict('action_mask': MultiBinary(68), 'observation': Box(…221…))` | `Dict('action_mask': MultiBinary(65), 'observation': Box(…126…))` |

Zgodność z pracą: §6.2 podaje „state vector of 126 values on Macao and 221 on
Klondike” i „action spaces of 68 (Klondike) and 65 (Macao)” — **zgadza się**.

---

## 3. Kształt epizodu

Zmierzone na 200 rozdaniach z puli TEST; pełna tabela w
[`tables/episode_shape.csv`](tables/episode_shape.csv).

| | Klondike | Macao |
|---|---|---|
| `max_steps` | **300** | **200** |
| co kończy epizod „normalnie” | 52 karty na bazach (wygrana) albo brak legalnych ruchów | ktoś pozbył się kart |
| brak legalnych ruchów przy `max_passes=None` | **niemożliwy** — akcja 0 zawsze legalna | – |
| `truncated` przy polityce losowej | **100,0 %** | 97,0 % (dwóch losowych) |
| `truncated` przy heurystyce | 54,5 % | 0,5 % (dwie heurystyki) |
| średnia długość epizodu, polityka losowa | 300,0 | 195,9 |
| średnia długość epizodu, heurystyka | 230,7 | 38,7 |
| średnia długość epizodu w treningu (zapisane przebiegi) | 299,2–300,0 (PPO 272,3) | 45,1–64,6 |
| nagroda terminalna przy `truncated` | **brak** (tylko koszt ruchu −0,01) | różnica rozmiarów rąk × 0,1 |

---

## 4. Wersje bibliotek

| Pakiet | Wersja użyta do pomiarów | Deklaracja w `pyproject.toml` |
|---|---|---|
| Python | **3.12.10** | `>=3.10` (CI: 3.10, 3.11, 3.12) |
| NumPy | **2.4.6** | `numpy>=1.21` |
| PyTorch | **2.12.0** (CPU) | `torch>=2.0` |
| Gymnasium | **1.3.0** | `gymnasium>=0.29.0` |
| Matplotlib | **3.10.9** | `matplotlib>=3.5` |
| tqdm | **4.67.3** | `tqdm>=4.64` |
| stable-baselines3 | **2.9.0** | *nie jest zależnością biblioteki* — zainstalowana wyłącznie do testu interoperacyjności z §5 [`gymnasium.md`](gymnasium.md) |

> **Uwaga do §6.2 pracy.** Zdanie „depends on NumPy (≥ 1.21), PyTorch (≥ 2.0),
> Gymnasium (≥ 0.29), Matplotlib (≥ 3.5) … and tqdm (≥ 4.64)” opisuje **dolne
> ograniczenia**, a nie wersje, na których zmierzono wyniki. Warto dopisać
> zdanie z faktycznymi wersjami, bo NumPy 2.x i Torch 2.12 to inny świat niż
> 1.21 / 2.0.

---

## 5. Sprzęt i czas

| | |
|---|---|
| procesor | **AMD Ryzen 5 3600**, 6 rdzeni / 12 wątków, 3,95 GHz |
| pamięć | **48 GB** (51 446 001 664 B) |
| GPU / CUDA | **niedostępne** — `torch.cuda.is_available() == False`; wszystkie przebiegi na CPU |
| system | Windows 11 Pro, build 10.0.26200 |
| wątki PyTorch w przebiegach z pracy | domyślne (6) |
| wątki PyTorch w nowych przebiegach | **1** na proces, **10** procesów równolegle — zmierzone jako szybsze na proces (0,92 s/epizod przy 1 wątku vs 1,29 s/epizod przy 6) |
| narzut zrównoleglenia | przy 10 procesach naraz jeden przebieg trwa ok. **2,4×** dłużej niż zmierzone w izolacji (Klondike DQN: 182 min zamiast 77) — rdzenie i przepustowość pamięci są dzielone. Sumaryczny czas CPU w tabelach poniżej jest więc *zmierzony pod obciążeniem*, nie ekstrapolowany z pojedynczego przebiegu |

### Czas treningu — przebiegi opisane w pracy (5000 epizodów, 1 seed)

| Gra | Agent | Trening [s] | Trening | Ewaluacja [s] |
|---|---|---:|---|---:|
| Klondike | Double DQN | 8 184 | 2 h 16 min | 8,1 |
| Klondike | DQN | 6 460 | 1 h 48 min | 9,3 |
| Klondike | PPO | 1 310 | 22 min | 7,0 |
| Klondike | Q-learning | 530 | 9 min | 5,5 |
| Macao | Double DQN | 455 | 8 min | 2,3 |
| Macao | DQN | 448 | 7 min | 2,0 |
| Macao | PPO | 92 | 1,5 min | 1,5 |
| Macao | Q-learning | 25 | 25 s | 0,8 |
| **razem** | | **17 504** | **4 h 52 min** | 36,5 |

Do tego baseliny: MCTS(20) na Klondike **635 s na 30 rozdań** (to najdroższy
pojedynczy pomiar w repozytorium), MCTS(40) na Macao 50 s na 30 rozdań.

### Czas treningu — nowe przebiegi (3 seedy, 3 ramiona)

Wypełniane automatycznie z `raw/runs/*.json`; patrz
[`tables/ablation_fixes.csv`](tables/ablation_fixes.csv), kolumna
`train_seconds_mean`.

<!-- TIMING_TABLE -->

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

Łączny czas CPU treningu w nowym sweepie: **70.5 h**.

---

## 6. Commity

| Etykieta | Pełny hash | Data | Czego dotyczy |
|---|---|---|---|
| `f3afc50` | `f3afc50fc0df80d9fa91e6020cb22befc657504d` (merge PR #6, *feat/report-example-artifacts*) | 2026-07-21 12:24 | commit 6 z 8 przebiegów opisanych w pracy |
| `fae820f` | `fae820f7ec430ce9a72f284af47eee4c920e5cbc` (merge PR #8, *fix/dqn-legal-action-masking*) | 2026-07-21 19:43 | commit 2 przebiegów DQN (Klondike i Macao) — po naprawie maskowania celu TD |
| `2bd42ab` | `2bd42abce993308c1f9dc6fa7306e2b022d9e432` | 2026-07-22 21:50 | „Add MCTS simulation-budget sweep for Macao”; HEAD **poprzedniego** kompletu pomiarów, dziś w [`raw/archive_2bd42ab/`](raw/archive_2bd42ab/) |
| `3167467` | `31674673714e84cc18e87ef541e9402d8d41335b` (merge PR #34, *fix/seed-time-limit-bootstrap-deals*) | 2026-08-25 13:14 | **HEAD w chwili wykonania bieżących pomiarów**; ostatni z serii PR-ów #24–#34 |

> **Uwaga do §6.2 pracy.** Zdanie „Every run reported here was executed on CPU
> at commit f3afc50” jest **nieprawdziwe dla dwóch z ośmiu przebiegów**
> (`klondike__dqn` i `macao__dqn` są z `fae820f`). Poprawka jest już
> zaproponowana w `thesis_paste_ready.md` §6.2.
>
> **Bieżące pomiary są z `3167467`**, czyli z biblioteki **po** scaleniu
> poprawek z [`diagnosis.md`](diagnosis.md) — zdanie „żaden plik w `packages/`
> nie był modyfikowany”, prawdziwe dla `2bd42ab`, dla tego kompletu **już nie
> obowiązuje**. Ramię `fixed` to biblioteka bez zmian; ramię `asis` odtwarza
> stan sprzed poprawek przez podklasy w
> [`scripts/harness.py`](scripts/harness.py). Konkretne wartości każdego
> przebiegu zapisuje pole `arm_config` w `raw/runs/*.json`. Stan roboczy przy
> pomiarach: `3167467` + zmiany w `thesis_notes/scripts/` z tego samego
> przemiarowania (gałąź `chore/rerun-metrics-on-head`).
>
> **Uściślenie do pola `host.git_commit`.** 55 z 60 przebiegów ma tam
> `3167467`; pięć ostatnich (`klondike q_learning noloop s1/s2`,
> `macao ppo fixed s2`, `macao q_learning fixed s1/s2`) ma hashe commitów
> z tej gałęzi, bo kończyły się już w trakcie commitowania notatek.
> **Nie oznacza to różnicy w mierzonym kodzie**: żaden z tych commitów nie
> dotyka `packages/` (`git diff --name-only 3167467 <commit> -- packages/`
> jest pusty dla każdego z nich), więc biblioteka była bajt w bajt ta sama we
> wszystkich 60 przebiegach.

---

## 7. Pule rozdań (nowy protokół)

| Pula | Seedy | Liczność | Do czego |
|---|---|---:|---|
| **TRAIN** | 0 – 9 999 | 10 000 | losowanie jednego rozdania na epizod treningowy |
| **TEST** | 100 000 – 100 199 | **200** | ewaluacja zachłanna wszystkich agentów i baselinów |
| **TEST_SOLVABLE** | podzbiór TEST | **102** | benchmark czasu rozwiązania (tylko Klondike) |

Rozłączność TRAIN ∩ TEST jest gwarantowana konstrukcyjnie (10 000 ≤ 100 000)
i sprawdzana w każdym przebiegu — pole
`protocol.train_test_overlap` w `raw/runs/*.json` wynosi **0** dla wszystkich
przebiegów.

**TEST_SOLVABLE** wyznaczono, uruchamiając solver pełnej informacji
(`solve_klondike`, budżet **20 000** węzłów) na wszystkich 200 rozdaniach TEST:

| Werdykt | Liczba rozdań |
|---|---:|
| **wygrywalne (dowód)** | **102** |
| niewygrywalne (dowód wyczerpania) | **0** |
| nierozstrzygnięte w budżecie | **98** |

Ważne zastrzeżenie do pracy: 102 to **dolne ograniczenie** liczby wygrywalnych
rozdań w puli, a nie ich rzeczywista liczba. Literatura podaje dla wariantu
draw-1 z nieograniczonymi przejściami ok. 80–90 % rozdań wygrywalnych; przy
budżecie 20 000 węzłów solver po prostu nie zdążył rozstrzygnąć pozostałych 98.
Zdanie w podpisie tabeli musi brzmieć „rozdania, dla których solver **znalazł**
linię wygrywającą”, a nie „rozdania wygrywalne”.

---

## 8. Liczby, które warto mieć pod ręką przy pisaniu

| Fakt | Wartość | Skąd |
|---|---|---|
| pakietów w monorepo | 5 (`cardgames`, `core`, `examples`, `report`, `visualizer`) | `packages/` |
| plików `.py` w `packages/` | 69 | `glob` |
| konkretnych agentów | **9**: `RandomAgent`, `GreedyLookaheadAgent`, `MCTSAgent`, `KlondikeHeuristicAgent`, `MacaoHeuristicAgent`, `QLearningAgent`, `DQNAgent`, `DoubleDQNAgent`, `PPOAgent` (plus 3 klasy abstrakcyjne: `Agent`, `GameAwareAgent`, `HeuristicAgent`) | `agents/` + `games/heuristics.py` |
| metody kontraktu gry | 7 | `@abstractmethod` |
| ε osiąga 0,05 | epizod **599** z 5000 (88 % treningu na dnie) | analiza `0.995^n` |
| aktualizacji sieci docelowej w 5000 epizodach | Klondike **3 000**, Macao **460** | 500 kroków gradientu ≈ 500 kroków środowiska |
| pojemność bufora odtwarzania | **50 000** (nie 64) | `learners.py:54,62` |
| rozmiar tablicy Q po treningu | Klondike **1 253 141**, Macao **318 688** | serie w `results/models/*__q_learning/run.json` |
| przyrost tablicy Q na krok | Klondike 0,836, Macao 0,987 | j.w. |
| rozmiar checkpointu tablicy Q | Klondike **1,76 GB**, Macao 0,36 GB (na seed) | `thesis_notes/checkpoints/` |
| Q-learning na Klondike, TEST 200 rozdań, 3 seedy | 11,31 ± 0,26 → **11,33 ± 0,20** karty przy baseline losowym 11,59 | `tables/klondike_cards_to_foundation.csv` |
| heurystyka Klondike, 200 rozdań TEST | 45,5 % wygranych, 28,74 karty | `tables/episode_shape.csv` |
| heurystyka Macao przeciw sobie, gracz 0 | 55,5 % wygranych | j.w. (przewaga pierwszego ruchu) |
| MCTS Macao, 120 symulacji | 88 % przeciw losowemu (100 partii) | `results/mcts_budget_sweep/` |
