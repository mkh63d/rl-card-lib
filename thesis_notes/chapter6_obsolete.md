# Zdania z obecnej pracy, które po tych zmianach przestały być prawdziwe

> Cytaty pochodzą z `Praca_Inzynierska_-_MH_3.docx` (tekst wyekstrahowany
> z `word/document.xml`). Kolumna „dlaczego” odsyła do pliku, w którym jest
> dowód. Kolejność: od najpoważniejszych.
>
> Legenda:
> **✗ nieprawda** — zdanie jest sprzeczne z kodem lub z danymi ·
> **⟳ zdezaktualizowane** — było prawdziwe, przestało być po nowych pomiarach ·
> **⚠ mylące** — formalnie obronne, ale prowadzi czytelnika w złą stronę.

---

## 0. Najważniejsza zmiana: wniosek o Klondike jest artefaktem protokołu

Ten punkt trzeba przeczytać przed resztą listy, bo unieważnia nie pojedyncze
liczby, tylko **wniosek**.

Te same, już wytrenowane checkpointy, te same 200 rozdań TEST, zmieniona
**wyłącznie reguła zamiany wyjścia sieci na akcję**:

| Reguła | karty na bazach | wygrane rozdania |
|---|---:|---:|
| `argmax` nad polityką PPO — protokół raportowany w pracy | **7,54** | **0,0 %** |
| próbkowanie **tej samej** polityki PPO | **22,45** | **28,5 %** |
| *baseline losowy na tej samej puli* | *11,59* | *0,0 %* |
| *MCTS(20) na tej samej puli* | *26,80* | *37,0 %* |
| *heurystyka na tej samej puli* | *28,74* | *45,5 %* |

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| 0.1 | „On Klondike **no learner cleared the random baseline** of 12.2 cards to the foundation” | **✗ nieprawda** | PPO grający polityką, której się nauczył, osiąga **22,45** karty przy baseline **11,59** — prawie dwukrotnie powyżej. Wynik 7,4 karty z pracy pochodzi z `argmax` nad rozkładem, który PPO optymalizował jako rozkład. |
| 0.2 | „**greedy win rates were zero across all three learners**” | **✗ nieprawda** | PPO wygrywa **28,5 %** rozdań. |
| 0.3 | „the learned greedy policy is **weaker than the behaviour that produced it**” | **⚠ prawdziwe, ale opisane jako właściwość agenta** | To jest właściwość *reguły wyboru akcji*, nie agenta. W środowisku deterministycznym z ruchami odwracalnymi polityka deterministyczna, która wróci do znanego stanu, wchodzi w cykl; podczas treningu ε > 0 ten cykl przerywało. Zmierzone: udział kroków wracających do znanej pozycji spada z 78,0 % (ε = 0) do 46,8 % (ε = 0,20), a wynik rośnie z 7,5 do 20,6 karty. |
| 0.4 | „the sparse, long-horizon single-player game **stayed hard for every learner**, as the literature on Klondike would predict” | **⚠ za mocne** | Przy poprawnym odczycie polityki PPO plasuje się między `GreedyLookahead(1)` (9,22) a MCTS(20) (26,80). Klondike pozostaje trudne, ale nie „dla każdego uczącego się agenta”. |

Dowody: [`diagnosis.md`](diagnosis.md) D11 · rysunek
[`figures/action_rule_klondike.png`](figures/action_rule_klondike.png) ·
dane [`tables/action_rule_klondike.csv`](tables/action_rule_klondike.csv).

---

## A. Tabela 6.1 — hiperparametry

| # | Cytat / komórka | Status | Dlaczego |
|---|---|---|---|
| A1 | wiersz „Replay-buffer capacity”, kolumna **DQN / Double DQN: `64`** | **✗ nieprawda** | Kod ma **50 000** — `harness/learners.py:54` i `:62`, potwierdzone w `results/models/*/run.json` i w historii git od pierwszego commita tych agentów. Wygląda na przepisanie `batch_size` do wiersza pojemności. → [`diagnosis.md`](diagnosis.md) D0 |
| A2 | wiersz „Replay-buffer capacity”, kolumna **PPO: `64`** | **✗ nieprawda** | PPO nie ma bufora odtwarzania — jest on-policy. Odpowiednikiem jest `rollout_steps = 1024`, który w tabeli figuruje osobno jako „Rollout length”. → [`diagnosis.md`](diagnosis.md) D0 |
| A3 | wiersz „Epsilon schedule: 1.0 → 0.05, ×0.995 per episode” | **⚠ mylące** | Reguła jest zgodna z kodem, ale **zapisany w raporcie `epsilon_start` to 0,8647 (Klondike) i 0,7440 (Macao)**, bo ewaluacja przed treningiem sama obniża ε. Warto dopisać, że dno 0,05 jest osiągane na epizodzie **599** z 5000. → [`diagnosis.md`](diagnosis.md) D6, D7 |

Gotowa, wygenerowana z kodu tabela zastępcza:
[`tables/hyperparameters.csv`](tables/hyperparameters.csv).

---

## B. §6.2 — Experimental setup

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| B1 | „Every run reported here was executed on CPU **at commit f3afc50**.” | **✗ nieprawda** | Dwa z ośmiu przebiegów (`klondike__dqn`, `macao__dqn`) są z `fae820f`. Widać to w `host.git_commit` w ich `run.json`. → [`facts.md`](facts.md) §6 |
| B2 | „Each used **seed 0** and 20 evaluation episodes per checkpoint, so the runs are **single-seed** and the figures below carry **no cross-seed variance**.” | **⟳ zdezaktualizowane** | Nowe przebiegi to **3 seedy inicjalizacji** na każdą kombinację i raportowana jest średnia ± odchylenie standardowe. → [`results.md`](results.md) |
| B3 | „**All seeded runs are reproducible** through the per-instance generators described in section 4.9.” | **⚠ mylące** | Mechanizm istnieje i działa (`env.reset(seed=…)` daje identyczne rozdanie), ale **żaden z opisanych przebiegów go nie użył**: pętla treningowa woła `env.reset()` bez seeda, a rozdania pochodzą z entropii systemu. Zmierzone: 10 kolejnych resetów = 10 różnych rozdań; `evaluate_klondike` uruchomiony dwa razy daje 12,57 i 12,83 karty. → [`protocol.md`](protocol.md) a), d) |
| B4 | „depends on NumPy (≥ 1.21), PyTorch (≥ 2.0), Gymnasium (≥ 0.29) …” | **⚠ niepełne** | To są dolne ograniczenia z `pyproject.toml`, nie wersje pomiarowe. Pomiary wykonano na NumPy **2.4.6**, Torch **2.12.0**, Gymnasium **1.3.0**, Python **3.12.10**. → [`facts.md`](facts.md) §4 |
| B5 | „TODO: specify the exact CPU model and RAM of the training machine.” | *do uzupełnienia* | AMD Ryzen 5 3600 (6 rdzeni / 12 wątków, 3,95 GHz), 48 GB RAM, Windows 11, **bez GPU** (`torch.cuda.is_available() == False`). → [`facts.md`](facts.md) §5 |

---

## C. §6.5 — protokół ewaluacji

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| C1 | „…then evaluated greedily on **fixed deals**, with the baselines of section 6.2 measured on **the same deals** for reference.” | **✗ nieprawda** | Rozdania nie były ustalone i nie były te same. `evaluate_klondike` woła `random.seed(10_000 + seed)`, ale `KlondikeSolitaire()` tworzy `random.Random(None)` — globalny seed nie ma na nią wpływu. Zmierzone: to samo „rozdanie nr 0” pięć razy = 5 różnych rozdań. Każdy agent i każdy baseline grał na **innym** losowym zestawie 30 rozdań. → [`protocol.md`](protocol.md) d) |
| C2 | „The runs are single-seed (seed 0), so the numbers give **within-run spread** rather than cross-seed variance; **a multi-seed sweep is left to further work**.” | **⟳ zdezaktualizowane** | Wykonany: 3 seedy × 4 agentów × 2 gry × 3 ramiona. → [`results.md`](results.md) |
| C3 | „…evaluated greedily…” na **30** rozdaniach (`--eval-episodes 30`) | **⚠ za mała próba** | Przy odchyleniu `cards_up` rzędu 5–8 kart błąd standardowy średniej z n = 30 to ok. 1–1,5 karty. Różnice w rodzaju „DQN 6,6 vs Double DQN 5,3” mieszczą się w szumie. Nowa pula TEST ma **200** rozdań. → [`results.md`](results.md) |
| C4 | „The learning curves in Fig. 6.2 and Fig. 6.3 are trailing averages measured during training with exploration on…” | **⚠ niepełne, TODO** | Zdanie jest prawdziwe, ale rysunki nigdy nie zostały wstawione (w tekście stoi „TODO: embed the trailing-average plot”), a podpis nie mówi ani jakie okno (**100 epizodów**), ani jaka wielkość (Klondike: karty na bazach; Macao: udział wygranych epizodów treningowych), ani że linie baselinów na tym samym wykresie są mierzone **zachłannie**. Rozdzielczość generowanych rysunków to 150 dpi, nie 300. → [`protocol.md`](protocol.md) e) |

---

## D. §6.5 — liczby dla Macao

Wszystkie liczby z tego akapitu są do wymiany na wartości z
[`results.md`](results.md) (200 rozdań, 3 seedy, średnia ± odch. std.).
Osobno: część z nich była nieaktualna **jeszcze przed** tymi zmianami.

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| D1 | „tabular Q-learning and **plain DQN both finished at 0.0 %**” | **✗ nieprawda** | Bieżący `macao__dqn/run.json` daje 3,3 %. Wynik 0,0 % pochodzi sprzed naprawy maskowania celu TD (`fae820f`). |
| D2 | „**DQN's loss diverged during training, spanning more than three orders of magnitude**, and its policy never recovered.” | **✗ nieprawda** | W bieżących przebiegach nie ma dywergencji: Macao DQN — maksimum straty 0,53, średnia z ostatnich 500 epizodów 0,231. Klondike DQN ma jeden przejściowy skok (max 72,7), ale kończy na 0,079. |
| D3 | „…edges past both the one-ply lookahead (**26.7 %**) and MCTS at 40 simulations (**23.3 %**), though it stays below the heuristic mirror match (**53.3 %**).” | **✗ nieprawda** | Zapisany artefakt `results/baselines/macao.json` (30 epizodów) podaje 30,0 % / 26,7 % / 40,0 %. Żadna z trzech liczb w tekście nie zgadza się z artefaktem w repozytorium. Nowe wartości na 200 rozdaniach TEST — w [`results.md`](results.md). |
| D4 | „It reached a **36.7 %** greedy win rate … up from **6.7 %** … mean shaped reward of **3.75**” | **⟳ zdezaktualizowane** | Liczby z n = 30 na nieustalonych rozdaniach i jednego seeda. Do wymiany. |
| D5 | „Double DQN reached **10.0 %**” | **⟳ zdezaktualizowane** | j.w. |

---

## E. §6.5 — liczby dla Klondike

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| E1 | „**No plain-DQN run is reported for Klondike**; the value-based slot is filled by Double DQN.” | **✗ nieprawda** | `results/models/klondike__dqn/run.json` istnieje (2,7 → 6,6 karty, commit `fae820f`), czyli lepiej niż Double DQN (5,3). |
| E2 | „…the random baseline of **12.2 cards**” | **✗ nie do odtworzenia** | Artefakt `results/baselines/klondike.json` podaje 13,27. Nowy pomiar na 200 rozdaniach TEST: **11,59**. Trzy różne liczby dla tej samej wielkości — bo rozdania nie były ustalone. |
| E3 | „none came close to MCTS at 20 simulations (**27.8 cards**)” | **✗ nie do odtworzenia** | Artefakt podaje 22,0. Nowa wartość na TEST w [`results.md`](results.md). |
| E4 | „PPO improved most over training, from 2.2 to 7.4 cards, and Double DQN from 2.3 to 5.3, but both greedy figures fall well short of their own exploratory training averages (**about 19 and 11 cards** respectively)” | **⟳ zdezaktualizowane** | Do wymiany na 3-seedowe wartości; średnie treningowe też są przeliczone (Klondike PPO: bloki po 500 epizodów w zakresie 15,7–22,8; Double DQN 8,3–15,2). |
| E5 | „Tabular Q-learning ended at 11.4 cards, the best of the learners but still below random and, **unusually**, worse than before training.” | **⚠ mylące** | Nie ma tu nic nietypowego. Tablica Q rośnie o **0,836 wpisu na krok** (1 253 141 wpisów po 1 499 936 krokach) — praktycznie każdy stan jest nowy, więc wszystkie legalne akcje mają remis 0,0 i agent losuje. To jest polityka losowa, a jej „spadek” po treningu to szum próby n = 30. Potwierdzenie: w `results/solve_benchmark/klondike.json` wytrenowany Q-learning ma `cards_up = 14,56`, **dokładnie tyle samo co Random**. → [`diagnosis.md`](diagnosis.md) D8 |
| E6 | „greedy win rates were zero across all three learners, so **the solvable-only win rate that solve_klondike() was built to certify adds nothing here and is omitted**.” | **✗ nieprawda** | Sprzeczne z własnym repozytorium: `results/solve_benchmark/klondike.json` **jest** taką ewaluacją (Heurystyka 68 %, MCTS(20) 62 %, PPO 2 %). W tej pracy dochodzi do tego benchmark na puli TEST_SOLVABLE (102 rozdania). Zdanie trzeba zastąpić — propozycja gotowa w `thesis_paste_ready.md`. → [`results.md`](results.md) |
| E7 | „**all three learners**” (Klondike) | **✗ nieprawda** | Uczących się agentów jest **czterech** (Q-learning, DQN, Double DQN, PPO) i wszystkie cztery mają zapisany przebieg na Klondike. |

---

## F. §6.4 i streszczenie — MCTS

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| F1 | „…rose from about 3 %, which is below random play, to **about 87 % at 60 simulations per move**” (§6.4, powtórzone w streszczeniu i we wstępie) | **✗ nie zgadza się z artefaktem** | Zapisany sweep `results/mcts_budget_sweep/macao_mcts_budget_sweep.csv` (100 partii na punkt, `determinizations=1`, `rollout_depth=20`, seed 0) podaje przy 60 symulacjach **83 %**, a 88 % dopiero przy **120**. Poprawne sformułowanie: „ok. 83 % przy 60 symulacjach na ruch i 88 % przy 120”. → [`tables/mcts_budget_sweep.csv`](tables/mcts_budget_sweep.csv), [`figures/mcts_budget_sweep.png`](figures/mcts_budget_sweep.png) |
| F2 | „TODO: embed the sweep plot; the two anchors are about 3 % at the buggy backup and about 87 % at 60 simulations per move.” | *do uzupełnienia + korekta* | Rysunek jest gotowy: [`figures/mcts_budget_sweep.png`](figures/mcts_budget_sweep.png) / `.svg`, 300 dpi. Kotwice do poprawienia jak w F1. Uwaga: **3 % dla zbugowanego backupu nie jest punktem tego sweepu** — sweep mierzy wyłącznie kod po naprawie; 3 % to osobny, historyczny pomiar i tak trzeba je opisać w podpisie. |
| F3 | (docstring `sweep_mcts_budget.py`) „reproduces the headline anchors: ~77 % win rate at 40 simulations and ~90 % at 60” | **✗ nieprawda** | Ten sam plik CSV podaje 73 % i 83 %. Do poprawy w kodzie, nie w pracy. |

---

## G. §6.6 i wnioski rozdz. 4 / 7 — rola modułu `rules`

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| G1 | (rozdz. 4, podsumowanie) „The extension contract for a new game is seven methods, and **the shared rule helpers make satisfying them quick** rather than merely possible.” | **✗ nieuzasadnione** | Obie gry importują z `rl_card_lib.cardgames` wyłącznie obiekty domenowe (`Card, Suit, Rank, Deck, Player, CardGame`) i **nie wołają niczego z `cardgames/rules.py`**. Logikę układania mają własną, inline. → [`facts.md`](facts.md) §1 |
| G2 | (wniosek 1, rozdz. 7) „…seven methods plus one card, deck and player class, **and a set of shared rule predicates**, are enough to express games as different as…” | **✗ nieuzasadnione** | Jak G1. Predykaty istnieją, ale nie zostały użyte przez żadną z dwóch gier referencyjnych. |
| G3 | §6.6 „TODO: … Note how much of each game's rule logic is a call into the shared rules module rather than new code” | *odpowiedź: praktycznie zero* | Liczby do wstawienia: Klondike **640** wierszy / **21** metod, Macao **693** / **17**, obie **7/7** metod kontraktu, plus opcjonalny solver Klondike **132** wiersze. → [`facts.md`](facts.md) §1 |

---

## H. §6.7 — Discussion

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| H1 | „**Plain DQN diverged on Macao**, which the shared trainer neither caused nor prevented; it is an agent-level tuning problem…” | **✗ nieprawda** | Patrz D2 — w bieżącym kodzie DQN nie diverguje. Cały ten akapit („Two limitations showed up…”) trzeba przepisać. |
| H2 | „…the gap between the greedy and exploratory scores on Klondike shows that **the first-version choice of treating the observation as a state** … costs the most on the longest-horizon, most partially observable game” | **⚠ nieuzasadniona atrybucja** | Luka zachłanne-vs-eksploracyjne ma zmierzone i prostsze wyjaśnienie: polityka zachłanna **zapętla się**. 80,3 % kroków wytrenowanego DQN (83,2 % dla Double DQN) wraca do pozycji już widzianej w tym epizodzie, wobec 23,0 % dla polityki losowej; 70,2 % ruchów DQN to „dobierz kartę”. Mechanizm: w środowisku deterministycznym polityka deterministyczna, która wróci do znanego stanu, wchodzi w cykl — a `repeated_position_penalty` w `CardGameEnv` jest ustawiona na 0,0 i **nigdy nie została włączona**. Podczas treningu ε > 0 przerywa cykl, przy ewaluacji zachłannej nic go nie przerywa; stąd cała luka. Hipoteza o stanie częściowo obserwowalnym może być prawdziwa, ale nie jest tym, co pokazują dane. → [`diagnosis.md`](diagnosis.md) D2, D3 |
| H3 | „Whether the learners beat the baselines is a qualified yes: on the adversarial game a policy-gradient learner **cleared every scripted baseline**…” | **✗ nieprawda już w starych danych** | Nawet w `results/baselines/macao.json` heurystyka w lustrzanym meczu ma 40,0 %, a PPO 36,7 % — czyli PPO **nie** przebił wszystkich baselinów. Nowa wersja zdania musi wynikać z [`results.md`](results.md). |
| H4 | „the completed runs in this chapter are the evidence, and they settle the first question” | **⚠ osłabione** | Pierwsze pytanie (czy pipeline działa end-to-end) pozostaje odpowiedziane twierdząco, ale trzeba dopisać zastrzeżenie: środowisko nie przechodzi `gymnasium.utils.env_checker.check_env`, nie współpracuje z wrapperami Gymnasium ani z SB3 bez adaptera, a epizod złożony z samych nielegalnych akcji nigdy się nie kończy. → [`gymnasium.md`](gymnasium.md) |

---

## I. Streszczenie / wstęp

| # | Cytat | Status | Dlaczego |
|---|---|---|---|
| I1 | „…fixing the MCTS backup raised its Macao win rate against a random opponent from about 3 % **to about 87 % at 60 simulations per move**.” | **✗ nie zgadza się z artefaktem** | Jak F1: 83 % przy 60, 88 % przy 120. Zdanie występuje **dwa razy** (streszczenie i §6.4) — poprawić oba. |

---

## J. Rzeczy, które pozostają prawdziwe (żeby nie przepisywać za dużo)

* §6.2: „The two games present action spaces of **68** (Klondike) and **65** (Macao)” — zgadza się z kodem.
* §6.2: „state vector of **126** values on Macao and **221** on Klondike” — zgadza się.
* §6.2: „the tabular Q-learner keys on a hashed, rounded state instead” — zgadza się (`np.round(obs, 2).tobytes()`).
* §6.3 (cały): opis trzech exploitów reward hackingu i ich napraw jest zgodny z komentarzami i historią kodu. Tabela 6.5 (dawniej 6.2) nadal wymaga domiaru — nie było to przedmiotem tych prac.
* §6.5: „the runs exercise the interface end to end on both games without any change to the environment, the trainer or the encoders” — zgadza się; nowe przebiegi też tego nie zmieniały (poprawki są podklasami poza `packages/`).
* §6.1: cztery pytania badawcze pozostają dobrze postawione; zmieniają się odpowiedzi na trzecie.
* Tabela 6.1 poza wierszem „Replay-buffer capacity”: wszystkie pozostałe wartości zgadzają się z kodem — sprawdzone przez odczyt z żywych obiektów agentów.

---

## Kolejność wprowadzania poprawek (sugestia)

1. **A1, A2** — jedna komórka tabeli, największy zarzut promotora, najprostsza poprawka.
2. **B1, B3, C1, C2** — akapit o protokole; bez tego reszta liczb wisi w powietrzu.
3. **D, E** — wymiana liczb na te z [`results.md`](results.md).
4. **F1, I1** — dwie liczby MCTS w dwóch miejscach.
5. **E6** — sekcja o benchmarku na rozdaniach wygrywalnych zamiast zdania o pominięciu.
6. **G1, G2, G3** — zestrojenie zdań o module `rules` z kodem.
7. **H1, H2, H3** — przepisanie dyskusji.
