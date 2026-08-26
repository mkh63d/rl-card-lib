# Protokół eksperymentu — odpowiedzi na pytania promotora

> **Status:** materiał źródłowy dla rozdziału 6. Każda odpowiedź jest oparta na
> cytacie z kodu i, gdzie się dało, na pomiarze.
> Surowe dane: [`raw/protocol_probe.json`](raw/protocol_probe.json) ·
> log: [`logs/protocol_probe.log`](logs/protocol_probe.log) ·
> skrypt: [`scripts/probe_protocol.py`](scripts/probe_protocol.py)
> Wszystkie pomiary w tym pliku dotyczą kodu **przed** poprawkami; opis
> poprawek jest w [`diagnosis.md`](diagnosis.md), nowy protokół w
> [`results.md`](results.md).

## Skrót odpowiedzi

| Pytanie | Odpowiedź |
|---|---|
| **a)** jedno rozdanie czy wiele? | **Wiele, losowych i nieodtwarzalnych.** `env.reset()` w pętli treningowej jest wołane **bez seeda**, seed nigdy nie jest inkrementowany, bo nigdy nie jest przekazywany. |
| **b)** czym jest epizod? | Klondike: jedno rozdanie, limit 300 ruchów. Zmierzone: polityka losowa kończy **100 %** epizodów przez `truncated`, heurystyka 54,5 %. W zapisanych przebiegach treningowych średnia długość epizodu wynosi 299,2–300,0 z limitu 300, czyli praktycznie każdy epizod dobija do limitu. Macao: jedno rozdanie 2-osobowe, limit 200 tur; udział `truncated` zależy od siły graczy (0,5 % dla dwóch heurystyk, 97 % dla dwóch agentów losowych). |
| **c)** z kim gra agent w Macao? | **Jeden** przeciwnik, `MacaoHeuristicAgent` (heurystyka, nie snapshot), stały przez cały trening. W ewaluacji **dwaj inni** przeciwnicy: `RandomAgent` i `MacaoHeuristicAgent`. Klasa trenera to `SelfPlayTrainer`, ale flaga `self_play` = `False`. |
| **d)** czy zbiory train/test są rozłączne? | **Zbiory nie są zdefiniowane.** Obie pule to nieodtwarzalne losowania z pełnej przestrzeni rozdań. Wyciek testu jest praktycznie niemożliwy (52! rozdań), ale **deklaracja „fixed deals” w kodzie i w pracy jest nieprawdziwa**, a agenci są porównywani na **różnych** rozdaniach. To jest błąd metodologiczny — naprawiony w [`results.md`](results.md). |
| **e)** czym są krzywe z Fig. 6.2/6.3? | Średnia krocząca po **100 epizodach**, mierzona **w trakcie treningu, z włączoną eksploracją**; dla Klondike wielkością są karty odłożone na bazy, dla Macao — udział wygranych epizodów treningowych. |

---

## a) Czy agent trenuje na jednym rozdaniu, czy na wielu losowych?

### Co robi kod

Pętla treningowa woła `reset()` **bez żadnego argumentu**, w obu trenerach:

```python
# packages/core/src/rl_card_lib/trainer/trainer.py:195   (Trainer._run_episode)
observation, info = self.env.reset()

# packages/core/src/rl_card_lib/trainer/trainer.py:423   (SelfPlayTrainer._run_episode)
observation, info = self.env.reset()
```

`CardGameEnv.reset()` przekazuje seed do gry **tylko wtedy, gdy go dostanie**:

```python
# packages/core/src/rl_card_lib/env/card_game_env.py:93-97
self._step_count = 0
if seed is not None and self._game_reset_accepts_seed():
    observation = self.game.reset(seed=seed)
else:
    observation = self.game.reset()
```

A `Game.reset()` bez seeda **nie resetuje** prywatnego generatora, tylko tasuje
nim dalej:

```python
# packages/examples/src/rl_card_lib/games/klondike.py:120-123
if seed is not None:
    self._rng = random.Random(seed)
self.deck = Deck()
self.deck.shuffle(rng=self._rng)
```

`self._rng` powstaje w konstruktorze jako `random.Random(seed)` z domyślnym
`seed=None` ([klondike.py:98](../packages/examples/src/rl_card_lib/games/klondike.py#L98),
[macao.py:109](../packages/examples/src/rl_card_lib/games/macao.py#L109)), czyli
zasiany z entropii systemu.

### Zmierzone

| Pomiar | Wynik |
|---|---|
| 10 kolejnych `env.reset()` w Klondike — ile różnych rozdań | **10 z 10** |
| 10 kolejnych `env.reset()` w Macao — ile różnych rozdań | **10 z 10** |
| ta sama sekwencja po `random.seed(1234)` + `np.random.seed(1234)` — odtwarzalna? | **nie** |
| ta sama sekwencja przez `env.reset(seed=7)` — odtwarzalna? | **tak** |

### Odpowiedź

Agent trenuje na **wielu różnych rozdaniach** — jedno świeże rozdanie na
epizod, czyli 5000 rozdań w przebiegu 5000-epizodowym. **Seed nie jest
inkrementowany, ponieważ nie jest w ogóle przekazywany.** Rozdania pochodzą
z entropii systemu operacyjnego, więc:

* przebieg jest **nieodtwarzalny** — powtórzenie tego samego skryptu z tym samym
  `--seed 0` daje inne rozdania treningowe;
* zdanie z §6.2 pracy „All seeded runs are reproducible through the per-instance
  generators described in section 4.9” jest prawdziwe o **mechanizmie**
  (`reset(seed=…)` faktycznie działa), ale **nieprawdziwe o przeprowadzonych
  przebiegach**, bo żaden z nich tego mechanizmu nie użył.

**Wniosek pozytywny:** nie ma przeuczenia na jedno rozdanie. To był realny
scenariusz do wykluczenia i został wykluczony.

---

## b) Czym dokładnie jest jeden epizod?

### Klondike

Jeden epizod = jedno rozdanie pasjansa, od rozdania kart do zakończenia.

| Cecha | Wartość | Źródło |
|---|---|---|
| `max_steps` środowiska | **300** | [registration.py:20](../packages/examples/src/rl_card_lib/games/registration.py#L20) `KLONDIKE_MAX_STEPS = 300` |
| `terminated = True` gdy | wszystkie 52 karty na bazach (wygrana) **albo** brak legalnych ruchów | [klondike.py:353-364](../packages/examples/src/rl_card_lib/games/klondike.py#L353-L364) |
| `truncated` z gry | **nigdy** — `klondike.py:367` ustawia na stałe `truncated = False` | [klondike.py:367](../packages/examples/src/rl_card_lib/games/klondike.py#L367) |
| `truncated` z środowiska | gdy `_step_count >= 300` | [card_game_env.py:135-136](../packages/core/src/rl_card_lib/env/card_game_env.py#L135-L136) |

**Kluczowy fakt — i to się zmieniło.** Domyślne `max_passes = None` *klasy*
`KlondikeSolitaire` nadal obowiązuje, a przy nieograniczonych przejściach przez
talię akcja 0 (dobierz / przełóż odkrytą kupkę z powrotem) jest **zawsze
legalna**. Rozdanie nie może wtedy „umrzeć”: gałąź `LOSS_REWARD = -1.0` jest
kodem nieosiągalnym.

Ale **gra dołączona do biblioteki nie używa już tej konfiguracji**. PR
[#30](https://github.com/mkh63d/rl-card-lib/pull/30) wprowadził
`KlondikeSolitaire.BUNDLED_MAX_PASSES = 3`, którego używa każdy bundlowany punkt
wejścia (trening, ewaluacja, baseliny, solver curating puli). Domyślna wartość
klasy została na `None`, żeby użytkownik biblioteki mógł grać bez limitu —
rozróżnienie „domyślne dla klasy” vs „reguła gry w repo” jest tu istotne.

Skutek: w ramieniu `fixed` **2–11 % epizodów Klondike kończy się terminacją**,
więc `LOSS_REWARD` przestał być kodem nieosiągalnym. Wiersze `max_passes=3`
w tabeli poniżej opisują więc dziś **konfigurację domyślną eksperymentów**, a nie
wariant.

Zmierzone na 200 rozdaniach (`seed` 100000–100199):

| Polityka | średnia liczba kroków | `terminated` | `truncated` | wygrane | martwe rozdania | karty na bazach |
|---|---:|---:|---:|---:|---:|---:|
| losowa | **300,0** | 0,0 % | **100,0 %** | 0,0 % | 0,0 % | 11,6 |
| heurystyka | 230,7 | 45,5 % | 54,5 % | 45,5 % | 0,0 % | 28,7 |
| losowa, `max_passes=3` | 290,6 | 4,5 % | 95,5 % | 0,0 % | **4,5 %** | 9,8 |
| heurystyka, `max_passes=3` | 237,5 | 40,0 % | 60,0 % | 38,5 % | 1,5 % | 25,8 |

Dla porównania — z zapisanych przebiegów treningowych (`results/models/*/run.json`,
5000 epizodów każdy) średnia liczba kroków wynosiła: DQN 299,4; Double DQN
299,2; Q-learning 300,0; PPO 272,3. Czyli **praktycznie każdy epizod treningowy
kończył się limitem 300 ruchów**, nie zakończeniem gry.

Wniosek: dla wszystkich uczących się agentów Klondike jest w praktyce zadaniem
o **stałej długości 300 kroków, kończącym się zawsze truncation**, bez żadnego
sygnału terminalnego. Konsekwencje w [`diagnosis.md`](diagnosis.md) §D1 i §D4.

### Macao

Jeden epizod = jedna partia dwuosobowa, od rozdania po 5 kart do końca.

| Cecha | Wartość | Źródło |
|---|---|---|
| `max_steps` środowiska | **200** | [registration.py:21](../packages/examples/src/rl_card_lib/games/registration.py#L21) `MACAO_MAX_STEPS = 200` |
| `max_turns` gry | **200** | [macao.py:84](../packages/examples/src/rl_card_lib/games/macao.py#L84) |
| `terminated = True` gdy | któryś gracz pozbył się wszystkich kart | [macao.py:470-472](../packages/examples/src/rl_card_lib/games/macao.py#L470-L472) |
| `truncated = True` gdy | `_turn_count >= max_turns` i nikt nie wygrał | [macao.py:495](../packages/examples/src/rl_card_lib/games/macao.py#L495) |
| nagroda przy truncation | `0.1 × (średni rozmiar ręki przeciwników − rozmiar ręki gracza)` | [macao.py:499-508](../packages/examples/src/rl_card_lib/games/macao.py#L499-L508) |

Uwaga o jednostce: krok środowiska to **jedna akcja jednego gracza**, a nie
jedna „runda”. Deklaracja koloru/figury po asie lub walecie to osobny krok tego
samego gracza ([macao.py:369-392](../packages/examples/src/rl_card_lib/games/macao.py#L369-L392)).

Zmierzone na 200 rozdaniach (`seed` 100000–100199):

| Zestawienie | średnia liczba kroków | mediana | `terminated` | `truncated` (remis) |
|---|---:|---:|---:|---:|
| losowy vs losowy | **195,9** | 200 | 3,0 % | **97,0 %** |
| heurystyka vs heurystyka | **38,7** | 34 | 99,5 % | 0,5 % |
| losowy vs heurystyka | 67,9 | 59,5 | 95,5 % | 4,5 % |

Z zapisanych przebiegów treningowych: DQN 46,5; Double DQN 45,1; PPO 48,7;
Q-learning 64,6 kroku na epizod.

**To jest ważny fakt dla rozdziału 6:** przy słabej polityce Macao degeneruje
się do remisu przez wyczerpanie limitu tur (97 % przy dwóch graczach losowych).
Agent na początku treningu prawie nigdy nie widzi nagrody terminalnej +10.

---

## c) Z kim gra trenowany agent w Macao?

### Konfiguracja odczytana z kodu

```python
# packages/examples/src/rl_card_lib/games/registration.py:80-87
register_sweep_game(
    "macao",
    env_factory=lambda: CardGameEnv(Macao(num_players=2), max_steps=MACAO_MAX_STEPS),
    max_steps=MACAO_MAX_STEPS,
    evaluate=_evaluate_macao,
    self_play=True,
    opponent_factory=lambda seed: MacaoHeuristicAgent(seed=seed),
    ...
)
```

```python
# packages/examples/scripts/run_sweep.py:90-97
if spec.self_play:
    # --self-play forces the zero-lag mirror; otherwise the game's declared
    # opponent (a fixed heuristic) gives an absolute number to read.
    opponent = None if args.self_play else (
        spec.opponent_factory(args.seed) if spec.opponent_factory else None
    )
    trainer = SelfPlayTrainer(env=env, agent=agent, opponent=opponent, **trainer_kwargs)
```

Flaga `--self-play` **nie była podana** w żadnym z zapisanych przebiegów —
potwierdza to zapis konfiguracji w `results/models/macao__*/run.json`:

```json
"trainer": {
  "type": "SelfPlayTrainer",
  "opponent_update_interval": 1000,
  "self_play": false,
  "opponent": "MacaoHeuristicAgent"
}
```

Zmierzone przez skonstruowanie trenera dokładnie tak, jak robi to sweep:

| Pytanie | Odpowiedź |
|---|---|
| liczba graczy | 2 |
| liczba przeciwników w epizodzie | **1** |
| typ przeciwnika w treningu | `MacaoHeuristicAgent` — **ręcznie napisana heurystyka**, nie snapshot, nie agent losowy |
| czy przeciwnik się zmienia w czasie treningu | **nie** — jedna instancja, `seed=args.seed`, stała przez wszystkie 5000 epizodów |
| `opponent_update_interval` | 1000, ale **nieaktywny**: [trainer.py:391-392](../packages/core/src/rl_card_lib/trainer/trainer.py#L391-L392) wychodzi od razu, gdy `self.self_play` jest `False` |
| kiedy byłby snapshot | tylko przy `--self-play`; wtedy `opponent` = zamrożona `deepcopy` agenta, odświeżana co 1000 epizodów |
| tryb przeciwnika | wymuszony `eval()` na czas jego ruchu ([trainer.py:443-446](../packages/core/src/rl_card_lib/trainer/trainer.py#L443-L446)) |
| czy agent uczy się z ruchów przeciwnika | **nie** — tylko z własnych ([trainer.py:455-465](../packages/core/src/rl_card_lib/trainer/trainer.py#L455-L465)) |
| miejsce agenta przy stole | zawsze gracz 0 |

### Ewaluacja: inni przeciwnicy niż w treningu

```python
# packages/examples/src/rl_card_lib/games/registration.py:42-51
def _evaluate_macao(agent, episodes, seed):
    return evaluate_macao_suite(
        agent,
        {
            "random": RandomAgent(action_size=..., seed=seed),
            "heuristic": MacaoHeuristicAgent(seed=seed),
        },
        episodes, MACAO_MAX_STEPS,
    )
```

| | trening | ewaluacja |
|---|---|---|
| przeciwnicy | `MacaoHeuristicAgent` (1) | `RandomAgent` **i** `MacaoHeuristicAgent` (2 osobne serie) |
| tryb agenta | `train()`, ε-zachłanny / próbkowanie polityki | `eval()`, zachłanny |
| liczba epizodów | 5000 | 30 (`--eval-episodes`, domyślnie 30) |

**Odpowiedź:** przeciwnik treningowy i jeden z dwóch przeciwników ewaluacyjnych
to ta sama klasa heurystyki z tym samym seedem. Metryka nagłówkowa Macao
(`win_rate_vs_heuristic`) mierzy zatem grę **przeciwko dokładnie temu
przeciwnikowi, na którym agent był trenowany**. To nie jest wyciek rozdań, ale
jest to wyciek *przeciwnika*: nie mierzy uogólnienia na inny styl gry. Druga
kolumna (`win_rate_vs_random`) mierzy przeciwnika niewidzianego w treningu i
w pracy powinna być raportowana obok.

Dla kontekstu — heurystyka przeciwko samej sobie, zmierzona na 200 rozdaniach
TEST, wygrywa z pozycji gracza 0 w 55,5 % partii (przewaga pierwszego ruchu),
więc **~55 % jest sufitem, do którego warto porównywać**, a nie 100 %.

---

## d) Czy zbiory rozdań treningowych i ewaluacyjnych są rozłączne?

### Krótka odpowiedź: nie ma zbiorów, które można by rozłączyć

Protokół ewaluacyjny biblioteki nazywa się „fixed-deal evaluation protocols”
([evaluation.py:1](../packages/examples/src/rl_card_lib/harness/evaluation.py#L1))
i wygląda tak:

```python
# packages/examples/src/rl_card_lib/harness/evaluation.py:40-45
for seed in range(episodes):
    random.seed(10_000 + seed)
    np.random.seed(10_000 + seed)

    game = KlondikeSolitaire()
    env = CardGameEnv(game, max_steps=max_steps)
```

`KlondikeSolitaire()` bez argumentu tworzy `random.Random(None)`, czyli
generator zasiany z entropii systemu — **globalny `random.seed()` nie ma na to
żadnego wpływu**. Ta sama konstrukcja jest w `evaluate_macao`
([evaluation.py:93-97](../packages/examples/src/rl_card_lib/harness/evaluation.py#L93-L97))
i w `run_klondike_baselines` / `run_macao_baselines`
([baselines.py:72-77](../packages/examples/src/rl_card_lib/harness/baselines.py#L72-L77),
[baselines.py:125-130](../packages/examples/src/rl_card_lib/harness/baselines.py#L125-L130)).

Zmierzone:

| Pomiar | Wynik |
|---|---|
| „rozdanie nr 0” protokołu ewaluacyjnego, wywołane 5 razy — ile różnych | **5 z 5** |
| `evaluate_klondike(RandomAgent(seed=0), 30)` uruchomione dwa razy | `cards_up` = **12,57** i **12,83**; wyniki **nie są identyczne** |

### Co z tego wynika

1. **Nie ma wycieku zbioru testowego w klasycznym sensie.** Rozdania treningowe
   i ewaluacyjne to niezależne losowania z przestrzeni 52! rozdań, więc
   prawdopodobieństwo trafienia w treningu tego samego rozdania co w teście
   jest zerowe w praktyce. Agent nie mógł zapamiętać rozdania testowego.

2. **Ale jest gorszy problem: brak sparowania i nieodtwarzalność.** Deklaracja
   z §6.5 pracy — „evaluated greedily on **fixed deals**, with the baselines of
   section 6.2 measured on **the same deals**” — jest nieprawdziwa. Każdy agent
   i każdy baseline był mierzony na **innym** losowym zestawie 30 rozdań.
   Przy `cards_up` o odchyleniu standardowym rzędu 5–8 kart i n = 30, błąd
   standardowy średniej to ok. 1–1,5 karty; różnice typu „DQN 6,6 vs Double DQN
   5,3” leżą w całości w tym szumie.

3. **Liczby w pracy nie są odtwarzalne.** Ponowne uruchomienie tego samego kodu
   da inne wartości w tabelach 6.2–6.3.

4. Ten sam plik `evaluation.py` przyznaje w nagłówku, że reseeduje globalne
   generatory i że „an evaluation perturbs the training RNG stream”
   ([evaluation.py:8-11](../packages/examples/src/rl_card_lib/harness/evaluation.py#L8-L11)).
   Prawdziwy koszt jest jednak inny, niż tam napisano: reseedowanie nie tylko
   zaburza strumień, ono **nie osiąga swojego celu** — rozdanie i tak nie jest
   ustalone.

**To jest błąd metodologiczny do naprawy.** Naprawiony w
[`results.md`](results.md): zdefiniowana pula TRAIN (seedy 0–9999), pula TEST
(seedy 100000–100199, 200 rozdań, ta sama dla wszystkich agentów i baselinów)
oraz TEST_SOLVABLE (podzbiór TEST potwierdzony solverem).

Warto odnotować, że mechanizm potrzebny do zrobienia tego dobrze **już był
w bibliotece** i był używany w jednym miejscu — benchmark czasu rozwiązania
resetuje przez `env.reset(seed=seed)`
([solve_benchmark.py:65](../packages/examples/src/rl_card_lib/harness/solve_benchmark.py#L65),
[:112](../packages/examples/src/rl_card_lib/harness/solve_benchmark.py#L112)).
Nie był używany przez główną ścieżkę treningu i ewaluacji.

---

## e) Czym są krzywe uczenia z Fig. 6.2 i Fig. 6.3?

W pracy oba rysunki mają jeszcze TODO („embed the trailing-average plot from
`TrainingMetrics.plot()`”). Faktycznie generowane przez repozytorium krzywe
porównawcze powstają w
[`figures.py:751-821`](../packages/report/src/rl_card_lib/report/figures.py#L751-L821)
(`_cmp_curves`) i mają następujące własności:

| Własność | Wartość | Źródło |
|---|---|---|
| **jaka wielkość — Klondike** | liczba kart odłożonych na bazy na koniec epizodu (`cards_up`), 0–52 | `headline_key="cards_up"` w [registration.py:71](../packages/examples/src/rl_card_lib/games/registration.py#L71); seria zbierana per-epizod przez `_klondike_extras` ([registration.py:30-35](../packages/examples/src/rl_card_lib/games/registration.py#L30-L35)) |
| **jaka wielkość — Macao** | udział wygranych **epizodów treningowych** (`win`, 0/1 per epizod) | `headline_key="win_rate_vs_heuristic"` nie jest serią per-epizod, więc `_cmp_curves` spada na `record.series("win")` ([figures.py:758](../packages/report/src/rl_card_lib/report/figures.py#L758)) |
| **uśredniana po ilu epizodach** | **100** | `window = max(5, min(100, longest // 10))`; dla 5000 epizodów `5000 // 10 = 500 → min(100, 500) = 100` ([figures.py:775-776](../packages/report/src/rl_card_lib/report/figures.py#L775-L776) w `_cmp_curves`; identycznie [:349](../packages/report/src/rl_card_lib/report/figures.py#L349) w wykresach per-przebieg) |
| rodzaj średniej | **krocząca wstecz** (trailing), okno rosnące na początku serii | `moving_average(values, window)` |
| **z eksploracją czy bez** | **z eksploracją** — to są epizody treningowe | tytuł osi: `f"{measure} (exploring)"` ([figures.py:788](../packages/report/src/rl_card_lib/report/figures.py#L788)); podpis: „measured during training with exploration on. Not comparable with the greedy-evaluation figures below.” |
| co oznaczają linie przerywane | baseliny (Random, Heuristic, GreedyLookahead, MCTS) — mierzone **zachłannie**, więc na tej samej osi co krzywe eksploracyjne | [figures.py:784-786](../packages/report/src/rl_card_lib/report/figures.py#L784-L786) |
| rozdzielczość zapisu | domyślnie **150 dpi** | `_Emitter.dpi = 150` ([figures.py:162](../packages/report/src/rl_card_lib/report/figures.py#L162)) |
| rozmiar czcionki | `font.size = 9`, etykiety osi 8 pt | [figures.py:95](../packages/report/src/rl_card_lib/report/figures.py#L95), [:116-117](../packages/report/src/rl_card_lib/report/figures.py#L116-L117) |

Istnieje też druga rodzina krzywych, per-przebieg, w
[`_fig_headline_curve`](../packages/report/src/rl_card_lib/report/figures.py#L400-L444) —
ta sama wielkość i to samo okno, ale bez porównania między agentami.

### Trzy rzeczy, które trzeba przy tych rysunkach napisać w podpisie

1. **Wartości z krzywej i wartości z tabeli nie są porównywalne.** Krzywa jest
   z eksploracją (ε ≈ 0,05 po ~600 epizodach, ale PPO próbkuje politykę cały
   czas), tabela jest zachłanna. W pracy to jest już powiedziane w tekście
   §6.5, ale musi być również w podpisie rysunku, bo czytelnik porówna liczby.

2. **Linie baselinów na tych wykresach są zachłanne**, a krzywe nie — czyli
   przecięcie krzywej z linią baseline'u nie oznacza „agent dogonił baseline”.

3. **Rozdzielczość 150 dpi nie spełnia wymogu 300 dpi.** Nowe rysunki w
   [`figures/`](figures/) są renderowane w 300 dpi, z czcionką ≥ 10 pt.

---

## Dodatek: dwie rzeczy znalezione przy okazji

Obie są opisane szczegółowo w [`diagnosis.md`](diagnosis.md), ale należą też
do opisu protokołu:

**Obie zostały naprawione w bibliotece** — opis zostaje, bo tłumaczy, skąd
wzięły się liczby w zapisanych przebiegach sprzed poprawek.

* **Ewaluacja zużywała harmonogram eksploracji.** `Trainer.evaluate()` wołał
  `agent.reset()` na każdy epizod ewaluacyjny, a `reset()` był właśnie miejscem,
  w którym malało ε. Dowód liczbowy: stare `run.json` zapisuje ε *przed*
  treningiem jako 0,8647 dla Klondike i 0,7440 dla Macao, zamiast 1,0 — dokładnie
  `0,995^29` i `0,995^59`, czyli 30 epizodów ewaluacji przed treningiem dla
  Klondike oraz 2 × 30 dla Macao (dwóch przeciwników).
  **Naprawione w PR [#28](https://github.com/mkh63d/rl-card-lib/pull/28)**: zanik
  ε przeniesiono do `agent.on_episode_end()`, wołanego tylko dla epizodów
  treningowych. W nowych przebiegach ε przed treningiem wynosi dokładnie 1,0.

* **Epizod złożony z samych nielegalnych akcji nigdy się nie kończył.**
  `CardGameEnv.step()` wracał przed inkrementacją licznika kroków, więc
  `max_steps` nie działało. Zmierzone wtedy: 5000 nielegalnych akcji,
  `_step_count = 0`. Nie dotyczyło agentów biblioteki (dostają `legal_actions`),
  dotyczyło każdego agenta zewnętrznego.
  **Naprawione w PR [#25](https://github.com/mkh63d/rl-card-lib/pull/25)**: ta
  sama próba kończy się dziś truncation po 200 krokach — patrz
  [`gymnasium.md`](gymnasium.md) §5c.
