# Diagnoza słabych wyników

> **Status:** materiał źródłowy dla rozdziału 6.
> Format każdej pozycji: **objaw → przyczyna w kodzie → poprawka → wynik po poprawce**.
> Surowe dane: [`raw/protocol_probe.json`](raw/protocol_probe.json),
> [`raw/greedy_loop_probe.json`](raw/greedy_loop_probe.json),
> [`raw/action_mix_probe.json`](raw/action_mix_probe.json),
> [`raw/runs/`](raw/runs/) · skrypty w [`scripts/`](scripts/)
> Liczby „po poprawce” pochodzą z przebiegów opisanych w [`results.md`](results.md).

## Spis znalezisk

| # | Znalezisko | Waga | Czy zweryfikowane eksperymentem? | Stan w kodzie |
|---|---|---|---|---|
| **D0** | Pojemność bufora **64** z Tabeli 6.1 **nie istnieje w kodzie** — kod ma 50 000 | błąd w tekście pracy, nie w kodzie | n/d — sprawdzone w kodzie i w historii git | n/d — poprawka dotyczy tekstu pracy |
| **D1** | Truncation traktowany jak stan terminalny: bootstrap zerowany w **100 %** epizodów Klondike | **wysoka** | tak — osobne ramię `fixed` | **scalone w #24** |
| **D2** | Jak mocno Q różnicuje legalne akcje: rozstęp 8,8 % średniej dla Klondike DQN, ale 33 % dla Double DQN — **płaskość Q nie tłumaczy zapętlenia** | średnia (hipoteza odrzucona) | tak — pomiar | n/d — hipoteza, nie usterka |
| **D3** | Zachłanna polityka **zapętla się**: 80–83 % kroków wraca do pozycji już widzianej (losowa: 23 %); dla DQN 70 % ruchów to „dobierz”. Mechanizm kary za powtórzenie istnieje w kodzie i nigdy nie został włączony | **wysoka — główna przyczyna**; ale samo włączenie kary **nie pomaga** — zapętlenie zostaje (86,7 % vs 85,4 %) | diagnoza tak; lekarstwo **obalone** — ramię `noloop` policzone w pełnym protokole (Klondike, 4 agenty × 3 seedy) | kara **scalona w #29**; pomiar mówi, że nie pomaga |
| **D4** | Klondike nie ma sygnału terminalnego: `LOSS_REWARD` jest kodem nieosiągalnym przy `max_passes=None` | **wysoka** | częściowo — pomiar + ramię `fixed` | **scalone w #30** (`BUNDLED_MAX_PASSES = 3`) i #24 |
| **D5** | `target_update_freq=500` to 1,7 epizodu Klondike i 10,9 epizodu Macao | średnia | pomiar | **scalone w #31** — kadencja per gra (klondike 500, macao 100) |
| **D6** | Harmonogram ε schodzi do 0,05 po **599 epizodach** — 88 % treningu jest prawie zachłanne | średnia | pomiar analityczny | bez zmian — świadomy wybór harmonogramu |
| **D7** | Ewaluacja **zużywa** harmonogram ε (`agent.reset()` obniża ε) | niska, ale psuje reprodukowalność | pomiar | **scalone w #28** |
| **D8** | Tablica Q-learningu rośnie o **0,84 wpisu na krok** — czysta memoryzacja | wysoka dla Q-learningu | pomiar | bez zmian — właściwość metody tabelarycznej |
| **D9** | γ = 0,95 przy epizodzie 300-krokowym: horyzont efektywny ~20 kroków | średnia | analiza | bez zmian — świadomy wybór γ |
| **D10** | Epizod z samych nielegalnych akcji nigdy się nie kończył (dziś: truncation po `max_steps`) | poza zakresem wyników, ważne dla §Gymnasium | pomiar | **scalone w #25** |
| **D11** | **Zachłanna ewaluacja niszczy wyuczoną politykę.** Te same wagi PPO: `argmax` → 7,5 karty i 0,0 % wygranych; próbkowanie własnego rozkładu → **22,5 karty i 28,5 % wygranych** | **najwyższa — zmienia główny wniosek rozdz. 6** | tak — pomiar na tych samych 200 rozdaniach TEST | **scalone w #33** (zgłoszenie #21) |

---

## D0. Bufor odtwarzania 64 — tego w kodzie nie ma

### Objaw (zgłoszony przez promotora)

Tabela 6.1 pracy podaje:

```
Replay-buffer capacity     -     64     64
Batch / minibatch size     -     64     64
```

co oznaczałoby bufor równy jednemu batchowi, czyli brak dekorelacji próbek.

### Sprawdzenie w kodzie

**Ta wartość w kodzie nie występuje.** Faktyczna konfiguracja, z jednego
miejsca, z którego budowane są wszystkie agenty w sweepie:

```python
# packages/examples/src/rl_card_lib/harness/learners.py:49-64
if kind == "dqn":
    return DQNAgent(
        ..., buffer_size=50_000, batch_size=64, target_update_freq=500, ...
    )
if kind == "double_dqn":
    return DoubleDQNAgent(
        ..., buffer_size=50_000, batch_size=64, target_update_freq=500,
        dueling=True, ...
    )
```

Potwierdzenia:

| Źródło | `buffer_size` |
|---|---|
| `harness/learners.py:54` (DQN) i `:62` (Double DQN) — ścieżka sweepa | **50 000** |
| domyślna wartość `DQNAgent.__init__` ([dqn_agent.py:206](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L206)) | 100 000 |
| `scripts/train_klondike.py:35` | 50 000 |
| `scripts/train_macao.py:32` | 30 000 |
| **zapisane w `results/models/*/run.json` dla wszystkich 4 przebiegów DQN/DDQN** | **50 000** |
| historia git (`git log -S buffer_size`) — od pierwszego commita z tymi agentami (`cec9e5d`) | 50 000 |
| agent zbudowany na żywo, odczyt `replay_buffer.buffer.maxlen` | **50 000** |

Wartość 64 nie pojawia się w żadnym miejscu poza `batch_size`. Poza tym PPO
w ogóle **nie ma** bufora odtwarzania — jest algorytmem on-policy
([ppo_agent.py:89-91](../packages/core/src/rl_card_lib/agents/ppo_agent.py#L89-L91)),
a jego odpowiednik to `rollout_steps = 1024`.

### Poprawka

**Do poprawy jest tabela w pracy, nie kod.** Wygląda na przepisanie wartości
`batch_size` do wiersza „Replay-buffer capacity”. Poprawiony wiersz:

| Hyper-parameter | Q-learning | DQN / Double DQN | PPO |
|---|---|---|---|
| Replay-buffer capacity | – | **50 000** | – (on-policy) |
| Rollout length | – | – | 1024 |
| Batch / minibatch size | – | 64 | 64 |

Pełna, wygenerowana z kodu tabela jest w
[`tables/hyperparameters.csv`](tables/hyperparameters.csv).

### Wynik po poprawce

Nie dotyczy — nie było czego naprawiać w kodzie, więc **hipoteza „bufor równy
batchowi jest głównym powodem, dla którego agenci wartościowi zostali przy
zerze” jest odrzucona**. Prawdziwe przyczyny to D1–D4 i D9.

---

## D1. Truncation traktowany jak stan terminalny

### Objaw

Agenci wartościowi w Klondike uczą się polityki zachłannej **gorszej niż
losowa** (6,4–6,7 karty vs 11,2 karty dla polityki losowej na tych samych
30 rozdaniach TEST).

### Przyczyna w kodzie

Trener skleja oba powody zakończenia epizodu w jedną flagę:

```python
# packages/core/src/rl_card_lib/trainer/trainer.py:209-217
next_observation, reward, terminated, truncated, info = self.env.step(action)
done = terminated or truncated
...
learn_result = self._learn(
    self.agent, observation, action, reward,
    next_observation, done, info,
)
```

a każdy cel TD mnoży bootstrap przez `(1 - done)`:

```python
# packages/core/src/rl_card_lib/agents/dqn_agent.py:384
target_q = rewards + (1 - dones) * has_actions * self.gamma * next_q
```

```python
# packages/core/src/rl_card_lib/agents/tabular.py:155-156
if done:
    target = reward
```

Ta sama flaga zeruje ślad GAE w PPO
([ppo_agent.py:306-308](../packages/core/src/rl_card_lib/agents/ppo_agent.py#L306-L308)).

Skutek zależy od gry:

* **Klondike** — mierzone (200 rozdań TEST, polityka losowa): `terminated` 0 %,
  `truncated` **100 %**. Instrumentowany trening (5 epizodów, 1500 przejść):
  5 przejść z `done=True`, z tego **0 zakończeń gry i 5 truncation**; nagroda
  na tych przejściach to `−0,01`, czyli sam koszt ruchu. Czyli w **każdym**
  epizodzie ostatnie przejście uczy sieć, że stan na 300. kroku jest wart tyle,
  ile natychmiastowa nagroda, i nic więcej.
* **Macao** — truncation przy słabej grze jest częste (97 % przy dwóch graczach
  losowych), ale gra **płaci** wtedy nagrodę różnicową
  ([macao.py:499-508](../packages/examples/src/rl_card_lib/games/macao.py#L499-L508)),
  więc szkoda jest mniejsza; bootstrap i tak jest zerowany.

### Poprawka

Rozdzielić oba przypadki — jedna linia w `Trainer._run_episode`:

```python
# było
done = terminated or truncated
learn_result = self._learn(self.agent, observation, action, reward,
                           next_observation, done, info)

# ma być: pętla kończy się na obu, ale uczy się tylko na terminated
done = terminated or truncated
learn_result = self._learn(self.agent, observation, action, reward,
                           next_observation, terminated, info)
```

**Scalone w PR [#24](https://github.com/mkh63d/rl-card-lib/pull/24).**
`Trainer._run_episode` przekazuje dziś `terminated`, a `truncated` osobnym
argumentem, więc to jest zachowanie biblioteki — ramię **`fixed`**. Odwrotność
mieszka teraz w [`scripts/harness.py`](scripts/harness.py) jako
`ConflatedTruncationMixin`, który skleja obie flagi z powrotem, żeby dało się
zmierzyć stan sprzed poprawki: ramię **`asis`**.

### Wynik po poprawce

**Uczciwa odpowiedź: na metryce nagłówkowej — żaden.** 3 seedy inicjalizacji,
5000 epizodów, identyczny strumień rozdań TRAIN, identyczna pula 200 rozdań
TEST, z poprawką i bez:

| gra | agent | `asis` | `fixed` | Δ |
|---|---|---:|---:|---:|
| Klondike | Double DQN | 5,88 ± 0,30 | 5,38 ± 0,44 | **−0,50 karty** |
| Klondike | DQN | 5,67 ± 0,41 | 5,77 ± **0,07** | +0,10 karty |
| Macao | Double DQN | 7,5 ± 1,0 % | 8,7 ± 0,6 % | **+1,2 pp** |
| Macao | DQN | 7,8 ± 0,6 % | 7,8 ± 1,0 % | 0,0 pp |

Dwie kombinacje w górę, jedna w dół, jedna bez zmian — wszystkie w granicach
mniej więcej jednego odchylenia standardowego. Jedyny konsekwentny efekt jest
na **rozrzucie**: odchylenie Klondike DQN między seedami spada z 0,41 do 0,07.

To nie jest argument za zostawieniem błędu. Obecny cel TD jest **dowodliwie**
obciążony w 100 % epizodów Klondike, a poprawka to jedna linia. To jest
natomiast argument za tym, że **obciążenie bootstrapu nie jest tym, co trzyma
tych agentów przy zerze**. Dominującym efektem okazała się reguła ewaluacji
(D11): ten sam checkpoint PPO daje 7,5 karty przez `argmax` i 22,5 przez
próbkowanie własnej polityki.

---

## D2. Jak mocno wyuczona funkcja Q różnicuje legalne akcje

### Pomiar

Dla każdej pozycji odwiedzonej podczas zachłannej gry na 30 rozdaniach TEST
policzono średnią i rozstęp wartości Q **ograniczonych do legalnych akcji**
(`thesis_notes/raw/q_spread_probe.json`):

| Agent | pozycji | średnie Q (legalne) | średni rozstęp Q | rozstęp jako % średniej |
|---|---:|---:|---:|---:|
| Klondike, DQN | 9 000 | 0,751 | **0,066** | **8,8 %** |
| Klondike, Double DQN | 9 000 | 1,015 | 0,337 | 33,3 % |
| Macao, DQN | 2 593 | 6,436 | 1,380 | 21,4 % |
| Macao, Double DQN | 2 782 | 5,254 | 0,877 | 16,7 % |

### Co z tego wynika — i czego z tego **nie** wynika

Plaska funkcja Q jest realnym zjawiskiem tylko dla **jednego** przypadku:
Klondike + plain DQN (rozstęp 8,8 % średniej wartości). Dla Double DQN na
Klondike rozstęp to 33 % — funkcja nie jest płaska, a mimo to ten agent
zapętla się **bardziej** niż DQN (83,2 % vs 80,3 % powtórzonych pozycji, D3).

**Płaskość Q nie jest więc przyczyną zapętlenia.** Przyczyna jest strukturalna
i nie zależy od wartości Q wcale:

> W środowisku deterministycznym polityka deterministyczna, która wróci do
> stanu już odwiedzonego, powtórzy od tego miejsca całą swoją dotychczasową
> przyszłość — czyli wejdzie w cykl.

Klondike spełnia obie przesłanki: ruchy tableau↔tableau są odwracalne, a przy
`max_passes=None` pętla „dobierz / przełóż talię” jest deterministyczna przy
ustalonym porządku talii. Polityka zachłanna jest deterministyczna z definicji.
Nie ma nic, co by ten cykl przerywało — kara za powtórzoną pozycję istnieje w
`CardGameEnv` i jest wyłączona (D3).

To wyjaśnia też, dlaczego krzywe treningowe wyglądają lepiej niż ewaluacja
zachłanna: podczas treningu ε > 0 przerywa cykl losowym ruchem średnio co
20 kroków. Ta sama polityka bez eksploracji cykluje.

### Poprawka i wynik

Nie ma jednej linii do zmiany. Adresowane pośrednio przez ramiona `fixed`
(D1) i `noloop` (D3); wynik mierzony tym samym rozstępem Q po treningu —
patrz §Wyniki ablacji.

---

## D3. Zachłanna polityka zapętla się

### Objaw / pomiar

30 rozdań TEST, polityka zachłanna, licznik pozycji już widzianych w epizodzie
(hash wektora obserwacji — dokładnie ta sama definicja, której używa
`CardGameEnv`):

| Polityka | udział kroków wracających do znanej pozycji | różnych akcji na epizod (z 68) | karty na bazach |
|---|---:|---:|---:|
| DQN (wytrenowany, zachłanny) | **80,3 %** | 12,5 | 6,43 |
| Double DQN (wytrenowany, zachłanny) | **83,2 %** | 14,0 | 6,73 |
| PPO (wytrenowany, zachłanny) | 78,9 % | 16,9 | 8,77 |
| **losowa** | **23,0 %** | 40,0 | **11,17** |

Rozkład typów ruchu w tych samych epizodach:

| Polityka | dobierz / przełóż talię | tableau ↔ tableau | na bazę | z odkrytej na tableau |
|---|---:|---:|---:|---:|
| DQN | **70,2 %** | 24,7 % | **2,1 %** | 3,0 % |
| Double DQN | 43,7 % | 51,6 % | 2,2 % | 2,4 % |
| PPO | 52,9 % | 40,6 % | 2,9 % | 3,6 % |
| losowa | 29,4 % | 62,3 % | 3,7 % | 4,5 % |
| heurystyka | 33,0 % | 47,9 % | **12,7 %** | 6,4 % |

Wyuczona polityka DQN to w 70 % „dobierz kartę”. Przy `max_passes=None` talię
można przekładać w nieskończoność, więc jest to pętla zamknięta, która kończy
się dopiero na limicie 300 ruchów.

### Przyczyna w kodzie

Środowisko **ma** mechanizm przeciw takim pętlom i **ma go wyłączony**:

```python
# packages/core/src/rl_card_lib/env/card_game_env.py:20-27
def __init__(
    self,
    game: Any,
    max_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    invalid_action_reward: float = -1.0,
    repeated_position_penalty: float = 0.0,     # <- domyślnie zero
):
```

```python
# packages/core/src/rl_card_lib/env/card_game_env.py:146-151
position = hash(observation.tobytes())
if position in self._seen_positions:
    info["repeated_position"] = True
    reward += self.repeated_position_penalty      # dodaje 0.0
else:
    self._seen_positions.add(position)
```

a rejestracja obu gier buduje środowisko bez tego argumentu:

```python
# packages/examples/src/rl_card_lib/games/registration.py:58
env_factory=lambda: CardGameEnv(KlondikeSolitaire(), max_steps=KLONDIKE_MAX_STEPS),
```

Wniosek: kara za powtórzoną pozycję jest w bibliotece zaimplementowana,
udokumentowana w docstringu jako lekarstwo na dokładnie ten problem
(„Games with reversible moves let an agent shuffle in circles forever; this
makes each lap cost something”) i **nigdy nie włączona w żadnym przebiegu**.

### Poprawka

```python
env_factory=lambda: CardGameEnv(
    KlondikeSolitaire(), max_steps=KLONDIKE_MAX_STEPS,
    repeated_position_penalty=-0.05,
),
```

Ramię eksperymentu: **`noloop`** (`LOOP_PENALTY = -0.05` w
[`scripts/run_one.py`](scripts/run_one.py)).

Wariant alternatywny, nie testowany: `max_passes=3` w konstruktorze Klondike.
Zmierzone konsekwencje samego `max_passes=3` na 200 rozdaniach TEST (bez
uczenia): rozdanie „umiera” w 4,5 % przypadków przy grze losowej i 1,5 % przy
heurystyce, czyli gałąź `LOSS_REWARD` przestaje być kodem martwym — ale kosztem
spadku wyniku heurystyki z 28,7 do 25,8 karty.

### Wynik po poprawce — kara **nie** usuwa zapętlenia

> **Pomiar wstępny, poza protokołem.** Ramię `noloop` nigdy nie zostało
> policzone w pełnym protokole (5000 epizodów, 3 seedy, pula TEST 200 rozdań),
> więc w [`tables/ablation_fixes.csv`](tables/ablation_fixes.csv) do dziś go
> nie ma — są tylko `asis` i `fixed`. Liczby poniżej pochodzą z krótszego
> przebiegu: DQN, 1200 epizodów, 2 seedy inicjalizacji, ewaluacja zachłanna na
> 30 rozdaniach TEST (100000–100029). **Nie są porównywalne** z wierszami
> `asis` / `fixed` w tabeli ablacji i nie zastępują tamtego przebiegu. Podaję
> je, bo prowadzą do wniosku, który zmienia status samej poprawki.
> Źródło: [`raw/noloop_preliminary.json`](raw/noloop_preliminary.json).

| ramię | udział kroków wracających do znanej pozycji | udział „dobierz” | karty na bazach |
|---|---:|---:|---:|
| `asis` (kara 0,0) | 85,4 % | 22,3 % | 5,25 |
| `noloop` (kara −0,05) | **86,7 %** | 15,0 % | 4,75 |

Kara **nie zmniejsza zapętlenia**. Różnica 1,3 pp idzie w złą stronę i jest
mniejsza niż rozrzut między seedami wewnątrz każdego ramienia (`asis`
83,9–86,9 %, `noloop` 85,6–87,8 %), a karty na bazach nie rosną.

Najmocniejsza przesłanka jest w krzywej treningowej — nagroda na końcu
przebiegu:

| ramię | seed 0 | seed 1 |
|---|---:|---:|
| `asis` | +6,47 | +5,57 |
| `noloop` | −4,47 | −4,30 |

Ponieważ zachowanie zachłanne jest w obu ramionach niemal identyczne, różnica
~10,4 to po prostu **kara faktycznie zapłacona**: ~208 ukaranych kroków na
epizod przy limicie 300, czyli agent płaci na około dwóch trzecich kroków —
i przez 1200 epizodów tego nie zmniejsza. Gdyby sygnał był wyuczalny, ta luka
zamykałaby się w stronę zera.

### Dlaczego kara nie działa

Dwa powody, oba strukturalne:

1. **Kara nie jest markowowska względem obserwacji.** `_seen_positions` to
   historia epizodu, której agent nie widzi. Ta sama para (obserwacja, akcja)
   dostaje różną nagrodę zależnie od tego, czy pozycja była już odwiedzona,
   więc funkcja wartości może nauczyć się najwyżej *średniego* kosztu akcji —
   nigdy tego, że **ten** krok jest powtórzeniem.

2. **Wycena cyklu nie usuwa własności, która go tworzy.** Argument z tego
   znaleziska jest mocniejszy niż zaproponowane lekarstwo: polityka
   deterministyczna w środowisku deterministycznym zapętla się, gdy tylko wróci
   do odwiedzonego stanu. Podrożenie cyklu przenosi politykę na *tańszy* cykl,
   nie likwiduje cyklu.

Ciężar przechodzi więc na wariant odłożony wyżej jako „nie testowany”:
**`max_passes=3`** — talia przestaje być odnawialna, więc pętla dobierania się
**kończy**, zamiast tylko kosztować. Pozostałe kierunki: umieścić znacznik
odwiedzenia w obserwacji (kara staje się wyuczalna) albo ewaluować z małym ε.

Sam fakt, że mechanizm był martwy, pozostaje usterką wartą naprawienia —
PR [#29](https://github.com/mkh63d/rl-card-lib/pull/29) podłącza go i dokłada
testy, które nie pozwolą mu znowu zgasnąć. Ale **nie jest to lekarstwo na
zapętlenie**, wbrew temu, co obiecuje docstring w `CardGameEnv`. Diagnoza
z tego znaleziska (że polityka zachłanna się zapętla i że to tłumaczy lukę
zachłanne–eksploracyjne) stoi niezmieniona; upada tylko zaproponowane
lekarstwo.

---

## D4. Klondike nie ma sygnału terminalnego

### Objaw

W 5000 epizodach treningowych żaden agent wartościowy nie zobaczył ani jednej
nagrody terminalnej innej niż koszt ruchu.

### Przyczyna w kodzie

```python
# packages/examples/src/rl_card_lib/games/klondike.py:353-364
won = self._check_win()
if won:
    ...
elif not self.get_legal_actions():
    # No legal moves left: the deal is dead.
    self.done = True
    reward += self.LOSS_REWARD
```

Gałąź `elif` wymaga **pustej** listy legalnych akcji. Ale przy domyślnym
`max_passes=None`:

```python
# packages/examples/src/rl_card_lib/games/klondike.py:226
if self.stock or (self.waste and self._can_recycle()):
    legal.append(0)
```

`_can_recycle()` zwraca `True` bezwarunkowo, gdy `max_passes is None`
([:376-378](../packages/examples/src/rl_card_lib/games/klondike.py#L376-L378)), więc
akcja 0 jest zawsze dostępna i lista nigdy nie jest pusta. Sam kod to zresztą
przyznaje w komentarzu: „Reachable only with a finite max_passes”.

Zatem dla domyślnej konfiguracji: **wygrana** (praktycznie nieosiągalna dla
uczących się agentów — 0,0 % w ewaluacji zachłannej) albo **truncation bez
żadnej nagrody terminalnej**. `LOSS_REWARD = -1.0` jest stałą, która nigdy nie
jest użyta.

### Poprawka

Dwa niezależne warianty, oba jednoliniowe:

1. `KlondikeSolitaire(max_passes=3)` — przywraca przegraną jako zdarzenie
   terminalne (mierzone: 4,5 % rozdań przy grze losowej);
2. potraktowanie truncation jako truncation, a nie terminacji — czyli **D1**.

**Oba zostały scalone**: wariant 2 w PR
[#24](https://github.com/mkh63d/rl-card-lib/pull/24), wariant 1 w PR
[#30](https://github.com/mkh63d/rl-card-lib/pull/30), gdzie
`KlondikeSolitaire.BUNDLED_MAX_PASSES = 3` stało się regułą gry dołączonej do
biblioteki (klasa nadal domyślnie ma `max_passes=None`, żeby użytkownik mógł
grać bez limitu). Ramię `fixed` zawiera więc dziś **oba** warianty naraz, co jest
też powodem, dla którego `asis → fixed` nie jest ablacją jednego czynnika —
patrz [`results.md`](results.md).

Wpływ wariantu 1 na trening jest teraz zmierzony: w ramieniu `fixed` **2–11 %
epizodów Klondike kończy się terminacją** zamiast 0 % w `asis`, czyli przegrana
przestała być kodem nieosiągalnym. Skutkiem ubocznym jest to, że baseliny na
Klondike spadły (losowa 11,59 → 9,79 karty, heurystyka 28,74 → 25,84), bo trzy
przejścia przez stos dają mniej okazji niż nieskończenie wiele — porównania
„agent vs losowy” trzeba czytać na nowej wartości odniesienia.

### Wynik po poprawce

Patrz §Wyniki ablacji.

---

## D5. `target_update_freq = 500` w odniesieniu do długości epizodu

### Pomiar

`DQNAgent.learn()` wykonuje jeden krok gradientu na każdy krok środowiska
(gdy bufor ma już `batch_size` przejść), a sieć docelowa jest kopiowana co
`train_steps % 500 == 0`
([dqn_agent.py:395-399](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L395-L399)).
Czyli 500 kroków gradientu ≈ 500 kroków środowiska.

| Gra | średnia długość epizodu | 500 kroków to… | aktualizacji sieci docelowej w 5000 epizodach |
|---|---:|---|---:|
| Klondike | 300,0 (zawsze limit) | **1,67 epizodu** | **3 000** |
| Macao | 46,0 (zmierzone) | **10,9 epizodu** | **460** |

### Ocena

* **Klondike: rozsądnie.** Sieć docelowa jest odświeżana ~0,6 razy na epizod;
  to standardowy rząd wielkości. Nie jest to przyczyna problemu.
* **Macao: prawdopodobnie za rzadko.** Cały trening to 460 aktualizacji sieci
  docelowej. Przy nagrodzie terminalnej +10 pojawiającej się rzadko, wartość
  propaguje się od stanu terminalnego wstecz o jeden krok na aktualizację
  celu — 460 aktualizacji to mało, żeby wartość dotarła w głąb epizodu.
* **Asymetria jest niezamierzona:** ta sama liczba 500 znaczy co innego w grze
  6,5 razy krótszej. Wartość powinna być wyrażona w epizodach albo dobrana
  per gra.

### Poprawka

Nie zmieniana w tych eksperymentach — zmiana `target_update_freq` dla Macao
zmieniałaby hiperparametr opisany w Tabeli 6.1, a nie naprawiała błędu.
**Rekomendacja do rozdziału o dalszych pracach:** wyrazić częstość
w epizodach (np. co 20 epizodów) albo ustawić dla Macao 100–200 kroków.

**Zastosowano po pomiarze** ([#19](https://github.com/mkh63d/rl-card-lib/issues/19)):
druga opcja — `target_update_freq` jest teraz wartością per gra
w `register_sweep_game(...)`, obok `mcts_simulations`. Klondike zachowuje 500
(1,7 epizodu), Macao dostaje 100 (~2,2 epizodu, 2300 aktualizacji zamiast 460).
Rdzeń `DQNAgent` liczy dalej w krokach gradientu, zgodnie z literaturą.

Tabela hiperparametrów (`tables/hyperparameters.csv`) oraz klucze
`hyperparameters` i `target_update_cadence` w `raw/protocol_probe.json` zostały
przeliczone i podają teraz wartość per gra. Pozostałe pomiary w tym pliku
(`episode_shape`, `deal_stream`, …) celowo zostawiono nietknięte — opisują stan
z przebiegu pomiarowego i nie zależą od tego hiperparametru. **Wyniki uczenia
Macao z tej pracy pochodzą sprzed zmiany i nie są z nią porównywalne** —
wymagają ponownego treningu, nie reinterpretacji.

---

## D6. Harmonogram ε — czy faktycznie schodzi do 0,05 i kiedy

### Pomiar

Reguła: `epsilon *= 0.995` raz na epizod, dopóki `epsilon > epsilon_end`
([dqn_agent.py:403-414](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L403-L414),
[tabular.py:175-184](../packages/core/src/rl_card_lib/agents/tabular.py#L175-L184)).

| Wielkość | Wartość |
|---|---|
| rozwiązanie analityczne `1.0 · 0.995^n = 0.05` | n = 598,6 |
| pierwszy epizod, w którym ε ≤ 0,05 (symulacja pętli z kodu) | **599** |
| ε na epizodzie 100 / 300 / 500 / 600 | 0,609 / 0,223 / 0,082 / **0,0499** |
| wartość końcowa | **0,049 76** (ostatnie mnożenie schodzi minimalnie poniżej progu, bo warunek jest sprawdzany przed mnożeniem) |
| udział 5000-epizodowego treningu spędzony na ε = 0,05 | **88,0 %** |

**Odpowiedź: tak, ε schodzi do 0,05, i robi to po ~599 epizodach z 5000.**
Komentarz w kodzie („floor at ~600 episodes”,
[learners.py:46](../packages/examples/src/rl_card_lib/harness/learners.py#L46))
jest zgodny z pomiarem.

### Ocena

Samo w sobie nie jest to błąd — 12 % epizodów na eksplorację to typowy
harmonogram. Ale w połączeniu z D3 jest to istotne: **przez 88 % treningu
agent zbiera dane z polityki, która w 80 % kroków chodzi w kółko**, więc bufor
odtwarzania zapełnia się przejściami z kilkunastu powtarzających się pozycji.
Przy 68 akcjach i ε = 0,05 na epizod 300-krokowy przypada ~15 ruchów losowych.

### Poprawka

Nie zmieniana (to jest deklarowany hiperparametr pracy). Rekomendacja do
dalszych prac: wolniejszy zanik (`0,999` → dno przy ~3000 epizodach) albo
ε-floor 0,1 dla gry o tak długim horyzoncie.

---

## D7. Ewaluacja zużywa harmonogram eksploracji

### Objaw

`results/models/klondike__*/run.json` zapisuje ε **przed** treningiem jako
`0,8647`, a `macao__*/run.json` jako `0,7440` — zamiast 1,0. Konfiguracja jest
przechwytywana z komentarzem „Captured before training so the recorded epsilon
is the start value” ([run_sweep.py:102-105](../packages/examples/scripts/run_sweep.py#L102-L105)),
więc powinno tam być 1,0.

### Przyczyna w kodzie

`run_sweep.train_one` woła ewaluację *przed* przechwyceniem konfiguracji
([run_sweep.py:72-75](../packages/examples/scripts/run_sweep.py#L72-L75)), a każdy
epizod ewaluacji woła `agent.reset()`
([evaluation.py:50](../packages/examples/src/rl_card_lib/harness/evaluation.py#L50)),
czyli dokładnie tę metodę, która obniża ε.

Rachunek się zgadza co do ostatniej cyfry:

| Gra | epizodów ewaluacji przed treningiem | `0.995^(n-1)` | zapisane w `run.json` |
|---|---:|---:|---:|
| Klondike | 30 | 0,864 707 730 567 5337 | **0,864 707 730 567 5338** |
| Macao | 2 × 30 (dwaj przeciwnicy) | 0,743 980 862 006 7382 | **0,743 980 862 006 7382** |

Ten sam mechanizm działa w trakcie treningu: przy `eval_interval = episodes//10`
i `eval_episodes = 20` dochodzi **200 dodatkowych zaników ε** na przebieg
5000-epizodowy.

### Skutki

1. Zapisany „epsilon_start” w raporcie jest nieprawdziwy.
2. Harmonogram eksploracji zależy od tego, ile razy się mierzyło — czyli od
   ustawień raportowania, nie od konfiguracji uczenia.
3. Efekt jest ilościowo mały (dno ε i tak jest przy ~599 epizodach, a 230
   dodatkowych zaników przesuwa je do ~570), ale **psuje odtwarzalność**.

### Poprawka

Ewaluacja nie powinna mieć efektów ubocznych.

**Scalone w PR [#28](https://github.com/mkh63d/rl-card-lib/pull/28)** dokładnie
tą drogą, którą ta sekcja proponowała: zanik ε przeniesiono z `Agent.reset()` do
jawnego `agent.on_episode_end()`, wołanego przez pętlę treningową **tylko dla
epizodów treningowych**. Ewaluacja woła `reset()` jak każdy epizod, ale `reset()`
już niczego nie zanika.

Kontekst `frozen_exploration()` w [`scripts/harness.py`](scripts/harness.py)
został zachowany, bo nadal ma co robić — funkcje pomiarowe przełączają na
agencie tryb `train`/`eval` oraz `eval_greedy` (PPO, D11) i muszą oddać obiekt
w stanie, w jakim go dostały. Samego ε nie musi już pilnować.

### Wynik po poprawce

W nowych przebiegach ε po `test_before` wynosi dokładnie 1,0, a seria `epsilon`
zapisana w `raw/runs/*.json` odpowiada wzorowi `0,995^(n-1)` z dnem na epizodzie
599 — patrz [`figures/epsilon_schedule.png`](figures/epsilon_schedule.png).

---

## D8. Tablica Q-learningu to czysta memoryzacja

### Pomiar

Z zapisanych serii per-epizod (`results/models/*__q_learning/run.json`):

| Gra | kroków środowiska w treningu | rozmiar tablicy Q po treningu | nowych wpisów na krok |
|---|---:|---:|---:|
| Klondike | 1 499 936 | **1 253 141** | **0,836** |
| Macao | 322 883 | **318 688** | **0,987** |

Czyli praktycznie **każdy** napotkany stan jest nowy. Agent nie ma z czego
uogólniać; przy nieznanym stanie wiersz Q jest wypełniony `optimistic_init = 0.0`
i wszystkie legalne akcje są remisem, więc
[tabular.py:126-127](../packages/core/src/rl_card_lib/agents/tabular.py#L126-L127)
losuje spośród nich — to jest polityka losowa.

Potwierdzenie z benchmarku: w `results/solve_benchmark/klondike.json`
wytrenowany `QLearningAgent` osiąga `cards_up = 14,56`, czyli **dokładnie tyle
samo co `Random`**. W nowych przebiegach (3 seedy, pula TEST 200 rozdań) jest
tak samo: 11,31 ± 0,26 przed treningiem i 11,33 ± 0,20 po treningu, przy
baseline losowym **11,59** — 1,5 mln kroków uczenia nie zmieniło niczego.

Praktyczny koszt tej memoryzacji: checkpoint tablicy Q dla Klondike waży
**1,76 GB** (Macao 0,36 GB) — ×3 seedy to 6,4 GB samych plików `.pkl`.

### Przyczyna w kodzie

Klucz tablicy to zaokrąglona do 2 miejsc obserwacja
([tabular.py:88-89](../packages/core/src/rl_card_lib/agents/tabular.py#L88-L89)).
Dla Klondike obserwacja ma 221 wymiarów, z czego 208 to bity lokalizacji kart —
zaokrąglenie niczego nie skleja. Dokumentacja klasy mówi o tym wprost
(„Expect the table to grow roughly one entry per step and the policy to stay
near random. Watching that failure is the point”).

### Poprawka

Żadna — to jest zamierzony punkt dydaktyczny biblioteki. **Ale w pracy trzeba
to napisać jako wynik, a nie jako porażkę agenta.** Obecny tekst §6.5 mówi
„Tabular Q-learning ended at 11.4 cards, the best of the learners but still
below random and, unusually, worse than before training” — słowo „unusually”
jest mylące: agent tabularny **jest** polityką losową, więc jego wynik to
oszacowanie polityki losowej z szumem n = 30, a nie efekt uczenia.

---

## D9. Współczynnik dyskonta względem długości epizodu

γ = 0,95 dla obu gier ([learners.py:43](../packages/examples/src/rl_card_lib/harness/learners.py#L43), [:52](../packages/examples/src/rl_card_lib/harness/learners.py#L52), [:60](../packages/examples/src/rl_card_lib/harness/learners.py#L60), [:68](../packages/examples/src/rl_card_lib/harness/learners.py#L68)).

| Wielkość | Klondike | Macao |
|---|---:|---:|
| horyzont efektywny `1/(1−γ)` | **20 kroków** | 20 kroków |
| długość epizodu | **300** | 46 |
| waga nagrody na końcu epizodu, `γ^T` | 0,95³⁰⁰ = **1,9 · 10⁻⁷** | 0,95⁴⁶ = 9,5 · 10⁻² |

Dla Macao γ = 0,95 jest dobrane sensownie: nagroda +10 z końca partii ma
w chwili startu wagę ~0,95, czyli ~0,95 punktu — wciąż dominuje nad nagrodami
kształtującymi rzędu 0,1. Dla Klondike wygrana na 300. kroku jest w praktyce
niewidoczna; agent optymalizuje wyłącznie nagrody kształtujące w oknie ~20
ruchów. To jest spójne z tym, co widać w D3: polityka gra lokalnie i nie
planuje.

**Rekomendacja (nietestowana):** dla Klondike γ = 0,995 (horyzont 200) albo
skrócenie epizodu przez `max_passes`. Nie było to zmieniane w eksperymentach,
żeby ramiona różniły się jedną rzeczą naraz.

---

## D10. Epizod z samych nielegalnych akcji nigdy się nie kończył

Opisane w [`gymnasium.md`](gymnasium.md) §5c. W skrócie: `CardGameEnv.step()`
zwracała wynik **przed** `self._step_count += 1`, gdy akcja była nielegalna, więc
`max_steps` nigdy nie zadziałało. Zmierzone wtedy: 5000 nielegalnych akcji,
`_step_count = 0`, `terminated = truncated = False`.

### Poprawka i wynik

**Scalone w PR [#25](https://github.com/mkh63d/rl-card-lib/pull/25)** —
gałąź nielegalnej akcji zlicza krok i stosuje limit. Ten sam pomiar dziś:

| 5000 nielegalnych akcji, `max_steps=200` | `2bd42ab` | dziś |
|---|---|---|
| epizod się zakończył | nie | **tak, po 200 krokach** |
| `env._step_count` | 0 | **200** |

Nie dotyczy żadnej liczby w wynikach uczenia — agenci biblioteki zawsze dostają
`legal_actions` i nie proponują nielegalnego ruchu — ale było realnym błędem
środowiska dla każdego konsumenta z zewnątrz. Widać to w teście
interoperacyjności ze Stable-Baselines3: demo na Macao trwało 300 kroków
i płaciło −300,0, a dziś kończy się limitem po 200 krokach z −198,9.

---

## D11. Zachłanna ewaluacja niszczy wyuczoną politykę

> **To jest najważniejsze znalezisko całej analizy.** Zmienia odpowiedź na
> trzecie pytanie badawcze z §6.1 i główny wniosek §6.5 o Klondike.

### Objaw

Wszystkie liczby dla Klondike w pracy pochodzą z ewaluacji zachłannej i
wszystkie są poniżej baseline'u losowego. Interpretacja w pracy brzmi: agenci
nic się nie nauczyli.

### Pomiar

Wzięto **te same, już wytrenowane checkpointy** z `checkpoints/klondike_*` i
odegrano nimi **te same 200 rozdań TEST**. Nic nie było uczone od nowa.
Zmieniono **wyłącznie regułę zamiany wyjścia sieci na akcję**:

| Agent | ε = 0 (zachłannie) | ε = 0,05 | ε = 0,20 | baseline losowy |
|---|---:|---:|---:|---:|
| PPO | 7,54 | 13,90 | **20,61** | 11,59 |
| DQN | 5,84 | 8,13 | **13,55** | 11,59 |
| Double DQN | 5,14 | 6,44 | **10,46** | 11,59 |
| Q-learning | 11,24 | 12,07 | 11,91 | 11,59 |

Udział wygranych rozdań w tych samych przebiegach:

| Agent | ε = 0 | ε = 0,05 | ε = 0,20 |
|---|---:|---:|---:|
| PPO | **0,0 %** | 10,5 % | **24,0 %** |
| DQN | 0,0 % | 0,5 % | 5,5 % |
| Double DQN | 0,0 % | 0,0 % | 0,0 % |
| Q-learning | 0,0 % | 0,0 % | 0,0 % |

Udział kroków wracających do pozycji już widzianej spada monotonicznie wraz z ε
(PPO: 78,0 % → 63,8 % → 46,8 %; DQN 80,8 % → 70,8 % → 55,3 %), a wynik rośnie
monotonicznie. Q-learning jest płaski w obu tabelach, bo jest już polityką
losową (22,9 % powtórzeń wobec 23,0 % dla agenta losowego) — patrz D8.

Rysunek: [`figures/action_rule_klondike.png`](figures/action_rule_klondike.png),
dane: [`tables/action_rule_klondike.csv`](tables/action_rule_klondike.csv),
[`raw/greedy_vs_epsilon.json`](raw/greedy_vs_epsilon.json).

### Dla PPO jest jeszcze ostrzejszy test

PPO uczy się **rozkładu** nad akcjami, a nie funkcji wartości. Ale jego tryb
ewaluacyjny wyrzuca ten rozkład i bierze `argmax`:

```python
# packages/core/src/rl_card_lib/agents/ppo_agent.py:211-212
if not self.training:
    return int(logits.argmax(dim=1).item())
```

Zmierzone na tych samych 200 rozdaniach TEST, tym samym checkpointem:

| Reguła wyboru akcji | karty na bazach | wygrane | powtórzone pozycje |
|---|---:|---:|---:|
| `argmax` nad polityką (protokół z pracy) | **7,54** | **0,0 %** | 78,0 % |
| próbkowanie z tej samej polityki | **22,45** | **28,5 %** | 44,8 % |

Dla porównania na tej samej puli: agent losowy 11,59 karty / 0,0 %,
`GreedyLookahead(1)` 9,22 / 1,0 %, MCTS(20) 26,80 / 37,0 %, heurystyka
28,74 / 45,5 %.

Czyli PPO grający **swoją własną polityką** jest niemal dwa razy lepszy od
baseline'u losowego, wygrywa 28,5 % rozdań i plasuje się między
`GreedyLookahead` a MCTS(20). Ten sam model oceniony przez `argmax` wygląda na
gorszy od losowego.

Na Macao efekt jest o rząd wielkości mniejszy (35,0 % → 39,5 %), bo epizody są
krótkie i **nie mają odwracalnych cykli** — udział powtórzonych pozycji wynosi
tam 0,0 %. To potwierdza mechanizm: problemem nie jest determinizm sam w sobie,
tylko determinizm **w grze z ruchami odwracalnymi**.

### Przyczyna

Ta sama co w D3, tylko widziana od strony protokołu, a nie środowiska:

* polityka zachłanna jest deterministyczna,
* Klondike ma ruchy odwracalne i nieograniczone przekładanie talii,
* więc pierwszy powrót do znanego stanu zamyka cykl na resztę epizodu,
* a jedyne, co ten cykl kiedykolwiek przerywało, to ε > 0 podczas treningu.

Do tego dochodzi błąd kategorii: PPO optymalizuje politykę stochastyczną, a
jest oceniany jako polityka deterministyczna. Dla metody on-policy próbkowanie
**jest** polityką; `argmax` nad nią to inna polityka, której nikt nie trenował.

### Poprawka

Trzy niezależne, każda działa osobno:

1. **Ewaluować politykę, której agent się nauczył.** Dla PPO to znaczy
   próbkować, a nie brać `argmax` — jedna linia w `PPOAgent.select_action`.
   **Scalone w PR [#33](https://github.com/mkh63d/rl-card-lib/pull/33)**
   (zgłoszenie #21): `PPOAgent` próbkuje w trybie ewaluacji, a `eval_greedy=True`
   pozwala jawnie poprosić o `argmax`. Ramię `fixed` używa próbkowania, ramię
   `asis` — `argmax`, więc obie liczby są w tabelach obok siebie.
2. **Włączyć karę za powtórzoną pozycję** (D3), żeby polityka zachłanna
   przestała cyklować — poprawka po stronie środowiska. **Scalone w PR
   [#29](https://github.com/mkh63d/rl-card-lib/pull/29)** (ramię `noloop`), ale
   pomiar mówi, że **to lekarstwo nie działa** — patrz D3.
3. **Raportować obie liczby.** Zachłanna i stochastyczna to dwie różne
   polityki; dla gry z ruchami odwracalnymi obie warto podać. Zrealizowane:
   `tables/solve_time_benchmark.csv` ma osobne wiersze `ppo (fixed)`
   i `ppo (fixed, sampled)`, a `hyperparameters.csv` podaje `eval_greedy`
   jako jawny hiperparametr.

### Konsekwencja dla tekstu pracy

Zdanie „On Klondike **no learner cleared the random baseline** of 12.2 cards”
przestaje być prawdziwe przy dowolnej z powyższych poprawek. Przy polityce,
której PPO faktycznie się nauczył, wynik to 22,45 karty przy baseline 11,59 —
prawie dwukrotnie powyżej. Podobnie „greedy win rates were zero across all
three learners” — PPO wygrywa 28,5 % rozdań.

---

## Wyniki ablacji

> Tabela wypełniana z [`raw/runs/`](raw/runs/) przez
> [`scripts/make_report.py`](scripts/make_report.py); wersja maszynowa w
> [`tables/ablation_fixes.csv`](tables/ablation_fixes.csv).
> Protokół: pula TRAIN = seedy 0–9999, pula TEST = 200 rozdań 100000–100199,
> ewaluacja zachłanna, 3 seedy inicjalizacji, średnia ± odchylenie standardowe.

<!-- ABLATION_TABLE -->

Metryka: Klondike — karty na bazach (0–52); Macao — win rate przeciwko heurystyce.

| gra | agent | ramię | seedy | TEST przed | TEST po | Δ |
|---|---|---|---|---|---|---|
| klondike | PPO | `asis` — as published | 3 | 2.24 ± 0.43 | 7.07 ± 0.25 | +4.83 |
| klondike | Double DQN | `asis` — as published | 3 | 1.90 ± 1.53 | 5.88 ± 0.30 | +3.98 |
| klondike | Double DQN | `fixed` — + time-limit bootstrap fix | 3 | 1.90 ± 1.53 | 5.38 ± 0.44 | +3.48 |
| klondike | DQN | `asis` — as published | 3 | 3.28 ± 0.93 | 5.67 ± 0.41 | +2.39 |
| klondike | DQN | `fixed` — + time-limit bootstrap fix | 3 | 3.28 ± 0.93 | 5.77 ± 0.07 | +2.49 |
| klondike | Q-learning | `asis` — as published | 3 | 11.31 ± 0.26 | 11.33 ± 0.20 | +0.02 |
| macao | PPO | `asis` — as published | 3 | 3.2 ± 2.9 % | 35.5 ± 1.7 % | +32.3 pp |
| macao | Double DQN | `asis` — as published | 3 | 0.0 ± 0.0 % | 7.5 ± 1.0 % | +7.5 pp |
| macao | Double DQN | `fixed` — + time-limit bootstrap fix | 3 | 0.0 ± 0.0 % | 8.7 ± 0.6 % | +8.7 pp |
| macao | DQN | `asis` — as published | 3 | 4.3 ± 7.5 % | 7.8 ± 0.6 % | +3.5 pp |
| macao | DQN | `fixed` — + time-limit bootstrap fix | 3 | 4.3 ± 7.5 % | 7.8 ± 1.0 % | +3.5 pp |
| macao | Q-learning | `asis` — as published | 3 | 2.2 ± 0.3 % | 2.2 ± 1.4 % | +0.0 pp |
