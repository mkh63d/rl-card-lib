# Integracja z Gymnasium — co pochodzi z biblioteki, a co jest własne

> **Status:** materiał źródłowy dla rozdziałów 4 i 6 pracy. Każde stwierdzenie
> pochodzi z kodu w `packages/` albo z uruchomionego testu.
> Surowe dane: [`raw/gymnasium_probe.json`](raw/gymnasium_probe.json) ·
> log: [`logs/gymnasium_probe.log`](logs/gymnasium_probe.log) ·
> skrypt: [`scripts/probe_gymnasium.py`](scripts/probe_gymnasium.py)
> Wersje użyte do pomiaru: `gymnasium 1.3.0`, `stable-baselines3 2.9.0`,
> Python 3.12.10.

## Odpowiedź w jednym akapicie

Z Gymnasium pochodzą **wyłącznie obiekty przestrzeni** (`spaces.Box`,
`spaces.Discrete`, a w wariancie maskowanym dodatkowo `spaces.Dict` i
`spaces.MultiBinary`) oraz **konwencja** pięcioelementowej krotki
`(observation, reward, terminated, truncated, info)`. Cała reszta — pętla
uczenia, bufor odtwarzania, maskowanie akcji, agenci, kodowanie obserwacji,
metryki — jest napisana od zera. Środowisko **nie dziedziczy** po
`gymnasium.Env`, więc `gymnasium.utils.env_checker.check_env` je odrzuca,
wrappery Gymnasium go nie przyjmują, a Stable-Baselines3 nie przyjmuje go bez
adaptera.

---

## 1. Wszystkie miejsca w repo, gdzie importowane jest `gymnasium`

Wyszukiwanie wzorca `gymnasium|import gym|gym\.` po całym repozytorium (bez
katalogu `site/`, który jest wygenerowaną dokumentacją) daje **dwa pliki
źródłowe**:

| Plik | Linie | Co jest importowane | Co z tego jest faktycznie użyte |
|---|---|---|---|
| [packages/core/src/rl_card_lib/env/card_game_env.py](../packages/core/src/rl_card_lib/env/card_game_env.py#L9-L14) | 9–14 | `import gymnasium as gym`, `from gymnasium import spaces` | **tylko `spaces`**; nazwa `gym` nie występuje nigdzie dalej w pliku |
| [packages/core/src/rl_card_lib/core/gym_wrapper.py](../packages/core/src/rl_card_lib/core/gym_wrapper.py#L7-L12) | 7–12 | `import gymnasium as gym`, `from gymnasium import spaces` | **tylko `spaces`**; nazwa `gym` nie występuje nigdzie dalej w pliku |

Pozostałe trafienia to deklaracje zależności i dokumentacja, nie kod:
`pyproject.toml:31` oraz `packages/core/pyproject.toml:29` (`gymnasium>=0.29.0`),
`README.md:269`, `TODO.md:18`, `docs/getting-started/installation.md:4`,
`.github/workflows/ci.yml:29`, `docx/package_diagram.puml:535` (klasa
`gymnasium.spaces` na diagramie pakietów) i komentarz w
`packages/examples/src/rl_card_lib/harness/registry.py:6`.

Oba importy są w bloku `try/except Exception` z fallbackiem `gym = None;
spaces = None`, czyli Gymnasium jest **zależnością opcjonalną** — biblioteka
uruchamia się bez niej, tracąc jedynie obiekty przestrzeni.

### 1a. Co konkretnie z `spaces` jest używane

| Konstrukcja Gymnasium | Miejsce w kodzie | Wartość dla obu gier |
|---|---|---|
| `spaces.Box(low=-inf, high=+inf, shape, float32)` | [card_game_env.py:58-63](../packages/core/src/rl_card_lib/env/card_game_env.py#L58-L63), [gym_wrapper.py:24-26](../packages/core/src/rl_card_lib/core/gym_wrapper.py#L24-L26) | Klondike `Box(-inf, inf, (221,), float32)`; Macao `Box(-inf, inf, (126,), float32)` |
| `spaces.Discrete(n)` | [card_game_env.py:74](../packages/core/src/rl_card_lib/env/card_game_env.py#L74), [gym_wrapper.py:33](../packages/core/src/rl_card_lib/core/gym_wrapper.py#L33) | Klondike `Discrete(68)`; Macao `Discrete(65)` |
| `spaces.Dict({...})` | [card_game_env.py:212-215](../packages/core/src/rl_card_lib/env/card_game_env.py#L212-L215) | tylko w `MaskedCardGameEnv` |
| `spaces.MultiBinary(n)` | [card_game_env.py:214](../packages/core/src/rl_card_lib/env/card_game_env.py#L214) | tylko w `MaskedCardGameEnv`; `MultiBinary(65)` dla Macao |

To są **cztery wywołania w dwóch plikach**. Nic więcej z Gymnasium nie jest
wołane w całym repozytorium.

### 1b. Czego z Gymnasium nie ma

Sprawdzone programowo dla wszystkich czterech klas środowisk
(`CardGameEnv`, `MaskedCardGameEnv`, `GymEnvWrapper`), dla obu gier:

| Element kontraktu Gymnasium | Obecny? |
|---|---|
| dziedziczenie po `gymnasium.Env` | **nie** — MRO to `['CardGameEnv', 'object']` |
| `metadata` (np. `render_modes`) | **nie** |
| `spec` | **nie** |
| `self.np_random` zasilany przez `Env.reset(seed=…)` | **nie** |
| `unwrapped` | **nie** |
| rejestracja przez `gymnasium.register` / `gymnasium.make` | **nie** — w repo nie ma ani jednego wywołania |
| `gymnasium.Wrapper`, `TimeLimit`, `RecordEpisodeStatistics`, `VectorEnv` | **nie używane i nie dające się użyć** (§4) |
| `render_mode` jako atrybut | tak, ale własna implementacja (`"human"` / `"ansi"`), bez wpisu w `metadata` |

---

## 2. Co w pętli uczenia jest w całości własne

Żaden z poniższych elementów nie ma odpowiednika importowanego z Gymnasium ani
z żadnej innej biblioteki RL. PyTorch jest używany wyłącznie jako biblioteka
sieci i optymalizatorów (`nn.Linear`, `Adam`, autograd) — nie dostarcza ani
jednego algorytmu uczenia ze wzmocnieniem.

| Element | Plik i linie | Rozmiar | Na czym polega własna robota |
|---|---|---|---|
| `Trainer` — pętla epizodów, logowanie, ewaluacja, checkpointy | [trainer/trainer.py:14-302](../packages/core/src/rl_card_lib/trainer/trainer.py#L14-L302) | ~290 linii | własne `_run_episode`, `evaluate`, `_save_checkpoint`, callback per-epizod |
| `SelfPlayTrainer` — gra wieloosobowa, zamrożony snapshot przeciwnika | [trainer/trainer.py:305-481](../packages/core/src/rl_card_lib/trainer/trainer.py#L305-L481) | ~176 linii | `_snapshot_agent()` robi `deepcopy` agenta z odpiętym buforem i grą; `_current_player()` pyta grę, kto jest na ruchu, zamiast naiwnie przełączać (czwórka w Macao pomija turę) |
| `ReplayBuffer` | [dqn_agent.py:59-103](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L59-L103) | 45 linii | `deque(maxlen=capacity)` + `random.sample` |
| `MaskedReplayBuffer` — bufor pamiętający także maskę legalnych akcji **następnego** stanu | [dqn_agent.py:106-179](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L106-L179) | 74 linie | rozwiązanie problemu specyficznego dla gier karcianych; nie ma go w standardowych implementacjach DQN |
| Maskowanie przy **wyborze** akcji | [dqn_agent.py:317-324](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L317-L324), [ppo_agent.py:202-216](../packages/core/src/rl_card_lib/agents/ppo_agent.py#L202-L216) | — | DQN dodaje `-inf` do nielegalnych; PPO używa `masked_fill(~mask, -1e8)` — prawdziwe `-inf` dawałoby `NaN` w członie entropii |
| Maskowanie w **celu TD** | [dqn_agent.py:379-384](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L379-L384) | — | `masked_fill(~next_masks, MASK_VALUE)` plus flaga `has_actions`, żeby stan bez ruchów nie bootstrapował |
| Ekspozycja maski przez środowisko | [card_game_env.py:178-182](../packages/core/src/rl_card_lib/env/card_game_env.py#L178-L182), [card_game_env.py:232-236](../packages/core/src/rl_card_lib/env/card_game_env.py#L232-L236) | — | **rozszerzenie poza kontrakt Gymnasium** |
| `QLearningAgent` (tabularny) | [agents/tabular.py](../packages/core/src/rl_card_lib/agents/tabular.py) | 243 linie | klucz tablicy = zaokrąglona obserwacja jako `bytes` |
| `DQNAgent` | [dqn_agent.py:182-462](../packages/core/src/rl_card_lib/agents/dqn_agent.py#L182-L462) | 281 linii | |
| `DoubleDQNAgent` + `DuelingQNetwork` | [agents/double_dqn_agent.py](../packages/core/src/rl_card_lib/agents/double_dqn_agent.py) | — | podwójny wybór/ocena, głowice V/A, strata Hubera |
| `PPOAgent` + `ActorCritic` + GAE | [agents/ppo_agent.py](../packages/core/src/rl_card_lib/agents/ppo_agent.py) | 462 linie | własne `_compute_advantages` (GAE), własny clipped surrogate |
| `MCTSAgent` (UCT + determinizacja) | [agents/mcts_agent.py](../packages/core/src/rl_card_lib/agents/mcts_agent.py) | — | zwroty per gracz, `_MinMaxStats` do normalizacji Q w UCB1 |
| `HeuristicAgent`, `GreedyLookaheadAgent`, `RandomAgent` | [agents/heuristic.py](../packages/core/src/rl_card_lib/agents/heuristic.py), [agents/random_agent.py](../packages/core/src/rl_card_lib/agents/random_agent.py) | — | |
| Kodowanie obserwacji | [klondike.py:151-213](../packages/examples/src/rl_card_lib/games/klondike.py#L151-L213), [macao.py:174-237](../packages/examples/src/rl_card_lib/games/macao.py#L174-L237) | — | Klondike: 52×4 (lokalizacja i widoczność karty) + 4 (wierzchołki baz) + 7 (rozmiary kolumn) + 2 = **221**. Macao: 52 (ręka) + 52 (wierzchołek stosu) + 4 (żądany kolor) + 13 (żądana figura) + 2 (faza deklaracji) + 1 (kara) + (n−1) (ręce przeciwników) + 1 (talia) = **126** dla 2 graczy |
| `TrainingMetrics` | [trainer/metrics.py](../packages/core/src/rl_card_lib/trainer/metrics.py) | 219 linii | listy `rewards/steps/wins/losses`, `get_moving_average` (okno 100), `save`/`load` JSON, `plot` |
| `TrainingReport`, `RunRecord`, `RunStore`, `HtmlReport` | [packages/report/](../packages/report/src/rl_card_lib/report/) | — | poza zakresem Gymnasium w całości |

**Zdanie do pracy.** Gymnasium pełni tu rolę *słownika typów*, nie frameworka.
Gdyby usunąć `gymnasium` z zależności i zastąpić `spaces.Box` / `spaces.Discrete`
dwiema własnymi dataklasami z polami `shape`, `dtype`, `n`, biblioteka
działałaby bez żadnej innej zmiany: obie klasy przestrzeni są używane wyłącznie
do odczytania `.shape[0]` i `.n` przy budowaniu agenta
([run_sweep.py:65-67](../packages/examples/scripts/run_sweep.py#L65-L67)).

---

## 3. Gdzie `CardGameEnv` realizuje kontrakt Gymnasium, a gdzie go rozszerza

### 3a. Realizuje (zgodnie z API Gymnasium ≥ 0.26)

Zweryfikowane w czasie wykonania, na obu grach:

| Wymaganie | Realizacja | Weryfikacja |
|---|---|---|
| `reset(*, seed=None, options=None) -> (obs, info)` | [card_game_env.py:78-103](../packages/core/src/rl_card_lib/env/card_game_env.py#L78-L103) | zwraca `(ndarray, dict)`; `info` ma klucz `legal_actions` |
| `step(action) -> (obs, reward, terminated, truncated, info)` | [card_game_env.py:120-156](../packages/core/src/rl_card_lib/env/card_game_env.py#L120-L156) | krotka 5-elementowa, typy `['ndarray','float','bool','bool','dict']` |
| rozdzielenie `terminated` / `truncated` (konwencja od Gymnasium 0.26) | `terminated` z gry; `truncated` z `max_steps` ([:135-136](../packages/core/src/rl_card_lib/env/card_game_env.py#L135-L136)) oraz z `Macao._finish_step` ([macao.py:495](../packages/examples/src/rl_card_lib/games/macao.py#L495)) | oba są `bool` |
| `observation_space` / `action_space` jako obiekty `gymnasium.spaces` | `Box` / `Discrete` | `observation_space.contains(obs)` = `True` dla obu gier |
| `render()` / `close()` | [:158-170](../packages/core/src/rl_card_lib/env/card_game_env.py#L158-L170) | `render_mode` ∈ {`None`, `"human"`, `"ansi"`} |
| seed nie dotyka globalnego RNG | [:105-118](../packages/core/src/rl_card_lib/env/card_game_env.py#L105-L118) — `_game_reset_accepts_seed()` przekazuje seed do `game.reset(seed=…)` | `env.reset(seed=7)` dwukrotnie → identyczne rozdanie (zmierzone) |

### 3b. Rozszerza poza kontrakt

| Rozszerzenie | Miejsce | Uwaga |
|---|---|---|
| `get_legal_actions() -> list[int]` | [:172-176](../packages/core/src/rl_card_lib/env/card_game_env.py#L172-L176) | metoda spoza API Gymnasium |
| `get_legal_action_mask() -> ndarray[bool]` | [:178-182](../packages/core/src/rl_card_lib/env/card_game_env.py#L178-L182) | jw. |
| `info["legal_actions"]` przy `reset` i `step` | [:100-102](../packages/core/src/rl_card_lib/env/card_game_env.py#L100-L102), [:140](../packages/core/src/rl_card_lib/env/card_game_env.py#L140) | Gymnasium nie definiuje żadnego standardu maski; to konwencja własna |
| `action_to_string(action)` | [:184-188](../packages/core/src/rl_card_lib/env/card_game_env.py#L184-L188) | |
| absorpcja nielegalnej akcji zamiast wyjątku | [:121-130](../packages/core/src/rl_card_lib/env/card_game_env.py#L121-L130) | gra **nie** jest krokowana; zwracane jest `invalid_action_reward = -1.0` |
| kara za powtórzenie pozycji | [:143-151](../packages/core/src/rl_card_lib/env/card_game_env.py#L143-L151) | `info["repeated_position"]`, klucz = hash bajtów obserwacji |

### 3c. `MaskedCardGameEnv` — obserwacja jako `Dict`

[card_game_env.py:191-236](../packages/core/src/rl_card_lib/env/card_game_env.py#L191-L236)
podmienia `observation_space` na

```
Dict('action_mask': MultiBinary(65),
     'observation': Box(-inf, inf, (126,), float32))
```

i zwraca z `reset`/`step` słownik `{"observation": …, "action_mask": …}`.
Maska jest typu `int8`, nie `bool` ([:232-236](../packages/core/src/rl_card_lib/env/card_game_env.py#L232-L236)),
bo `MultiBinary` wymaga typu całkowitego.

Dwie rzeczy warte odnotowania w pracy:

1. To jest konstrukcja **w pełni legalna w Gymnasium** (`Dict` to standardowa
   przestrzeń) i **dokładnie ten wzorzec**, którego oczekuje `sb3-contrib`
   `MaskablePPO`. Czyli droga do interoperacyjności jest w bibliotece już
   w połowie przetarta.
2. Ale `MaskedCardGameEnv` **nie jest używany przez żaden skrypt treningowy
   ani przez sweep**: `registration.py` buduje oba środowiska zwykłym
   `CardGameEnv`
   ([registration.py:58](../packages/examples/src/rl_card_lib/games/registration.py#L58),
   [:82](../packages/examples/src/rl_card_lib/games/registration.py#L82)).
   Agenci biblioteki dostają maskę kanałem `info["legal_actions"]`, a nie przez
   obserwację. `MaskedCardGameEnv` jest więc możliwością, nie ścieżką
   produkcyjną.

---

## 4. Czy środowisko przechodzi `gymnasium.utils.env_checker.check_env`?

**Nie. Żadna z czterech konfiguracji.** Uruchomione na gymnasium 1.3.0:

```
check_env(CardGameEnv(KlondikeSolitaire(seed=0), max_steps=300))   -> TypeError
check_env(CardGameEnv(Macao(num_players=2, seed=0), max_steps=200)) -> TypeError
check_env(MaskedCardGameEnv(Macao(num_players=2, seed=0)))          -> TypeError
check_env(GymEnvWrapper(Macao(num_players=2, seed=0)))              -> TypeError
```

Komunikat we wszystkich czterech przypadkach identyczny co do treści:

```
TypeError: The environment must inherit from the gymnasium.Env class,
actual class: <class 'rl_card_lib.env.card_game_env.CardGameEnv'>.
See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.
```

Check przerywa się na **pierwszym** teście (`isinstance`), więc reszta kontraktu
nie została w ogóle sprawdzona. To trzeba w pracy napisać uczciwie: *nie
wiadomo*, czy środowisko przeszłoby pozostałe testy — wiadomo tylko, że nie
przechodzi testu wejściowego.

To samo dotyczy wrapperów Gymnasium:

```
gym.wrappers.TimeLimit(CardGameEnv(Macao(...)), max_episode_steps=50)
  -> AssertionError: Expected env to be a `gymnasium.Env` but got
     <class 'rl_card_lib.env.card_game_env.CardGameEnv'>
gym.wrappers.RecordEpisodeStatistics(CardGameEnv(Macao(...)))
  -> AssertionError: (identyczny komunikat)
```

### 4a. Co trzeba zmienić, żeby przechodziło

Cztery zmiany, wszystkie mechaniczne:

1. `class CardGameEnv(gym.Env):` zamiast `class CardGameEnv:`, plus
   `super().__init__()`;
2. atrybut klasowy `metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}`;
3. `super().reset(seed=seed)` w `reset()`, żeby powstał `self.np_random`;
4. gałąź fallbacku `gym = None` musi wtedy dostarczyć atrapę klasy bazowej,
   inaczej biblioteka przestaje się importować bez Gymnasium.

Piąta rzecz nie jest formalnością, tylko realną wadą (§5c): przy polityce
proponującej same nielegalne akcje **epizod nigdy się nie kończy**, więc nawet
po powyższych zmianach `check_env` mógłby przejść, a mimo to generyczny
konsument Gymnasium by się zawiesił.

---

## 5. Minimalny przykład ze Stable-Baselines3

### 5a. Bez adaptera — nie działa

```python
from stable_baselines3 import PPO
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import Macao

PPO("MlpPolicy", CardGameEnv(Macao(num_players=2, seed=0), max_steps=200))
```

```
ValueError: The environment is of type
<class 'rl_card_lib.env.card_game_env.CardGameEnv'>, not a Gymnasium
environment. In this case, we expect OpenAI Gym to be installed and the
environment to be an OpenAI Gym environment.
```

Identyczny błąd dla Klondike. SB3 sprawdza `isinstance(env, gym.Env)`, zanim
zrobi cokolwiek innego, więc przyczyna jest ta sama co w §4.

### 5b. Z ~30-liniowym adapterem — działa

Adapter znajduje się w [`scripts/probe_gymnasium.py`](scripts/probe_gymnasium.py)
jako klasa `GymnasiumAdapter`. **Nic w `packages/` nie było zmieniane.**

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_card_lib.env import CardGameEnv
from rl_card_lib.games import Macao


class GymnasiumAdapter(gym.Env):
    """Dokłada to, czego CardGameEnv nie deklaruje: klasę bazową, metadata
    i self.np_random. Maskę podaje w info, bo SB3 jej nie rozumie."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, inner: CardGameEnv):
        super().__init__()
        self.inner = inner
        self.observation_space = inner.observation_space
        self.action_space = inner.action_space
        self.render_mode = inner.render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.inner.reset(seed=seed, options=options)
        info = dict(info)
        info["action_mask"] = self.inner.get_legal_action_mask()
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(int(action))
        info = dict(info)
        info["action_mask"] = self.inner.get_legal_action_mask()
        return (np.asarray(obs, dtype=np.float32), float(reward),
                bool(terminated), bool(truncated), info)

    def render(self):
        return self.inner.render()

    def close(self):
        self.inner.close()


env = GymnasiumAdapter(CardGameEnv(Macao(num_players=2, seed=0), max_steps=200))
check_env(env, warn=True, skip_render_check=True)     # przechodzi, bez ostrzeżeń
model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, seed=0, device="cpu")
model.learn(total_timesteps=2000)
```

Wynik uruchomienia (klucz `stable_baselines3` w `raw/gymnasium_probe.json`):

```
Macao     : sb3 check_env -> PASS (0 ostrzeżeń);  PPO.learn(2000) -> PASS
Klondike  : sb3 check_env -> PASS (0 ostrzeżeń);  PPO.learn(2000) -> PASS
```

Epizod demonstracyjny po treningu (`deterministic=True`, rozdanie `seed=100000`):

| Gra | kroki | suma nagród |
|---|---:|---:|
| Klondike | 300 (limit `max_steps`) | −4,2 |
| Macao | 300 (przerwane licznikiem pętli, patrz niżej) | −300,0 |

### 5c. Dlaczego ten wynik jest merytorycznie pusty — i to jest właśnie wynik

Nagroda −300,0 w 300 krokach Macao to dokładnie 300 × `invalid_action_reward`
(−1,0): **polityka SB3 nie wybrała ani jednej legalnej akcji**. To nie jest
awaria SB3. W Macao w typowej pozycji legalne są 2–4 akcje z 65 (w rozdaniu
`seed=100003`: 4 z 65, czyli ~6 %), a SB3 nie ma dostępu do maski.

Co więcej, epizod **nie zakończył się sam**. W
[card_game_env.py:121-130](../packages/core/src/rl_card_lib/env/card_game_env.py#L121-L130)
nielegalna akcja wraca z funkcji **przed** `self._step_count += 1`
([:133](../packages/core/src/rl_card_lib/env/card_game_env.py#L133)), więc licznik
kroków nie rośnie i `max_steps` nigdy nie zostaje osiągnięty. Zmierzone
bezpośrednio:

```
5000 kolejnych nielegalnych akcji przy max_steps=200
  -> terminated=False, truncated=False, env._step_count = 0
```

Dla generycznego konsumenta Gymnasium jest to **zawieszenie**, nie długi epizod.
Wewnętrzny `Trainer` biblioteki tego nie ujawnia, bo jego agenci zawsze
dostają `legal_actions` i nigdy nie proponują nielegalnego ruchu.

**Zdanie do pracy.** Kompatybilność z Gymnasium jest w tej bibliotece
*nominalna* (typy i konwencja sygnatur), a nie *operacyjna*. Uczciwe
sformułowanie: „środowisko przyjmuje konwencję API Gymnasium, ale nie
implementuje klasy `gymnasium.Env`, więc nie współpracuje z `env_checker`,
z wrapperami ani z zewnętrznymi bibliotekami algorytmów bez cienkiego adaptera;
dodatkowo, ponieważ maskowanie akcji odbywa się poza obserwacją, algorytm
zewnętrzny działa bez maski i w tych grach nie osiąga niczego.” Ścieżką
naprawy, której biblioteka ma już połowę, jest `MaskedCardGameEnv` +
`sb3-contrib MaskablePPO`; nie jest ona w repozytorium ani wykorzystana, ani
zmierzona.

---

## 6. Podsumowanie do wklejenia

| Warstwa | Źródło | Dowód |
|---|---|---|
| Przestrzenie obserwacji i akcji (`Box`, `Discrete`, `Dict`, `MultiBinary`) | **Gymnasium** | 4 wywołania w 2 plikach |
| Sygnatury `reset`/`step`, konwencja `terminated`/`truncated` | **Gymnasium (konwencja)** | typy zweryfikowane w runtime |
| Klasa bazowa `gymnasium.Env`, `metadata`, `np_random`, `spec`, rejestracja, wrappery, `VectorEnv` | **nieużywane** | `check_env` → `TypeError` na wszystkich 4 klasach |
| Pętla treningowa, self-play, bufory, maskowanie, agenci, kodowanie, metryki, raport | **własne** | **4 376** linii w warstwie uczącej (trener 699, agenci 2 262, środowisko 467, harness 948); całe `packages/` to 14 083 linie w 69 plikach |
| Współpraca ze Stable-Baselines3 | **wymaga adaptera** (~30 linii); po adapterze trening rusza, ale bez maski agent gra wyłącznie nielegalnie | log w `raw/gymnasium_probe.json` |
