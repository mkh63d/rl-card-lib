# Installation

The library requires **Python 3.9+**. Its main runtime dependencies are
`numpy`, `torch`, `gymnasium`, `matplotlib` and `tqdm`.

## Install everything (development)

From the repository root:

```bash
# Install the root metapackage (pulls in all sub-packages)
pip install -e .
```

The training and neural-network agents use PyTorch. If you do not need a GPU,
the CPU wheels are far smaller:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## Install individual packages

Each package is independently installable, so you can take only what you need:

```bash
pip install -e ./packages/core        # foundation: Game, env, agents, Trainer
pip install -e ./packages/cardgames   # Card, Deck, Player, CardGame, rules
pip install -e ./packages/visualizer  # text rendering of cards and tableaux
pip install -e ./packages/report      # TrainingReport, RunRecord, HtmlReport
pip install -e ./packages/examples    # Klondike, Macao, demo & training scripts
```

## Optional extras

```bash
# Charts and the HTML report (adds matplotlib)
pip install -e "./packages/report[charts]"

# Development tools (pytest, black, isort, mypy, flake8, pylint)
pip install -e ".[dev]"

# Build this documentation site (mkdocs, material theme, mkdocstrings)
pip install -e ".[docs]"
```

`matplotlib` is an optional extra of the report package. Without it, the JSON
and Markdown APIs still work and figure rendering simply becomes a no-op — the
same way `TrainingMetrics.plot()` behaves in the core package.

## Verify the install

```bash
python packages/examples/scripts/quick_demo.py
```

Next: the [Quickstart](quickstart.md).
