"""Smoke tests for public import surface."""

import importlib

import rl_card_lib


def _assert_all_exports(module):
    assert hasattr(module, "__all__")
    for name in module.__all__:
        getattr(module, name)


def test_top_level_imports():
    _assert_all_exports(rl_card_lib)

    required = [
        "Card",
        "Deck",
        "CardGame",
        "Player",
        "CardGameEnv",
        "DQNAgent",
        "RandomAgent",
        "Trainer",
    ]
    for name in required:
        getattr(rl_card_lib, name)


def test_subpackage_imports():
    submodules = [
        "rl_card_lib.core",
        "rl_card_lib.env",
        "rl_card_lib.games",
        "rl_card_lib.agents",
        "rl_card_lib.trainer",
    ]

    for module_path in submodules:
        module = importlib.import_module(module_path)
        _assert_all_exports(module)
