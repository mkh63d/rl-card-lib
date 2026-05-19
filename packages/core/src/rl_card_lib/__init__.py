"""Package entry for `rl_card_lib` when installed from `packages/core`.

Expose the `core` subpackage so consumers importing the package get
the core API available.
"""

from . import core

__all__ = ["core"]
