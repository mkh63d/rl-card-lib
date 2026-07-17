"""Package entry for `rl_card_lib` when installed from `packages/report`."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from . import report

__all__ = ["report"]
