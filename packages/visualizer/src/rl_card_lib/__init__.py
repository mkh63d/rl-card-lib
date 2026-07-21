"""Package entry for `rl_card_lib` when installed from `packages/visualizer`."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from . import visualizer

__all__ = ["visualizer"]
