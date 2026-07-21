"""MkDocs hooks for the documentation build.

``--strict`` escalates every WARNING log record to a build error. That is what
we want for MkDocs' own link and navigation checks, but griffe (via
mkdocstrings) also emits doc-quality nags such as "No type or annotation for
parameter ..." for library functions that document a parameter without
annotating it. Those are informational, not build-breaking, so this hook drops
that one class of message while leaving every other warning — including all of
MkDocs' link/nav validation — strict.
"""

from __future__ import annotations

import logging

_SILENCED_SUBSTRINGS = (
    "No type or annotation for",
)

# griffe logs through this logger under mkdocstrings; a filter here drops the
# record before it can reach MkDocs' strict warning counter on the root logger.
_ORIGINATING_LOGGERS = (
    "griffe",
    "mkdocs.plugins.griffe",
    "mkdocs.plugins.mkdocstrings",
)


class _DropDocQualityWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(s in message for s in _SILENCED_SUBSTRINGS)


def on_config(config, **kwargs):
    filt = _DropDocQualityWarnings()
    # getLogger creates the logger if griffe/mkdocstrings has not yet, so the
    # filter is in place regardless of plugin load order.
    for name in _ORIGINATING_LOGGERS:
        logging.getLogger(name).addFilter(filt)
    return config
