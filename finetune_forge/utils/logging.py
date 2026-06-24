# finetune_forge/utils/logging.py

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO, fmt: str = _DEFAULT_FORMAT) -> None:
    """Configure root logging once for the whole application.

    Safe to call multiple times — ``force=True`` resets handlers so repeated
    calls (e.g. from the CLI and from tests) don't duplicate log lines.
    """
    logging.basicConfig(level=level, format=fmt, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
