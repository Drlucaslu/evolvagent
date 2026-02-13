"""
Unified logging setup for EvolvAgent.

Configures root logger with console (INFO) and rotating file (DEBUG) handlers.
Call setup_logging() once at CLI entry point.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "",
    log_file: str = "evolvagent.log",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> None:
    """Configure root logger with console and optional file handler.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If empty, file logging is skipped.
        log_file: Name of the log file.
        max_bytes: Max bytes per log file before rotation.
        backup_count: Number of rotated backups to keep.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    fmt = logging.Formatter(_LOG_FORMAT)

    # Console handler — respects the requested level
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — always DEBUG, with rotation
    if log_dir:
        log_path = Path(log_dir).expanduser().resolve()
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def reset_logging() -> None:
    """Remove all handlers from root logger (for testing)."""
    global _initialized
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    _initialized = False
