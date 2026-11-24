"""Structured JSON logging (spec §537-538).

Telemetry (performance, operations) is written separately from science
(events, discoveries) so that operational noise never contaminates the
scientific record.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading

_LOCK = threading.Lock()
_FILE_HANDLER: logging.Handler | None = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "t": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        extra = getattr(record, "fields", None)
        tail = ""
        if extra:
            tail = " " + " ".join(f"{k}={v}" for k, v in extra.items())
        msg = f"[{ts}] {record.levelname[0]} {record.name}: {record.getMessage()}{tail}"
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return msg


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"genesis.{name}")
    root = logging.getLogger("genesis")
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(ConsoleFormatter())
        root.addHandler(h)
        root.setLevel(os.environ.get("GENESIS_LOG_LEVEL", "INFO"))
        root.propagate = False
    return logger


def attach_file_log(path: str) -> None:
    """Route all genesis logs additionally to a JSONL telemetry file."""
    global _FILE_HANDLER
    with _LOCK:
        root = logging.getLogger("genesis")
        if _FILE_HANDLER is not None:
            root.removeHandler(_FILE_HANDLER)
            _FILE_HANDLER.close()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
        _FILE_HANDLER = fh


def log_fields(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
