"""Shared local directory for service sockets, state, and diagnostic logs."""

import os
from pathlib import Path
import threading
from datetime import datetime


def auspex_directory():
    """Treat AUSPEX_HOME as a base directory, falling back to HOME."""
    return Path(os.environ.get("AUSPEX_HOME") or Path.home()).expanduser() / ".auspex"


def home_note():
    """Make implicit HOME-based service discovery visible to an operator."""
    if not os.environ.get("AUSPEX_HOME"):
        return f"AUSPEX_HOME not set; using {auspex_directory()} (HOME/.auspex)"
    return None


class DiagnosticLog:
    """Best-effort per-service debug log, truncated only after claiming its service lock."""

    def __init__(self, filename):
        self.path = auspex_directory() / filename
        self.stream = None
        self.lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.fchmod(fd, 0o600)
            self.stream = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
        except OSError as exc:
            print(f"[Auspex] Diagnostic log unavailable at {self.path}: {exc}", flush=True)

    def write(self, message):
        if self.stream is not None:
            with self.lock:
                try:
                    self.stream.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
                except (OSError, UnicodeError):
                    pass  # Logs are debugging aids, not service state.

    def close(self):
        if self.stream is not None:
            self.stream.close()
