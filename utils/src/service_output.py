"""Launcher-owned, bounded service output capture independent of the terminal UI."""

from collections import deque
import threading
import time


class ServiceOutput:
    """Drain service pipes immediately without ever waiting for a UI consumer."""

    def __init__(self, max_lines=5000):
        if max_lines < 1:
            raise ValueError("max_lines must be positive")
        self.lines = deque(maxlen=max_lines)
        self.dropped = 0
        self.lock = threading.Lock()
        self.readers = []
        self.finalized = False

    def attach(self, process, name):
        """Start capture as soon as Popen returns, before waiting for service readiness."""
        reader = threading.Thread(target=self._read, args=(process.stdout, name), daemon=True)
        self.readers.append((name, reader))
        reader.start()

    def _append(self, line):
        with self.lock:
            if len(self.lines) == self.lines.maxlen:
                self.dropped += 1
            self.lines.append(line)

    def _read(self, stream, name):
        """Only the reader closes its stream; closing a blocked buffered read elsewhere can hang."""
        try:
            for line in stream:
                self._append(line.rstrip("\n"))
        except (OSError, ValueError, UnicodeError) as exc:
            self._append(f"[Auspex] Output read failed for {name}: {exc}")
        finally:
            stream.close()

    def drain(self):
        """Consume pending lines and report overflow rather than silently losing diagnostics."""
        with self.lock:
            lines = list(self.lines)
            self.lines.clear()
            dropped, self.dropped = self.dropped, 0
        if dropped:
            lines.insert(0, f"[Auspex] Output buffer overflow: {dropped} service lines dropped; retaining newest lines.")
        return lines

    def finish(self, timeout=1.0):
        """Allow one total EOF budget after service shutdown; never kill a pipe-holding descendant."""
        if self.finalized:
            return
        deadline = time.monotonic() + timeout
        for name, reader in self.readers:
            reader.join(max(0, deadline - time.monotonic()))
        for name, reader in self.readers:
            if reader.is_alive():
                self._append(f"[Auspex] {name}: service exited, but output remains open; "
                             "a descendant may still be running. Inspect hardware and leases.")
        # Readers are daemons and may continue until EOF. Their buffer stays bounded;
        # they never call the UI and must not prevent the launcher from exiting.
        self.finalized = True
