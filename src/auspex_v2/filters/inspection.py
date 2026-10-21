"""Local, read-only snapshots for the optional filter-graph viewer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import socketserver
import threading

PROTOCOL_VERSION = 1
MAX_REQUEST = 4096
MAX_SNAPSHOT = 1024 * 1024


def fetch_snapshot(path):
    """Request one bounded snapshot without importing Qt or accessing filter workers."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3)
        client.connect(str(path))
        client.sendall(b'{"version":1,"op":"SNAPSHOT"}\n')
        with client.makefile("rb") as stream:
            reply = stream.readline(MAX_SNAPSHOT + 1)
        if not reply.endswith(b"\n") or len(reply) > MAX_SNAPSHOT:
            raise ValueError("Incomplete or oversized graph snapshot")
    message = json.loads(reply)
    if message.get("version") != PROTOCOL_VERSION or not message.get("ok"):
        raise ValueError(message.get("error", "Unsupported graph response"))
    return message["snapshot"]


class _Handler(socketserver.StreamRequestHandler):
    """Answer exactly one bounded, versioned request per connection."""

    def handle(self):
        self.request.settimeout(2)
        try:
            raw = self.rfile.readline(MAX_REQUEST + 1)
            if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST:
                raise ValueError("Invalid request length")
            request = json.loads(raw)
            if request != {"version": PROTOCOL_VERSION, "op": "SNAPSHOT"}:
                raise ValueError("Unsupported graph request")
            reply = {"version": PROTOCOL_VERSION, "ok": True,
                     "snapshot": self.server.snapshot()}
            encoded = json.dumps(reply, allow_nan=False).encode() + b"\n"
            if len(encoded) > MAX_SNAPSHOT:
                raise ValueError("Graph snapshot exceeds protocol limit")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            encoded = json.dumps({"version": PROTOCOL_VERSION, "ok": False, "error": str(exc)}).encode() + b"\n"
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    """Serve concurrent read-only requests without blocking the filter supervisor."""

    daemon_threads = True
    block_on_close = False


class GraphInspector:
    """Own a socket in a private directory for the lifetime of one filter graph."""

    def __init__(self, path, snapshot):
        self.path = Path(path).expanduser().absolute()
        parent = self.path.parent
        stat = parent.stat()
        if not parent.is_dir() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise ValueError("Inspection socket requires a user-owned private directory (mode 0700)")
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        self.server = None
        try:
            # umask is process-global; require a private parent, and do not start
            # the listening thread until chmod has restricted the socket itself.
            self.server = _UnixServer(str(self.path), _Handler)
            os.chmod(self.path, 0o600)
            self.server.snapshot = snapshot
            self.inode = self.path.stat().st_ino
            self.thread = threading.Thread(target=self.server.serve_forever,
                                           kwargs={"poll_interval": 0.1},
                                           daemon=True, name="auspex-filter-inspector")
            self.thread.start()
        except BaseException:
            if self.server is not None:
                self.server.server_close()
            self.path.unlink(missing_ok=True)
            raise

    def close(self):
        """Stop serving and remove only the socket this inspector created."""
        if self.server is None:
            return
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        if self.path.exists() and self.path.stat().st_ino == self.inode:
            self.path.unlink()
        self.server = None
