"""Synchronous, local-only client for the instrument coordinator."""

import json
import socket

from .bulk import decode_result
from .coordinator import MAX_MESSAGE, PROTOCOL_VERSION, default_socket


class RemoteError(Exception):
    """Coordinator rejected an operation; no result was committed to this client."""


class CoordinatorClient:
    """Connect notebooks or scripts to the persistent local instrument service."""

    def __init__(self, socket_path=None):
        """Open the Unix socket and verify the coordinator protocol version."""
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(10)
        try:
            self.socket.connect(str(socket_path or default_socket()))
        except BaseException:
            self.socket.close()
            raise
        self.file = self.socket.makefile("rwb")
        try:
            self.request("HELLO", version=PROTOCOL_VERSION)
        except Exception:
            self.close()
            raise

    def request(self, op, **kwargs):
        """Exchange one size-limited JSON request and raise on remote failure."""
        payload = json.dumps({"op": op, **kwargs}, allow_nan=False).encode() + b"\n"
        if len(payload) > MAX_MESSAGE:
            raise ValueError("Control request too large")
        self.file.write(payload)
        self.file.flush()
        line = self.file.readline(MAX_MESSAGE + 1)
        if not line or len(line) > MAX_MESSAGE:
            raise ConnectionError("Instrument coordinator disconnected or sent an oversized response")
        result = json.loads(line)
        if not result["ok"]:
            raise RemoteError(result["error"])
        try:
            return decode_result(result["result"], result["arrays"], self.file.read)
        except (KeyError, TypeError, ValueError, ConnectionError) as exc:
            self.close()  # never reuse a socket whose binary framing is uncertain
            raise ConnectionError(f"Invalid or incomplete instrument result: {exc}") from exc

    def status(self):
        return self.request("STATUS")

    def register(self, name, driver, resource):
        return self.request("REGISTER", name=name, driver=driver, resource=resource)

    def deregister(self, name):
        return self.request("DEREGISTER", name=name)

    def live(self, name):
        return Session(self, name, self.request("OPEN_LIVE", name=name)["session"], run=False)

    def begin_run(self, run_id):
        return Session(self, None, self.request("BEGIN_RUN", run_id=run_id)["session"], run=True)

    def reconcile_run(self, reason):
        return self.request("RECONCILE_RUN", reason=reason)

    def terminate(self, name, reason):
        return self.request("TERMINATE", name=name, reason=reason)

    def recover(self, name, reason=None):
        """Reconnect a quarantined instrument with an optional log note."""
        kwargs = {"name": name}
        if reason is not None:
            kwargs["reason"] = reason
        return self.request("RECOVER", **kwargs)

    def close(self):
        try:
            self.file.close()
        finally:
            self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Session:
    """One live instrument, or one run with access to the registered domain."""

    def __init__(self, client, name, token, *, run):
        self.client, self.name, self.token, self.run = client, name, token, run
        self.closed = False

    def _name(self, name):
        if self.closed:
            raise RuntimeError("Session is closed")
        if self.run:
            if name is None:
                raise ValueError("A run command requires an instrument name")
            return name
        if name is not None and name != self.name:
            raise ValueError("Live session is for one instrument only")
        return self.name

    def get(self, member, *, name=None):
        return self.client.request("GET", session=self.token, name=self._name(name), member=member)

    def set(self, member, value, *, name=None):
        return self.client.request("SET", session=self.token, name=self._name(name), member=member, value=value)

    def call(self, member, *args, name=None, **kwargs):
        return self.client.request("CALL", session=self.token, name=self._name(name),
                                   member=member, args=args, kwargs=kwargs)

    def close(self):
        if not self.closed:
            self.client.request("END_RUN" if self.run else "CLOSE_LIVE", session=self.token)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
