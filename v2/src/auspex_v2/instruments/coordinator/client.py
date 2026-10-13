"""Synchronous, local-only client for the instrument coordinator."""

import json
import socket

from .bulk import decode_result
from .proxy import InstrumentProxy
from .service import MAX_MESSAGE, PROTOCOL_VERSION, default_socket


class RemoteError(Exception):
    """Coordinator reported failure; a dispatched command may have changed hardware."""


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

    def connect(self, name):
        """Explicitly connect a registered instrument outside a run."""
        return self.request("CONNECT", name=name)

    def disconnect(self, name):
        return self.request("DISCONNECT", name=name)

    def disconnect_all(self):
        return self.request("DISCONNECT_ALL")

    def clear_instruments(self):
        return self.request("CLEAR_INSTRUMENTS")

    def live(self, name):
        opened = self.request("OPEN_LIVE", name=name)
        try:
            return Session(self, name, opened["session"], run=False,
                           capabilities=opened["capabilities"]).device
        except Exception:
            self.request("CLOSE_LIVE", session=opened["session"])
            raise

    def begin_run(self, run_id):
        return Session(self, None, self.request("BEGIN_RUN", run_id=run_id)["session"], run=True)

    def reconcile_run(self, reason):
        return self.request("RECONCILE_RUN", reason=reason)

    def terminate(self, name, reason=None):
        """Quarantine one instrument, optionally recording a console note."""
        kwargs = {"name": name}
        if reason is not None:
            kwargs["reason"] = reason
        return self.request("TERMINATE", **kwargs)

    def terminate_all(self):
        """Quarantine every registered instrument without releasing run ownership."""
        return self.request("TERMINATE_ALL")

    def recover(self, name, reason=None):
        """Acknowledge recovery and mark a quarantined instrument disconnected."""
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

    def __init__(self, client, name, token, *, run, capabilities=None):
        self.client, self.name, self.token, self.run = client, name, token, run
        self.closed = False
        self.device = InstrumentProxy(self, name, capabilities) if not run else None

    def connect(self, name):
        """Connect a registered instrument under this run's exclusive lease."""
        if not self.run:
            raise RuntimeError("Only run sessions can connect under a run lease")
        self._name(name)
        return self.client.request("CONNECT", name=name, session=self.token)

    def instrument(self, name):
        """Create a device view owned by this run, without opening another lease."""
        if not self.run:
            raise RuntimeError("Only run sessions select instruments")
        self._name(name)
        capabilities = self.client.request("DESCRIBE", session=self.token, name=name)
        return InstrumentProxy(self, name, capabilities)

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

    def get(self, member, *, name=None, **kwargs):
        return self.client.request("GET", session=self.token, name=self._name(name),
                                   member=member, kwargs=kwargs)

    def set(self, member, value, *, name=None, **kwargs):
        return self.client.request("SET", session=self.token, name=self._name(name),
                                   member=member, value=value, kwargs=kwargs)

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
