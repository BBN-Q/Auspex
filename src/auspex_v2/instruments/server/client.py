"""Synchronous, local-only client for the instrument server."""

import json
import socket

from .bulk import decode_result
from .proxy import InstrumentProxy
from .service import MAX_MESSAGE, PROTOCOL_VERSION, default_socket


class RemoteError(Exception):
    """Server reported failure; a dispatched command may have changed hardware."""


class InstrumentServerClient:
    """Connect notebooks or scripts to the persistent local instrument service."""

    def __init__(self, socket_path=None):
        """Open the Unix socket and verify the server protocol version."""
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Whole-command deadlines belong to each registered instrument. Do not
        # impose a shorter hidden client transport timeout.
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
            raise ConnectionError("Instrument server disconnected or sent an oversized response")
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

    def enumerate(self, interface="visa", *, include_metadata=False):
        """List unfiltered candidates; include_metadata adds whitelist path/warnings in an envelope."""
        return self.request("ENUMERATE", interface=interface, include_metadata=include_metadata)

    def identify(self, resource, interface="visa"):
        """Explicitly probe an unregistered address with *IDN?; failures are reported as data."""
        return self.request("IDENTIFY", interface=interface, resource=resource)

    def register(self, name, driver, resource, *, command_timeout=None, io_timeout=None):
        return self.request("REGISTER", name=name, driver=driver, resource=resource,
                            command_timeout=command_timeout, io_timeout=io_timeout)

    def set_command_timeout(self, name, timeout):
        """Set a whole-worker-request deadline in seconds, or None for no deadline."""
        return self.request("SET_COMMAND_TIMEOUT", name=name, timeout=timeout)

    def set_io_timeout(self, name, timeout):
        """Set per-VISA-I/O seconds while disconnected; None restores driver default."""
        return self.request("SET_IO_TIMEOUT", name=name, timeout=timeout)

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

    def _dispatch(self, operation, instrument, member, /, *, args=(), kwargs=None, value=None):
        """Proxy backend: driver arguments never share the routing namespace."""
        fields = {"session": self.token, "name": self._name(instrument),
                  "member": member, "kwargs": kwargs or {}}
        if operation == "SET":
            fields["value"] = value
        elif operation == "CALL":
            fields["args"] = args
        elif operation != "GET":
            raise ValueError(f"Unknown instrument operation: {operation}")
        return self.client.request(operation, **fields)

    def get(self, member, *, name=None, **kwargs):
        return self._dispatch("GET", name, member, kwargs=kwargs)

    def set(self, member, value, *, name=None, **kwargs):
        return self._dispatch("SET", name, member, value=value, kwargs=kwargs)

    def call(self, member, *args, name=None, **kwargs):
        return self._dispatch("CALL", name, member, args=args, kwargs=kwargs)

    def close(self):
        if not self.closed:
            self.client.request("END_RUN" if self.run else "CLOSE_LIVE", session=self.token)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
