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

    def __init__(self, socket_path=None, *, label=None):
        """Open a local connection; label is an optional human hint, not authentication."""
        if label is not None and (not isinstance(label, str) or not label.strip() or
                                  len(label) > 100 or not label.isprintable()):
            raise ValueError("Owner label must be 1–100 printable characters")
        self.label = label
        self._live = {}  # Keep notebook leases recoverable even without an assigned handle.
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

    def instruments(self, state=None):
        """Return a fresh name-keyed inventory snapshot, optionally filtered by connection state.

        This queries server metadata, not hardware. CONNECTED does not imply unowned.
        """
        if state is not None and state not in ("CONNECTED", "DISCONNECTED", "UNKNOWN", "TERMINATED"):
            raise ValueError("Unknown instrument state")
        return {name: spec for name, spec in sorted(self.status()["instruments"].items())
                if state is None or spec["state"] == state}

    def show_instruments(self):
        """Print a compact notebook-friendly table without optional display dependencies."""
        rows = [("Name", "Driver", "State", "Owner")]
        for name, spec in self.instruments().items():
            owner = spec.get("live_owner")
            text = (f"{owner['label']} [{owner['id']}]" if owner and owner["label"] else
                    f"[{owner['id']}]" if owner else "free")
            if owner and owner["revoking"]:
                text += " (releasing)"
            if spec["locked"]:
                text = "run locked"
            rows.append((name, spec["driver"], spec["state"], text))
        widths = [max(len(row[col]) for row in rows) for col in range(4)]
        print("\n".join("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
                        for row in rows))

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
        """Acquire or recover this client's live handle; never connect hardware implicitly."""
        if name in self._live:
            self.live_sessions()  # Notice operator revocation before returning a cached handle.
            if name in self._live:
                return self._live[name].device
        opened = self.request("OPEN_LIVE", name=name, label=self.label)
        try:
            session = Session(self, name, opened["session"], run=False,
                              capabilities=opened["capabilities"])
            self._live[name] = session
            return session.device
        except Exception:
            self.request("CLOSE_LIVE", session=opened["session"])
            raise

    def live_sessions(self):
        """Return this connection's live names, short IDs, labels and revocation state."""
        owned = self.request("LIVE_SESSIONS")
        for name, session in list(self._live.items()):
            info = owned.get(name)
            if info is None or info["revoking"] or info["id"] != session.token[:7]:
                session.closed = True
                del self._live[name]
        return owned

    def release(self, name):
        """Release this client's live ownership; leave the device connected. Idempotent."""
        session = self._live.get(name)
        if session is not None:
            session.close()

    def release_all(self):
        """Release all this client's live handles, including group-owned handles."""
        errors = []
        for session in list(self._live.values()):
            try:
                session.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Could not release all live sessions", errors)

    def live_group(self, *names):
        """Acquire independent live leases together; roll back newly acquired leases on failure.

        Existing client leases and duplicate names are rejected, so closing a group
        cannot unexpectedly release a pre-existing handle. This is not a run lease.
        """
        if not names or len(set(names)) != len(names):
            raise ValueError("A live group requires distinct instrument names")
        owned = self.live_sessions()
        if any(name in owned for name in names):
            raise ValueError("Live group instruments must not already be leased by this client")
        sessions = {}
        try:
            for name in names:
                self.live(name)
                sessions[name] = self._live[name]
        except Exception:
            LiveGroup(sessions).close()
            raise
        return LiveGroup(sessions)

    def revoke_live(self, name, *, session_id=None):
        """Operator revocation, including another client's lease; caller confirms first.

        A pending result means access is revoked but ownership remains until the
        current command finishes. Run ownership is never released by this action.
        """
        return self.request("RELEASE_LIVE", name=name, session_id=session_id)

    def begin_run(self, run_id, *, name=None):
        """Claim the run lease; an optional allocated folder stem is for operator display only."""
        return Session(self, None, self.request("BEGIN_RUN", run_id=run_id, name=name)["session"], run=True)

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
        """Close transport; server disconnect cleanup releases this client's live leases."""
        try:
            self.file.close()
        finally:
            self.socket.close()
            for session in self._live.values():
                session.closed = True
            self._live.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class LiveGroup:
    """Named device handles sharing explicit cleanup, not atomic acquisition or execution."""

    def __init__(self, sessions):
        self._sessions = dict(sessions)

    def __getitem__(self, name):
        session = self._sessions[name]
        session._name(name)
        return session.device

    def close(self):
        """Attempt every release, including after individual handles were already closed."""
        errors = []
        for session in self._sessions.values():
            try:
                session.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Could not close live group", errors)

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
        try:
            return self.client.request(operation, **fields)
        except RemoteError as exc:
            if not self.run and "Live session closed or revoked" in str(exc):
                self.closed = True
                if self.client._live.get(self.name) is self:
                    del self.client._live[self.name]
            raise

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
            if not self.run and self.client._live.get(self.name) is self:
                del self.client._live[self.name]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
