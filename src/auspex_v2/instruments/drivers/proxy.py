"""Driver-shaped, session-bound views of server-registered instruments."""


def _setting_property(member, readable, writable):
    """Expose a descriptor so completion engines need not evaluate a remote GET."""
    def get_setting(proxy):
        return proxy.get(member)

    def set_setting(proxy, value):
        proxy.set(member, value)

    return property(get_setting if readable else None, set_setting if writable else None)


class InstrumentProxy:
    """Expose declared settings and operations without constructing a local driver."""

    __slots__ = ("_session", "_name", "_settings", "_operations")

    def __init__(self, session, name, capabilities):
        settings = capabilities["settings"]
        operations = set(capabilities["operations"])
        helpers = {f"{verb}_{key}" for key, spec in settings.items()
                   for verb, enabled in (("get", spec["read"]), ("set", spec["write"])) if enabled}
        reserved = set(dir(type(self))) | {"session", "get", "set", "call", "close"}
        conflicts = (set(settings) | operations | helpers) & reserved
        conflicts |= operations & (set(settings) | helpers)
        conflicts |= set(settings) & helpers
        if conflicts or any(key.startswith("_") for key in set(settings) | operations | helpers):
            raise ValueError(f"Instrument proxy name collision: {sorted(conflicts)}")
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_operations", operations)
        # Jedi inspects dynamic attributes with getattr(), which would query hardware.
        # Real property descriptors let it list scalar settings without reading them.
        properties = {key: _setting_property(key, spec["read"], spec["write"])
                      for key, spec in settings.items() if not spec["args"]}
        if properties:
            # Keep the slot layout unchanged; only this instance gets these descriptors.
            view = type("InstrumentProxyView", (type(self),),
                        {"__slots__": (), "__module__": __name__, **properties})
            object.__setattr__(self, "__class__", view)

    def __dir__(self):
        """List declared members for completion without reading instrument settings."""
        names = set(dir(type(self))) | set(self._settings) | self._operations
        for name, spec in self._settings.items():
            if spec["read"]:
                names.add(f"get_{name}")
            if spec["write"]:
                names.add(f"set_{name}")
        return sorted(names)

    @property
    def session(self):
        """Access the underlying explicit GET/SET/CALL session API."""
        return self._session

    def _check(self):
        if self._session.closed:
            raise RuntimeError("Session is closed")

    def get(self, member, /, **kwargs):
        self._check()
        return self._session._dispatch("GET", self._name, member, kwargs=kwargs)

    def set(self, member, value, /, **kwargs):
        self._check()
        return self._session._dispatch("SET", self._name, member, value=value, kwargs=kwargs)

    def call(self, member, /, *args, **kwargs):
        self._check()
        return self._session._dispatch("CALL", self._name, member, args=args, kwargs=kwargs)

    def __getattr__(self, key):
        self._check()
        settings = self._settings
        if key in settings:
            if not settings[key]["read"]:
                raise AttributeError(f"{key} is not readable")
            if settings[key]["args"]:
                raise AttributeError(f"Use get_{key}(...) for parameterized settings")
            return self.get(key)
        for prefix, action in (("get_", "read"), ("set_", "write")):
            if key.startswith(prefix):
                member = key[len(prefix):]
                if member in settings and settings[member][action]:
                    expected = set(settings[member]["args"])

                    def helper(*args, **kwargs):
                        self._check()
                        if action == "write":
                            if len(args) != 1:
                                raise TypeError(f"{key} requires one value")
                            value = args[0]
                        elif args:
                            raise TypeError(f"{key} takes no positional arguments")
                        if set(kwargs) != expected:
                            raise TypeError(f"{key} requires keyword arguments {sorted(expected)}")
                        return self.set(member, value, **kwargs) if action == "write" else self.get(member, **kwargs)

                    return helper
        if key in self._operations:
            return lambda *args, **kwargs: self.call(key, *args, **kwargs)
        raise AttributeError(f"Not an exposed instrument member: {key}")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            raise AttributeError("Instrument proxy internals are read-only")
        self._check()
        spec = self._settings.get(key)
        if spec is None or not spec["write"]:
            raise AttributeError(f"Not a writable instrument setting: {key}")
        if spec["args"]:
            raise AttributeError(f"Use set_{key}(...) for parameterized settings")
        self.set(key, value)

    def close(self):
        self._session.close()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
