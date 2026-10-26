"""Small, hardware-independent instrument and command definitions.

A command remains a descriptor (rather than being replaced by a property), so
both direct Python access and server-facing metadata use one definition.
"""

from __future__ import annotations

import time
from types import MappingProxyType


class InstrumentError(Exception):
    """An instrument could not perform an operation."""


class Instrument:
    """One logical instrument; subclasses may use SCPI or a custom backend.

    Inject ``interface`` for tests or direct use with another transport. Nothing
    opens or probes hardware until ``connect`` is called.
    """

    KIND = None
    IDN_STRING = None  # Literal *IDN? substring; None for non-VISA drivers.
    commands = MappingProxyType({})
    properties = MappingProxyType({})
    operations = MappingProxyType({})

    def __init_subclass__(cls, **kwargs):
        """Collect inherited driver metadata and generate direct-use helpers."""
        super().__init_subclass__(**kwargs)
        commands = {}
        properties = {}
        operations = {}
        for base in reversed(cls.__mro__[1:-1]):
            commands.update(getattr(base, "commands", {}))
            properties.update(getattr(base, "properties", {}))
            operations.update(getattr(base, "operations", {}))
        for name, value in list(vars(cls).items()):
            if isinstance(value, Command):
                properties.pop(name, None)
                operations.pop(name, None)
                previous = commands.get(name)
                # An override also replaces inherited aliases: leaving an alias
                # pointing to the old command would operate different hardware.
                aliases = tuple(dict.fromkeys((*(alias for alias, cmd in commands.items()
                                               if alias != name and cmd is previous), *value.aliases)))
                commands[name] = value
                for alias in aliases:
                    if alias in commands and commands[alias] is not previous:
                        raise ValueError(f"Conflicting command alias: {alias}")
                    if alias not in commands and hasattr(cls, alias):
                        raise ValueError(f"Conflicting command alias: {alias}")
                    setattr(cls, alias, value)
                    commands[alias] = value
            elif isinstance(value, ManagedProperty):
                if name in commands:
                    raise TypeError(f"Override command {name} with another Command, not a property")
                properties[name] = value
            elif callable(value) and getattr(value, "_auspex_operation", False):
                operations[name] = value
            else:
                if name in commands:
                    raise TypeError(f"Override command {name} with another Command")
                properties.pop(name, None)
                operations.pop(name, None)
        for name, value in list(commands.items()):
            # Explicit definitions win over inherited declarations.
            if name in vars(cls) and not isinstance(vars(cls)[name], Command):
                del commands[name]
                continue
            if value.get_string is not None and f"get_{name}" not in vars(cls):
                setattr(cls, f"get_{name}", _getter(value))
            if value.set_string is not None and f"set_{name}" not in vars(cls):
                setattr(cls, f"set_{name}", _setter(value))
        cls.commands = MappingProxyType(commands)
        cls.properties = MappingProxyType(properties)
        cls.operations = MappingProxyType(operations)

    def __init__(self, resource_name=None, name=None, *, interface=None):
        self.resource_name = resource_name
        self.name = name or type(self).__name__
        self.interface = interface
        self._injected_interface = interface is not None
        self.connected = False

    @property
    def physical_resource(self):
        """Server identity; override when several channels share a device."""
        return self.resource_name

    def connect(self, resource_name=None):
        if resource_name is not None:
            self.resource_name = resource_name
        if self.connected:
            raise InstrumentError(f"{self.name} is already connected")
        if self.interface is None:
            raise InstrumentError(f"{self.name} has no transport; override connect()")
        self.connected = True

    def disconnect(self):
        try:
            if self.connected and self.interface is not None:
                self.interface.close()
        finally:
            self.connected = False
            if not self._injected_interface:
                self.interface = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def _transport(self):
        if not self.connected or self.interface is None:
            raise InstrumentError(f"{self.name} is not connected")
        return self.interface


class ManagedProperty(property):
    """Opt a handwritten property into server GET/SET discovery."""

    @property
    def readable(self):
        return self.fget is not None

    @property
    def writable(self):
        return self.fset is not None


def managed_property(getter):
    return ManagedProperty(getter)


def operation(func):
    """Expose a handwritten method as CALL, regardless of its return value.

    GET/SET apply only to declared commands and managed properties. A method
    may return data, trigger acquisition, or both; neither its name nor its
    return type implies a separate wire-level operation.
    """
    func._auspex_operation = True
    return func


def _getter(command):
    def get(self, **kwargs):
        return command.get(self, **kwargs)
    return get


def _setter(command):
    def set_(self, value, **kwargs):
        return command.set(self, value, **kwargs)
    return set_


class Command:
    """Typed SCPI property with optional explicit get/set strings.

    ``additional_args`` commands use generated get_*/set_* methods rather than
    property access. Declared ranges are checked before hardware I/O.
    """

    formatter = "{}"
    type = str

    def __init__(self, scpi_string=None, *, get_string=None, set_string=None,
                 formatter=None, value_map=None, value_range=None,
                 allowed_values=None, aliases=(), additional_args=(),
                 get_delay=0, set_delay=0, doc=""):
        """Validate and retain a command's SCPI strings and conversion rules."""
        if scpi_string is not None:
            if get_string is not None or set_string is not None:
                raise ValueError("Specify either scpi_string or explicit get/set strings")
            get_string = scpi_string + "?;"
            set_string = scpi_string + " " + (formatter or self.formatter)
        if get_string is None and set_string is None:
            raise ValueError("A getter or setter is required")
        if value_map is not None and (value_range is not None or allowed_values is not None):
            raise ValueError("value_map cannot be combined with range or allowed_values")
        if value_map is not None and len(set(value_map.values())) != len(value_map):
            raise ValueError("value_map values must be unique")
        self.get_string = get_string
        self.set_string = set_string
        self.value_map = value_map
        self.inverse_map = {str(v).strip(): k for k, v in value_map.items()} if value_map else None
        self.value_range = tuple(sorted(value_range)) if value_range is not None else None
        self.allowed_values = tuple(allowed_values) if allowed_values is not None else None
        self.aliases = tuple(aliases)
        self.additional_args = tuple(additional_args)
        self.get_delay = get_delay
        self.set_delay = set_delay
        self.__doc__ = doc

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.additional_args:
            raise AttributeError(f"Use get_{self.name}(...) for parameterized commands")
        if self.get_string is None:
            raise AttributeError(f"{self.name} is write-only")
        return self.get(instance)

    def __set__(self, instance, value):
        if self.additional_args:
            raise AttributeError(f"Use set_{self.name}(...) for parameterized commands")
        self.set(instance, value)

    def _check_args(self, kwargs):
        if set(kwargs) != set(self.additional_args):
            raise TypeError(f"Expected arguments {self.additional_args}, got {tuple(kwargs)}")

    def get(self, instrument, **kwargs):
        if self.get_string is None:
            raise AttributeError(f"{self.name} is write-only")
        self._check_args(kwargs)
        result = instrument._transport().query(self.get_string.format(**kwargs))
        if self.get_delay:
            time.sleep(self.get_delay)
        if self.inverse_map is not None:
            return self.inverse_map[str(result).strip()]
        return self.type(result)

    def set(self, instrument, value, **kwargs):
        """Validate and encode a value before writing it to the transport."""
        if self.set_string is None:
            raise AttributeError(f"{self.name} is read-only")
        self._check_args(kwargs)
        if self.value_range is not None and not self.value_range[0] <= value <= self.value_range[1]:
            raise ValueError(f"{self.name}: {value} outside {self.value_range}")
        if self.allowed_values is not None and value not in self.allowed_values:
            raise ValueError(f"{self.name}: {value} not in {self.allowed_values}")
        if self.value_map is not None and value not in self.value_map:
            raise ValueError(f"{self.name}: {value} not in {tuple(self.value_map)}")
        encoded = self.value_map[value] if self.value_map is not None else value
        instrument._transport().write(self.set_string.format(encoded, **kwargs))
        if self.set_delay:
            time.sleep(self.set_delay)


class StringCommand(Command):
    """Encode and decode textual instrument settings."""

    formatter = "{:s}"


class FloatCommand(Command):
    """Encode and decode floating-point instrument settings."""

    formatter = "{:E}"
    type = float


class IntCommand(Command):
    """Encode integer settings and parse numeric instrument readbacks."""

    formatter = "{:d}"
    type = staticmethod(lambda value: int(float(value)))


class BoolCommand(Command):
    """Read boolean settings with either an explicit map or standard tokens."""

    formatter = "{:s}"

    def get(self, instrument, **kwargs):
        if self.inverse_map is not None:
            return super().get(instrument, **kwargs)
        value = super().get(instrument, **kwargs).strip().upper()
        if value in ("1", "ON", "TRUE"):
            return True
        if value in ("0", "OFF", "FALSE"):
            return False
        raise ValueError(f"Unexpected boolean response: {value}")


class RampCommand(FloatCommand):
    """Synchronous ramp with dynamic start readback and a bounded increment."""

    def __init__(self, *, increment, pause=0, **kwargs):
        super().__init__(**kwargs)
        if increment <= 0 or pause < 0 or self.get_string is None:
            raise ValueError("Ramp needs positive increment, nonnegative pause and a getter")
        self.increment = increment
        self.pause = pause

    def set(self, instrument, value, **kwargs):
        import math
        if self.value_range is not None and not self.value_range[0] <= value <= self.value_range[1]:
            raise ValueError(f"{self.name}: {value} outside {self.value_range}")
        start = self.get(instrument, **kwargs)
        steps = max(1, math.ceil(abs(value - start) / self.increment))
        for index in range(1, steps + 1):
            super().set(instrument, start + (value - start) * index / steps, **kwargs)
            if index < steps and self.pause:
                time.sleep(self.pause)


class SCPIInstrument(Instrument):
    """Open a VISA transport lazily while preserving direct driver access."""

    def __init__(self, resource_name=None, name=None, *, interface=None, io_timeout=None):
        super().__init__(resource_name, name, interface=interface)
        if io_timeout is not None and (not isinstance(io_timeout, (int, float)) or io_timeout <= 0):
            raise ValueError("io_timeout must be positive seconds or None")
        self.io_timeout = io_timeout

    def connect(self, resource_name=None, *, visa_resource=None):
        if resource_name is not None:
            self.resource_name = resource_name
        if self.connected:
            raise InstrumentError(f"{self.name} is already connected")
        if not self.resource_name:
            raise ValueError("A VISA resource name is required")
        opened_here = self.interface is None
        try:
            if opened_here:
                from .visa import VisaTransport
                self.interface = VisaTransport(visa_resource or self.resource_name)
            if self.io_timeout is not None:
                self.interface.timeout = self.io_timeout * 1000
            super().connect()
        except BaseException:
            self.connected = False
            if self.interface is not None:
                try:
                    self.interface.close()
                except Exception:
                    pass  # Preserve the connection/configuration error.
                finally:
                    if opened_here:
                        self.interface = None
            raise
