# Writing V2 instrument drivers (preliminary)

V2 instruments work directly from Python or through the local instrument server.
Each bundled driver module explicitly exports its intended classes; that one list
controls top-level Python imports and service admission by canonical class name.
The server owns each connection in a worker process. See `core.py`, `agilent.py`,
`yokogawa.py` and the offline tests in `tests/instruments/`.

## Exporting a bundled driver

Add a flat vendor module under `src/auspex_v2/instruments/drivers/`, for example
`bench.py`, and declare its intended drivers in `__all__`:

```python
from .core import SCPIInstrument, FloatCommand
from ..kinds import InstrumentKind

__all__ = ["BenchSource"]

class BenchSource(SCPIInstrument):
    """Example declarations only; review commands/limits for the actual device."""
    KIND = InstrumentKind.MICROWAVE_SOURCE
    frequency = FloatCommand(scpi_string=":FREQ", value_range=(1e6, 10e9))
```

No changes to either package initializer or `registry.py` are needed. On a fresh
import this makes `from auspex_v2.instruments import BenchSource` available,
including import-name tab completion. The same class is automatically admitted
by InstrumentServer and available to experiment driver lookup. Simulators follow
the same export contract. Restart Python/service processes after adding drivers;
there is no hot reload or hardware discovery during import.

Discovery visits non-private, non-package modules in the bundled driver package
in deterministic module-name order. A module without `__all__` exports no drivers.
Export lists must contain only locally defined, concrete `Instrument` subclasses
with valid `KIND`, using their canonical class names. Helpers, base classes and
aliases must not be listed. Duplicate names, public-API collisions, invalid
exports and broken imports fail visibly rather than silently producing a partial
catalog. This is not a third-party plugin/entry-point loader.

**Exporting is service-admission intent**, not merely Python visibility. Driver
modules must use relative imports from `.core`/`..kinds` while the packages are
initializing. They must not construct instruments, probe hardware, or import
optional backend libraries eagerly. Put backend imports and I/O in `connect()`.
The catalog is built without constructing any driver. Unexported helpers remain
accessible through their module but are not top-level drivers or service choices.

## Hardware-validation designation

Mark newly ported, hardware-unvalidated drivers with `@experimental`:

```python
from auspex_v2.instruments import SCPIInstrument, experimental

@experimental
class UnvalidatedSource(SCPIInstrument):
    """Driver awaiting attended hardware validation."""
```

The decorator preserves class identity and sets `Driver.EXPERIMENTAL = True`;
subclasses inherit the designation. Imports, construction, capability inspection
and registration remain silent. Normal connection emits a Python `UserWarning`
that the named driver has not yet been validated with hardware (in the owning
worker's output when server-managed). Standard Python warning filters apply.
The marker does not change admission, commands or GUI presentation. Remove the
decorator after attended validation; absence of the marker is not a safety claim.
Existing lab-validated drivers retain their behavior. The currently ported models
and explicit initialization differences are listed in the
[main instrument guide](../../../../docs/components/instruments.md).
The remaining inventory is tracked in `.agents/PLAN.md`.

The sixteen remaining V1 models are deliberately unavailable stubs, not
experimental implementations. They remain accessible only in their vendor modules,
raise `NotImplementedError` on construction and are excluded from module `__all__`
and server admission. Manual porting and hardware review must precede enabling them.

For these V1 ports, automated checks cover imports, exports/admission and marker
behavior only, not hardware command correctness. No mocked instrument tests are
used; command/data/connection validation awaits attended hardware work.

## The model

`InstrumentKind` in `auspex_v2.instruments.kinds` is the fixed role vocabulary used
by plans and services: `MICROWAVE_SOURCE`, `DC_SOURCE`, `AWG`, `DIGITIZER`, `VNA`,
`SPECTRUM_ANALYZER`, `OSCILLOSCOPE`, `FUNCTION_GENERATOR`, `PULSE_GENERATOR`,
`LOCKIN`, `ATTENUATOR`, `SWITCH`, `MAGNET_SUPPLY`, `TEMPERATURE_CONTROLLER`, `DAQ`,
and `SENSOR`. Every admitted driver declares `KIND`; models sharing a role can fill
the same experiment role. Unsupported members still fail at command dispatch.

An `Instrument` represents a logical instrument. Construction and import must **not** open, enumerate, or reset hardware. `connect()` opens the connection; `disconnect()` closes it. Drivers can be used with `with`, which connects and disconnects automatically. A driver has a `resource_name` and a `physical_resource` identity; override the latter when several logical channels share one physical device. Do not open independent workers for channels that share a USB connection.

### VISA addresses and identity

`auspex_v2.instruments.drivers.addresses.normalize_visa_address()` is a pure,
PyVISA-independent function shared by connections, registration and discovery.
It standardizes common TCPIP, GPIB, USB and serial spellings without probing or
DNS lookup. Case-sensitive serial paths, USB serial numbers and LAN device names
are preserved; unknown aliases are left untouched, not guessed.

`SCPIInstrument.physical_resource` returns this canonical connection endpoint.
`SCPIInstrument.connect()` opens that same endpoint. A driver supporting bare
IPv4 shorthand declares `VISA_IPV4_DEVICE`: `"5025::SOCKET"` for N5183A, or
`"hpib7,16::INSTR"` for the current VNA drivers. Do not expand addresses separately
inside a driver override of `connect()`. A bare IP without a driver-specific
default is rejected; explicit addresses never inherit that default.

```python
from auspex_v2.instruments.drivers.addresses import normalize_visa_address

normalize_visa_address("gpib::05::00::instr")  # GPIB0::5::INSTR
normalize_visa_address("192.0.2.5", ipv4_device="5025::SOCKET")
# TCPIP0::192.0.2.5::5025::SOCKET
```

This is endpoint identity, not a hardware fingerprint: different endpoints on the
same IP are not automatically treated as the same physical instrument. Custom
multi-channel drivers must still declare their shared physical ownership identity.

The server uses three instrument operations:

| Protocol | Driver declaration | Direct Python usage |
| --- | --- | --- |
| `GET name` | Readable `Command` or `@managed_property` | `instrument.name` or `instrument.get_name()` |
| `SET name, value` | Writable `Command` or `@managed_property` | `instrument.name = value` or `instrument.set_name(value)` |
| `CALL name, args` | Method marked `@operation` | `instrument.name(...)` |

`CALL` may return data, trigger acquisition, or both: a method called `get_trace()` can change the device's state. **Names and return types do not imply protocol operations.** Generated `get_name()` and `set_name()` methods are direct-Python conveniences, not extra `CALL`s. Undecorated methods are not registered for server calls. Current metadata is available at `Driver.commands`, `Driver.properties`, and `Driver.operations`. This is discovery metadata, **not yet** a complete argument schema or security policy.

### A small SCPI driver

```python
from auspex_v2.instruments import SCPIInstrument, FloatCommand, BoolCommand

class BenchSource(SCPIInstrument):
    frequency = FloatCommand(scpi_string=":FREQ", value_range=(1e6, 10e9))
    power = FloatCommand(scpi_string=":POW", value_range=(-80.0, 10.0))
    output = BoolCommand(scpi_string=":OUTP", value_map={False: "0", True: "1"})

# In a lab with an appropriate VISA backend installed:
with BenchSource("TCPIP0::192.168.1.20::5025::SOCKET") as source:
    source.frequency = 5e9             # formats and sends a SCPI SET
    print(source.frequency)            # sends a SCPI query
    source.output = True
```

`scpi_string=":FREQ"` generates `:FREQ?;` for reads and `:FREQ <formatted value>` for writes, matching the V1 command convention. Explicit `get_string=` and/or `set_string=` define one-sided commands. Use `FloatCommand`, `IntCommand`, `StringCommand`, or `BoolCommand`; declare maps/ranges where appropriate. In particular, supply a `value_map` for a **writable** `BoolCommand` so its wire encoding is unambiguous. `RampCommand` performs synchronous stepped writes after reading the starting value; review its units, bounds, and pause for the actual hardware.

Aliases and parameterized commands are supported. For example, `aliases=("value",)` exposes the same command by two names; `additional_args=("channel",)` gives you `get_frequency(channel=2)` / `set_frequency(value, channel=2)` rather than a property. Keep aliases and overrides unambiguous. Declarations are descriptors and remain inspectable on the class; they are not replaced by generated properties as in V1.

PyVISA is optional and imported only when connecting a real SCPI resource. Install with `python -m pip install -e '.[visa]'` from the repository root. V2 consistently uses the PyVISA-py `@py` backend. `io_timeout=` is expressed in seconds and overrides the driver's per-I/O default. Driver-specific VISA address normalization, termination and chunk size belong in `connect()`; call `super().connect(...)` and disconnect if subsequent configuration fails. **Do not** convert connection errors into a mock or claim to be connected when hardware initialization failed.

### Current VNA acquisition assumptions

`AgilentE8363C` and `AgilentN5230A.get_trace()` return `(frequencies, complex_values)`.
The frequency axis is **linear spacing between read-back start/stop endpoints**,
using the read-back point count, not a queried stimulus vector. Configure linear
sweeps; logarithmic/segmented axes are not supported or automatically detected.
This deliberately retains the V1 laboratory convention.

`reaverage()` retains V1's single/group sweep trigger, `*OPC`, and ESR polling.
It first validates the optional timeout and reads `*ESR?` to discard stale events.
That read clears all ESR event bits, not just operation-complete; it does not issue
`*CLS`. Physical behavior still requires model-specific bench validation.

### Custom hardware and commands

A non-SCPI driver can inherit `Instrument`, implement `connect()`/`disconnect()` and a guarded transport, and expose handwritten state and methods. For example:

```python
from auspex_v2.instruments import Instrument, managed_property, operation

class TwoChannelSource(Instrument):
    @property
    def physical_resource(self):
        return self.resource_name.rsplit("-", 1)[0]  # channel 1/2 share one device

    @managed_property
    def frequency(self):
        return self._transport().read_frequency(self.resource_name)

    @frequency.setter
    def frequency(self, value):
        self._transport().set_frequency(self.resource_name, value)

    @operation
    def identify(self):
        return self._transport().identify(self.resource_name)
```

This illustrates the registration API, **not** a complete driver: the subclass must open its backend in `connect()`, close it in `disconnect()`, and provide `read_frequency`/`set_frequency`/`identify`. For devices like Holzworth, discover hardware and read dynamic limits **at connection time**, then validate settings against those limits in the worker. Backend libraries should be imported lazily; an unavailable USB/DLL backend must produce a clear exception, not a successful-looking dummy instrument.

## Testing without hardware

Inject an interface implementing `write`, `query`, and `close` (plus `query_binary_values` for a VNA). Drive commands against a fake transcript and assert **both** emitted strings and converted results. Test validation before I/O, failed connections, resource identity, and binary shape/endianness where relevant. `tests/instruments/test_instruments.py` contains concrete examples and runs with:

```sh
python -m unittest discover -s tests -v
```

These tests never touch lab hardware. Real-device smoke tests should be separate, opt-in, and reviewed for output and ramp safety. Direct-use drivers intentionally do not participate in server lockout. Server-managed use is available for the explicitly exported bundled drivers;
real-device tests remain opt-in lab work.
