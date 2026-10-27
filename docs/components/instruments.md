# Instruments

InstrumentServer keeps registration, connection health, and ownership separate.
It owns one spawned worker per connected device and serializes device commands.
The [server utility](../utilities/instrument-server.md) and optional
[GUI](../utilities/instrument-server-gui.md) expose operator actions.

## Roles and drivers

An experiment declares a fixed `KIND`, optionally inferred from an authoring driver.
An instance binds that role to an inventory name. The actual registered driver must
match KIND, but model-specific member support is still checked separately.

SimulatedSource and SimulatedVNA provide offline operation. AgilentN5183A,
AgilentE8363C, AgilentN5230A and YokogawaGS200 have been validated in the lab.
The following V1 VISA ports are **experimental**, not hardware-validated:

| Vendor | Admitted class names |
| --- | --- |
| Rohde & Schwarz | SMB100B |
| American Magnetics | AMI430 |
| BBN | RelayDriver, DigitalAttenuator |
| Berkeley Nucleonics | BNC845 |
| DS Instruments | DSInstrumentsSG12000, DSInstrumentsSG12000Pro, DAT64H |
| Keithley | Keithley2400 |
| Kepco | BOP2020M |
| Picosecond | Picosecond10070A |

Registration uses these canonical names. Implemented hardware-unvalidated classes
carry `@experimental`, setting inherited `EXPERIMENTAL = True` metadata. They
import and register normally, but emit a Python `UserWarning` on connection
(in the worker's output when server-managed). Standard warning filters apply;
the marker does not impose an admission gate or change GUI presentation.

The remaining 16 models are **unavailable stubs**, deferred to manual porting and
hardware validation. They can be imported only from their vendor modules under
`auspex_v2.instruments.drivers`; construction raises `NotImplementedError` before
any I/O. They are absent from module `__all__`, top-level exports and server
admission. Their docstrings identify V1 references and known review concerns.

| Module | Unavailable class stubs |
| --- | --- |
| `agilent` | HP33120A, Agilent33220A, Agilent33500B, Agilent34970A, AgilentN9010A |
| `bbn` | SpectrumAnalyzer |
| `keysight` | KeysightM8190A |
| `lakeshore` | LakeShore370, LakeShore335 |
| `lecroy` | HDO6104 |
| `stanford` | SR830, SR865 |
| `tektronix` | DPO72004C, DPO2024, DPO2014, RSA3308A |

Each bundled vendor/simulator module declares intended drivers in `__all__`.
Those classes are automatically exposed by `auspex_v2.instruments` (including
`from ... import` tab completion) and admitted by the server. There are no separate
package-level export or admission lists. Imports do not construct or connect
instruments, and do not require optional VISA backends.

To add a driver, see the [bundled export contract](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/instruments/drivers/instrument_drivers.md#exporting-a-bundled-driver).
Exporting is explicit service-admission intent: only concrete, locally defined
driver classes with valid `KIND` belong in the list. Helpers/base classes stay
unexported. Duplicate/invalid exports fail visibly; restart Python/services to
load newly installed drivers. No third-party plugin scan or hot reload is performed.

Drivers expose readable/writable settings and explicitly declared operations.
`GET`/`SET` address settings; `CALL` invokes an operation. A name such as `get_trace`
does not imply a side-effect-free query: acquisition may trigger a sweep.

Direct driver use remains possible without services, but bypasses server ownership.
Do not independently open a physical device already registered/in use elsewhere.

## Registration and connection

`register(name, driver, resource)` persists a disconnected definition without
constructing a worker or opening hardware. `connect(name)` starts its worker and
opens the resource. The experiment runner connects required disconnected devices
under its run lease; it does not register or recover them implicitly.

Closing a live lease leaves the device connected. Closing a GUI leaves the server
running. Normal service shutdown attempts disconnection but preserves inventory.

Real VISA operation uses PyVISA-py (`@py`), installed through the `visa` extra.
Addresses are normalized consistently for I/O and ownership. Endpoint normalization
does not prove physical identity across aliases or distinct interfaces.

### Explicit initialization for newly ported drivers

Connections configure transport/protocol only: no automatic output enabling,
reference selection, source-mode selection, field-unit change or trigger-source
change. Construction also performs no I/O. In particular:

- BNC845: call `configure_external_reference()` to select external 10 MHz and
  poll lock; set `output = True` separately when appropriate. A false lock result
  warns but does not fail. The model's pulse/mod/ALC settings intentionally encode
  `True` as wire `0`; RF output encodes `True` as `1`.
- BOP2020M: `configure_current_mode()` selects current mode and writes `VOLT MAX`,
  leaving output unchanged. Enable output separately. `shutdown()` issues ramps
  and an output-off command; it is not independent safety confirmation.
- AMI430: select `field_units = "T"` explicitly if Tesla is intended. V1's fixed
  current/field bounds remain; they do not identify or protect the attached magnet.
  `set_field()` blocks until HOLDING without a total timeout.
- Picosecond10070A: select `trigger_source = "GPIB"` explicitly if bus triggering
  is intended. Response headers are disabled on connection for parsing only.
- Keithley2400: `sweep()` explicitly configures reporting/trigger count, enables
  output, and disables it on success. It retains V1's twelve-hour read timeout;
  failures can leave output enabled. There is no automatic retry.

These are hardware ports, not hardware validation: automated checks cover only
imports, declarations/admission and the experimental marker, not device behavior.

## Interactive name completion

Live and run-session proxies support IPython/Jupyter name completion, for example
`vna.<TAB>`. Settings, available `get_`/`set_` helpers, and operations come from cached
capability metadata; listing names makes no server requests or hardware reads.
Reading `vna.frequency_center` still performs a GET. Completion of expressions
that themselves read values or call methods is outside this name-only guarantee.
Full signatures/docstring popups and static-IDE inference are not provided here.

## Discovery

Enumeration lists backend candidates without opening them. Explicit identification
opens a candidate, sends `*IDN?`, and closes it. Driver suggestions are advisory;
registration remains a separate operator choice and never connects automatically.

Serial, Prologix, and known permission-denied candidates are hidden by default.
Display flags and the server-reloaded whitelist can reveal them. Bulk probing
still excludes serial, Prologix, and registered resources. See the
[terminal workflow](../utilities/instrument-server.md#discovery).

## Ownership and recovery

A live session leases a connected instrument. An experiment run locks the **whole
registered domain**, not only the devices named by its plan. This protects
cooperating Auspex clients, not software connecting outside the server.

| State | Interpretation |
| --- | --- |
| `DISCONNECTED` | Definition exists; no connected worker |
| `CONNECTED` | Worker connection is available, subject to ownership |
| `UNKNOWN` | Connection outcome/state is uncertain |
| `TERMINATED` | Device is quarantined after failure or termination |

Connection/disconnection failures latch UNKNOWN; command failure or worker death
can latch TERMINATED. Termination kills software, not instrument output.
Recovery acknowledges operator inspection and returns to DISCONNECTED. It does not
read back hardware or reconnect.

After inspecting a lost run and the physical setup, an operator can reconcile an
orphaned lease explicitly:

```python
from auspex_v2.instruments.server import InstrumentServerClient

with InstrumentServerClient() as instruments:
    print(instruments.status())
    # Only after inspection, if this is the appropriate recovery action:
    # instruments.reconcile_run("Runner stopped; hardware inspected by operator")
    # instruments.recover("lab_source", reason="Inspected after runner loss")
```

Reconciliation releases ownership; recovery addresses an instrument's latched
state. Neither proves safety. Commands are not automatically retried, and restart
does not resume an experiment. `--clean` discards saved definitions/lockout; it is
not a physical recovery procedure.

## Timeouts and physical assumptions

`io_timeout` bounds a VISA I/O operation. `command_timeout` can bound a worker
request; its default is no whole-command deadline. Cancellation waits for an
in-flight command to return or reach its configured deadline.

The current real VNA drivers construct a linear axis from read-back endpoints and
point count. They do not query an arbitrary stimulus vector or establish that the
instrument is in linear-sweep mode. Review measurement routing and output policy
before using the real-hardware examples.

## Repository notes

- [Driver authoring](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/instruments/drivers/instrument_drivers.md)
- [InstrumentServer contracts](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/instruments/server/instrument_server.md)
- [Inventory GUI](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/instruments/server/instrument_server_gui.md)
