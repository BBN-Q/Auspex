# Instruments

InstrumentServer keeps registration, connection health, and ownership separate.
It owns one spawned worker per connected device and serializes device commands.
The [server utility](../utilities/instrument-server.md) and optional
[GUI](../utilities/instrument-server-gui.md) expose operator actions.

## Roles and drivers

An experiment declares a fixed `KIND`, optionally inferred from an authoring driver.
An instance binds that role to an inventory name. The actual registered driver must
match KIND, but model-specific member support is still checked separately.

The current admitted drivers are SimulatedSource, SimulatedVNA, AgilentN5183A,
AgilentE8363C, AgilentN5230A, and YokogawaGS200. Real ports are targeted subsets of
V1, not a complete driver migration. Registration uses these canonical class names.

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
