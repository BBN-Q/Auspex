# Instrument server

`instrument_server.py` starts only InstrumentServer. Use it when managing services
separately; do not also start the combined launcher on the same endpoints.

```sh
python utils/instrument_server.py
python utils/instrument_server.py --gui
python utils/instrument_server.py --socket /private/path/instruments.sock
```

`-q` / `--quiet` suppresses command/state output. `--no-splash` hides the banner.
The optional GUI requires `gui`; VISA discovery/hardware requires `visa`.

## Terminal actions

An interactive terminal accepts:

```text
list
connect 1
disconnect 1
recover 1
terminate 1
clear 1
help
```

Numeric IDs refer to the most recent `list` snapshot, not permanent identities.
`connect`, `disconnect`, `terminate`, and `clear` accept `all` where appropriate;
there is no bulk recovery. Inspect states before acting. Clear, terminate-all,
and standalone `shutdown` require confirmation.

`shutdown` disconnects devices, clears inventory, and exits only on success.
Ctrl+C instead stops the service while retaining saved registrations. Neither is
an emergency-off mechanism. `-c` / `--clean` forcibly discards saved inventory and
run lockout without touching hardware; it is not normal recovery.

## Discovery

```text
enumerate visa
enumerate visa --serial --prologix --show-permission-denied
identify b
identify all
register b AgilentN5230A
```

Letter IDs belong to discovered candidates and remain stable until re-enumeration.
Enumeration opens no resources. Identification explicitly sends `*IDN?` with bounded
I/O waits; it is not a universal compatibility test. Bulk identification confirms
first and excludes serial, Prologix, and registered devices.

Driver suggestions are advisory. Registration prompts for an inventory name and
leaves the device disconnected. Serial/Prologix/known access-denied candidates are
hidden by default; the flags or whitelist control visibility, not ownership or
bulk-probing permission. The whitelist uses complete VISA addresses, one per line,
ignoring blank lines and `#` comments. The server reloads it on enumeration.

For manual addresses and simulators, use the Python registration client shown in
the [quickstart](../quickstart.md#3-register-the-simulators).

## Recovery

Unknown or terminated states require operator inspection. `recover` returns a
device to DISCONNECTED; it does not reconnect or read back safe conditions. A lost
run may first need explicit reconciliation through the client API; see
[ownership and recovery](../components/instruments.md#ownership-and-recovery).
Saved duplicate-endpoint definitions are all rejected rather than choosing a
winner; retained definitions/errors remain available for correction.

[Source notes](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/instruments/server/instrument_server.md)
contain discovery filters, whitelist details, timeouts, and restart behavior.
