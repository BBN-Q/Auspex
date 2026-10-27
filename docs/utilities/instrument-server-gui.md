# Instrument server GUI

`instrument_server_gui.py` attaches an inventory window to an already running
InstrumentServer. It does not start or own the service.

```sh
python -m pip install -e ".[gui]"
python utils/instrument_server_gui.py
python utils/instrument_server_gui.py --socket /private/path/instruments.sock
```

Alternatively, `instrument_server.py --gui` or `launch_auspex.py --gui` starts a
service and a separate GUI client. Closing the window leaves the service running.

## Read the display

Connection state and ownership are separate columns. CONNECTED is not necessarily
available: live sessions and run ownership can block commands. The domain banner
also shows orphaned runs. “NO RUN LOCK” does not mean every device is free or that
physical hardware is safe. A failed status poll labels connectivity unknown rather
than presenting old status as current.

Connect/Disconnect act on worker connections. Recover acknowledges inspection and
returns to DISCONNECTED. Terminate quarantines the registration and kills its
worker if present; it does not switch hardware off. Per-device Recover and
Terminate have no extra confirmation dialog. Clear requires confirmation and a
disconnected, unowned device.

Global actions disconnect all, terminate all, or clear disconnected inventory.
Bulk termination and clearing require confirmation. The server revalidates every
request even when the GUI last showed an enabled action.

## Add instruments

Opening **Add instruments…** performs no discovery. Use **Enumerate VISA**, then
optionally **Identify** a candidate. Choose an admitted driver explicitly and
register a name. Identification suggestions can be overridden; registration does
not connect. **Identify all** confirms first and excludes serial, Prologix, and
registered devices.

Display checkboxes reveal serial/Prologix resources and known permission denials.
Whitelisting changes visibility only. Connect the new registration from the main
inventory when ready. A saved-registration error popup reports rejected definitions
without repeatedly reopening unchanged errors.

The GUI is not an instrument settings panel, measurement client, or plotter.
Identification reads `*IDN?`; no other acquisition/settings access is exposed.
See the retained [GUI notes](https://github.com/BBN-Q/Auspex/blob/auspex-v2/src/auspex_v2/instruments/server/instrument_server_gui.md).
