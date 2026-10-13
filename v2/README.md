# Auspex V2

An independent Python 3.13 package. V1 remains in `../src/auspex/`; see `../PLAN.md` for the broader design. The distribution name is `auspex-v2` and the provisional import name is `auspex_v2`.

## Instruments (first slice)

See the preliminary [driver-authoring guide](src/auspex_v2/instruments/drivers/instrument_drivers.md) and [local instrument coordinator guide](src/auspex_v2/instruments/coordinator/instrument_coordinator.md). The coordinator is currently simulated-device-only; it does not control the real drivers below. A separate optional [pyqtgraph/PySide6 operator GUI](src/auspex_v2/instruments/coordinator/instrument_coordinator_gui.md) shows status and worker termination only; it cannot access instrument data or settings.

The instrument API is usable directly, without an experiment or coordinator. The first ports are `AgilentN5183A`, `YokogawaGS200`, `AgilentE8363C`, and `AgilentN5230A`. These are **targeted ports**, not a complete migration of every V1 driver method. The VNA `get_trace()` returns actual instrument-readback frequencies and complex S-parameter values. Connecting does not select VNA measurements or enable RF output. Select a measurement explicitly with `vna.set_measurements(["S21"])` if none is active. `get_trace()` triggers a sweep by default and returns to continuous sweep mode afterward; `restart_measurement=False` reads the currently available trace without triggering.

```python
from auspex_v2.instruments import AgilentN5183A

with AgilentN5183A("TCPIP0::192.168.1.20::5025::SOCKET") as source:
    source.frequency = 5e9
    print(source.frequency)
```

Real VISA hardware requires `python -m pip install -e './v2[visa]'` from the repository root (or `python -m pip install -e '.[visa]'` here). PyVISA is imported only when opening a real resource. Choose an appropriate VISA backend for your lab; for PyVISA-py, pass `visa_backend="@py"`. Importing a driver never probes hardware. Commands raise on connection/transport failure; they do not silently fall back to a mock. The coordinator uses **GET**, **SET**, and **CALL**: declared commands/managed properties support GET/SET, while methods explicitly decorated with `@operation` support CALL, including methods that return data. Live sessions yield a driver-shaped proxy (`source.set_frequency(5e9)`, `source.frequency`, `source.identify()`); runs use `run.instrument(name)`. The explicit API remains available via `source.session`. Declared parameterized commands support GET/SET keyword arguments.

Run the offline tests with `python -m unittest discover -s v2/tests -v` from the repository root. They use injected fake transports and **never** access hardware. Real-device smoke tests are not implemented yet; check settings and RF safety before manually connecting.
