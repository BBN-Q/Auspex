# Auspex V2

An independent Python 3.13 package. V1 remains in `../src/auspex/`; see `../PLAN.md` for the broader design. The distribution name is `auspex-v2` and the provisional import name is `auspex_v2`.

## V2 data storage

The versioned `.auspex` dataset store is documented in [data.md](src/auspex_v2/data/data.md). The standalone read-only [DataBrowser](src/auspex_v2/data/data_browser.md) watches multiple OS folders for `.auspex` runs (or opens individual runs), plots published 1D datasets with pyqtgraph, and shows run/dataset metadata; higher-dimensional and ragged datasets are listed but not plotted. Format version 2 writes self-describing `.dat` files with values, actual coordinates (including dependent/ragged grids), names and units; old V2 format 1 is not read. An optional run settings snapshot can be supplied, but plans, settings-change events and reports are not yet implemented.

## Filter pipeline

The new [pipeline construction API](src/auspex_v2/filters/definitions.md) provides reusable, inert V1-style port wiring, explicit named inputs, filter/writer/buffer definitions and detached JSON compilation. It works with `ManualPipeline` without an experiment or server. Start with `v2/examples/pipeline_definition.py`. Single-ingress execution is supported; multi-input definitions and empty-source bindings are API-only for now.

See the [filter guide](src/auspex_v2/filters/README.md) for the reference DAG executor, dedicated-thread multi-worker Filter Server, explicit scan boundaries, one process-backed store owner for multiple required writer sinks, bounded overlapping execution and capacity policies, simulated VNA smoke tests, optional [read-only local graph viewer](src/auspex_v2/filters/inspection.md), and remaining measurement-plotting/whole-scan filter limits.

## Instruments (first slice)

See the preliminary [driver-authoring guide](src/auspex_v2/instruments/drivers/instrument_drivers.md) and [local instrument server guide](src/auspex_v2/instruments/server/instrument_server.md). The server is currently simulated-device-only; it does not control the real drivers below. A separate optional [pyqtgraph/PySide6 operator GUI](src/auspex_v2/instruments/server/instrument_server_gui.md) shows status and worker termination only; it cannot access instrument data or settings.

For ordinary acquisition scripts and notebooks, [standalone storage and manual pipelines](src/auspex_v2/filters/manual.md) provide direct `AuspexStore` writing and whole-window processing into file writers or bounded, queryable buffer sinks—without instrument or experiment services. A runnable synthetic example is `v2/examples/manual_data.py`.

The [experiment compiler](src/auspex_v2/experiment/experiment.md) supports symbolic sequences, fixed sweeps, explicit `OutputConnector`/`seq.publish`, internal or external pipeline attachment, and detached version-2 saved execution units. An offline `PipelineExecutor` feeds real filter workers/storage from injected command results, using virtual waits. See `v2/examples/compiled_publication.py`. Persistent ExperimentServer and supervised live execution are not implemented yet.

The instrument API is usable directly, without an experiment or server. The first ports are `AgilentN5183A`, `YokogawaGS200`, `AgilentE8363C`, and `AgilentN5230A`. These are **targeted ports**, not a complete migration of every V1 driver method. The VNA `get_trace()` returns actual instrument-readback frequencies and complex S-parameter values. Connecting does not select VNA measurements or enable RF output. Select a measurement explicitly with `vna.set_measurements(["S21"])` if none is active. `get_trace()` triggers a sweep by default and returns to continuous sweep mode afterward; `restart_measurement=False` reads the currently available trace without triggering.

```python
from auspex_v2.instruments import AgilentN5183A

with AgilentN5183A("TCPIP0::192.168.1.20::5025::SOCKET") as source:
    source.frequency = 5e9
    print(source.frequency)
```

Real VISA hardware requires `python -m pip install -e './v2[visa]'` from the repository root (or `python -m pip install -e '.[visa]'` here). PyVISA is imported only when opening a real resource. Choose an appropriate VISA backend for your lab; for PyVISA-py, pass `visa_backend="@py"`. Importing a driver never probes hardware. Commands raise on connection/transport failure; they do not silently fall back to a mock. The server uses **GET**, **SET**, and **CALL**: declared commands/managed properties support GET/SET, while methods explicitly decorated with `@operation` support CALL, including methods that return data. Live sessions yield a driver-shaped proxy (`source.set_frequency(5e9)`, `source.frequency`, `source.identify()`); runs use `run.instrument(name)`. The explicit API remains available via `source.session`. Declared parameterized commands support GET/SET keyword arguments.

Run the offline tests with `python -m unittest discover -s v2/tests -v` from the repository root. They use injected fake transports and **never** access hardware. Real-device smoke tests are not implemented yet; check settings and RF safety before manually connecting.
