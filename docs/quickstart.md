# Quickstart: a simulated measurement

This procedure runs the [annotated experiment](annotated-experiment.md) using only
SimulatedSource and SimulatedVNA. No VISA backend, GUI, or laboratory hardware is
needed. Run commands from the repository root.

## 1. Install

Use Python 3.13 or newer in a virtual environment:

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Optional extras are independent:

```sh
python -m pip install -e ".[visa]"         # Real VISA resources
python -m pip install -e ".[gui]"          # Desktop inspection tools
python -m pip install -e ".[tui]"          # Full-screen terminal console
python -m pip install -e ".[visa,gui,tui]" # All three
```

System permissions or interface libraries may also be needed for real hardware.
The simulation below does not require them.

## 2. Start the services

In **terminal 1**, activate the same environment, then:

```sh
export AUSPEX_HOME="$PWD/.auspex-demo"
mkdir -p "$AUSPEX_HOME"
python utils/launch_auspex.py
```

Keep the launcher running. It starts InstrumentServer, then ExperimentServer.
The isolated `AUSPEX_HOME` keeps this demo's saved inventory and sockets separate
from the normal home-directory defaults. Use this same setting in every terminal
and notebook involved in the demo. Do not register real devices in this inventory.

If desired, use `--tui` after installing the `tui` extra. It changes the terminal
interface, not the execution model.

## 3. Register the simulators

In **terminal 2**, activate the environment and repeat the demo path setting:

```sh
. .venv/bin/activate
export AUSPEX_HOME="$PWD/.auspex-demo"
mkdir -p "$AUSPEX_HOME" data/docs-demo
python - <<'PY'
from auspex_v2.instruments.server import InstrumentServerClient

with InstrumentServerClient() as instruments:
    instruments.register("lab_source", "SimulatedSource", "SIM::SOURCE")
    instruments.register("lab_vna", "SimulatedVNA", "SIM::VNA")
PY
```

Register once per demo inventory. A duplicate-name error on a later attempt means
the saved inventory may already contain these names; inspect it rather than using
`--clean` reflexively. Registration does not connect instruments. The runner
connects its required devices under an exclusive lease when the run starts.

## 4. Compile and submit

```sh
python examples/documented_vna_sweep.py "$PWD/data/docs-demo"
```

The script prints its plan, run ID, and allocated directory, then exits. Expect
warnings for the recorded NumPy calls, including:

> This compiled experiment executes runtime Python code. Auspex cannot guarantee repeatability

The services continue execution after the script exits. The run sweeps three
source frequencies and acquires one 201-point complex S21 trace per point. The
simulators are independent: changing the simulated source does not change the
synthetic VNA response.

## 5. Check completion

```sh
python utils/auspex_status.py
```

Reconnect to the run using the ID printed at submission:

```python
from auspex_v2.experiment import ExperimentServerClient

client = ExperimentServerClient()
run = client.run("PASTE_RUN_ID_HERE")
status = run.wait(timeout=60)
print(status)
print(run.log())
```

Check that `status["state"] == "SUCCEEDED"`. `wait()` times out without cancelling
the run. A failure can leave no published measurement dataset; inspect its reason
and execution log before retrying.

## 6. Read the data

Use `run.path` from the preceding Python session:

```python
from auspex_v2.data import AuspexStore

with AuspexStore(run.path) as store:
    print(store.list_datasets())
    dataset = store.read_dataset("measurement", "s21")

print(dataset.data.shape)                    # (3, 201)
print(dataset.axes["source_frequency"].unit) # Hz: outer experimental sweep
print(dataset.coordinates["frequency"].unit) # Hz: points within each VNA trace
```

The first run normally appears as `data/docs-demo/source_vna_000.auspex`.
Subsequent runs receive new numbers. Use the printed path rather than assuming a
particular number. DataBrowser can inspect metadata, but does not plot this 2D
dataset; use Python to select a trace.

## 7. Stop normally

Press **Ctrl+C in terminal 1**. The launcher requests cancellation if necessary,
waits for cleanup, and stops services while retaining registrations. Pressing it
again skips the cooperative wait and can leave ownership requiring inspection.
This is not an emergency-off mechanism.

Next: read the [annotated experiment](annotated-experiment.md), or consult
[status and recovery](utilities/auspex-status.md) if the run did not complete.
