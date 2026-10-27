# Auspex V2

![Auspex](assets/auspex.png)

Auspex is a Python framework for laboratory measurements. It connects instrument
control, reusable measurement definitions, numerical processing, and data storage.
Use only the parts needed for a measurement: drivers and storage also work without
experiment services.

**This is the draft V2 manual.** APIs and file formats are still evolving.
[V1 documentation](https://auspex.readthedocs.io/en/latest/) remains separate;
V2 does not promise compatibility with V1 experiments or data files.

## Start here

- [Quickstart](quickstart.md): run a simulated source/VNA sweep and inspect its data.
- [Annotated experiment](annotated-experiment.md): understand the same measurement,
  from declarations to cleanup.
- [Components](components/experiments.md): learn the responsibilities and limits
  of experiments, instruments, filters, and storage.
- [Utilities](utilities/index.md): choose a service, console, or inspection tool.

## The measurement path

1. Define an experiment and its processing pipeline in Python.
2. Configure instrument roles and sweep points, then compile an inspectable plan.
3. Submit it to ExperimentServer. The submitting notebook is not the run owner.
4. The runner acquires instrument ownership, executes the plan, and submits data.
5. Required writers publish completed datasets. Inspect final status and saved data.

Compilation is not acquisition. Submission is not completion. A completed point is
not yet a guarantee of successful cleanup or data publication.

## Operating domain

V2 currently supports one active writer-backed run on a local acquisition host.
The implementation uses Unix sockets and advisory file locks. Commands in this
manual assume a Unix-like shell and Python 3.13 or newer.

Examples use simulated instruments unless explicitly stated otherwise. They test
software behavior, not physical measurement validity or safe RF/bias settings.
Before using real hardware, review cabling, ranges, timing, output policy, and the
[ownership and recovery rules](components/instruments.md#ownership-and-recovery).
Software cancellation and process termination are not hardware emergency stops.

Recorded Python calculations execute trusted code. An inspectable plan does not
freeze its dependencies or establish repeatability; see
[recorded calculations](components/experiments.md#recorded-calculations).

## Existing material

This manual draws on the Markdown guides retained beside the implementation.
Component and utility pages link to those engineering notes and to example code;
they are not removed or replaced. Some notes retain historical milestone text.
The draft manual describes the current supported path rather than reproducing
those historical claims.

The [repository README](https://github.com/BBN-Q/Auspex/blob/auspex-v2/README.md)
contains project background, funding acknowledgement, and the development disclosure.
To edit or preview this manual, see [Maintaining the docs](maintaining.md).
