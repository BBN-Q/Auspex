"""Whole-window processing for ordinary acquisition scripts, without orchestration."""
from collections.abc import Mapping

from .core import Batch, SweepPosition
from .server import FilterServer


class ManualPipeline:
    """Submit whole windows to required writers and/or bounded buffer sinks.

    No instrument ownership, partial-window API, filter reconfiguration, or filter
    state inspection. Normal exit publishes the active scan and drains processing;
    exceptional exit cancels. A full buffer may block further progress: consume
    concurrently or provision capacity for the scan. Completion waits are bounded.
    """
    def __init__(self, schema, nodes=None, edges=None, *, writers=None, buffers=None,
                 input_schema=None, store_path=None, timeout=10, **options):
        from .definitions import PipelineDefinition, CompiledPipeline
        self._single_scan = False
        if isinstance(schema, (PipelineDefinition, CompiledPipeline)):
            if any(value is not None for value in (nodes, edges, writers, buffers)):
                raise ValueError("A definition supplies its own nodes, edges and sinks")
            if isinstance(schema, PipelineDefinition):
                if input_schema is not None and len(schema.inputs) != 1:
                    raise ValueError("input_schema requires a single pipeline input")
                bindings = {schema.inputs[0].name: input_schema} if input_schema is not None else None
                schema = schema.compile(inputs=bindings)
            elif input_schema is not None:
                raise ValueError("A compiled pipeline already has its input schemas")
            options.setdefault("plots", schema.plot_specs())
            schema, nodes, edges, writers, buffers, input_name = schema.runtime()
            options.setdefault("input_name", input_name)
            self._single_scan = bool(writers)
        elif input_schema is not None:
            raise ValueError("input_schema is only for pipeline definitions")
        if "results" in options or "buffer_sinks" in options:
            raise ValueError("Use completed buffer sinks, not raw/preview results")
        self._graph = FilterServer(schema, nodes, edges, {}, writers=writers,
                                   buffer_sinks=buffers, store_path=store_path, **options)
        self.buffers = self._graph.buffer_sinks
        self.timeout = timeout
        self._window = 0
        self._scan_number = 0
        self._scan = None
        self._pending_scans = []
        self._closed = False

    def submit(self, data, *, axes=(), sweep=(), sweep_units=None):
        """Copy one complete window; return its ID, not processing completion.

        sweep accepts SweepPosition objects or a name/value mapping. Mapping units
        can be supplied as sweep_units={"bias": "V"}. Values/axes must match schema;
        no implicit dtype conversions or coordinate fabrication are performed.
        """
        if self._closed:
            raise RuntimeError("Manual pipeline is closed")
        if isinstance(sweep, Mapping):
            units = sweep_units or {}
            if set(units) - set(sweep):
                raise ValueError("Sweep units name an absent coordinate")
            sweep = tuple(SweepPosition(name, value, units.get(name)) for name, value in sweep.items())
        elif sweep_units is not None:
            raise ValueError("sweep_units requires a sweep mapping")
        if self._scan is None:
            if self._single_scan and self._scan_number:
                raise ValueError("Named datasets support one scan per execution; start a new run")
            self._scan = f"sweep_{self._scan_number:03d}"
        window = self._window
        self._graph.push(Batch(group=None, window=window, sequence=0, revision=0,
                               data=data, axes=tuple(axes), scan=self._scan, sweep=tuple(sweep)))
        self._graph.end_window(window)
        self._window += 1
        return window

    def _seal_scan(self):
        """Send ordered scan end without waiting for publication (runner integration)."""
        if self._closed:
            raise RuntimeError("Manual pipeline is closed")
        if self._scan is None:
            return
        scan = self._scan
        self._graph.end_scan(scan)
        self._pending_scans.append(scan)
        self._scan = None
        self._scan_number += 1
        return scan

    def end_scan(self):
        """Wait for required endpoints and publication, including already sealed scans."""
        if self._closed:
            raise RuntimeError("Manual pipeline is closed")
        try:
            scan = self._seal_scan()
            for pending in self._pending_scans:
                self._graph.wait_scan(pending, timeout=self.timeout)
            self._pending_scans.clear()
            return scan
        except BaseException:
            self.cancel()
            raise

    def close(self):
        """Finish normally; retained buffer data survives process cleanup."""
        if self._closed:
            return
        try:
            self.end_scan()
            self._graph.finish()
            self._graph.wait_finished(timeout=self.timeout)
        finally:
            self._closed = True
            self._graph.close()

    def cancel(self):
        if not self._closed:
            self._closed = True
            self._graph.cancel()
            self._graph.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            self.cancel()
