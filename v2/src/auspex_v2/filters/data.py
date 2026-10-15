"""Pipeline consumers for in-memory results, previews, and completed datasets."""

import numpy as np

from ..data import Axis, AuspexStore
from .core import Batch, Filter, PipelineError


class Collector(Filter):
    """Collect required results or best-effort plot previews for an in-process run."""

    def __init__(self, schema, *, preview=False):
        self.input_schema = schema
        self.preview = preview
        self.best_effort = preview
        self.batches = []
        self.windows = []

    def on_batch(self, batch):
        # A transport-backed sink must copy before releasing the producer's slot.
        saved = Batch(batch.group, batch.window, batch.sequence, batch.revision,
                      np.array(batch.data, copy=True),
                      tuple(Axis(a.name, np.array(a.points, copy=True), a.unit) for a in batch.axes),
                      batch.count, batch.expected_count)
        if self.preview:
            self.batches[:] = [saved]  # bounded latest snapshot, not a data archive
        else:
            self.batches.append(saved)
        return ()

    def on_window_end(self, window):
        if self.preview:
            self.windows[:] = [window]
        else:
            self.windows.append(window)
        return ()


class DatasetWriter(Filter):
    """Publish completed, single-batch windows using one shared .auspex writer."""

    def __init__(self, schema, store: AuspexStore, *, group: str, when="window"):
        if when not in ("window", "group"):
            raise ValueError("Writer publishes at window or group completion")
        self.input_schema = schema
        self.store = store
        self.group = group
        self.when = when
        self.pending = None
        self.published = []

    def on_batch(self, batch):
        if self.pending is not None:
            raise PipelineError("Writer expects one completed array per window/group")
        self.pending = Batch(batch.group, batch.window, batch.sequence, batch.revision,
            np.array(batch.data, copy=True),
            tuple(Axis(a.name, np.array(a.points, copy=True), a.unit) for a in batch.axes),
            batch.count, batch.expected_count)
        return ()

    def _publish(self):
        if self.pending is not None:
            batch = self.pending
            name = f"{batch.group or 'window'}_{batch.window:04d}"
            self.store.write_dataset(self.group, name, batch.data, batch.axes)
            self.published.append((self.group, name))
            self.pending = None

    def on_window_end(self, window):
        if self.when == "window":
            if self.pending is None or self.pending.window != window:
                raise PipelineError("Writer is missing a complete window")
            self._publish()
        return ()

    def on_group_end(self, group):
        if self.when == "group":
            if self.pending is None or self.pending.group != group:
                raise PipelineError("Writer is missing a complete group result")
            self._publish()
        return ()

    def on_run_end(self):
        if self.pending is not None:
            raise PipelineError("Writer has an unpublished dataset")
        return ()
