"""Single spawned manifest owner for all required graph writer sinks."""

import os
import time

from ..data import AuspexStore
from .scan import ScanAssembler
from .transport import _send


def store_owner(path, writers, schemas, edges, status, stop, run_metadata, input_name):
    """Multiplex bounded writer edges, acknowledging only published datasets."""
    assemblers = {}
    try:
        with AuspexStore(path, mode="w") as store:
            if run_metadata is not None:
                store.write_run_metadata(run_metadata)
            assemblers = {name: ScanAssembler(schemas[name], max_scan_bytes=spec.max_scan_bytes,
                                               max_scan_windows=spec.max_scan_windows,
                                               scratch_dir=store.path)
                          for name, spec in writers.items()}
            _send(status, {"type": "ready"})
            finished = set()
            supervisor_pid = os.getppid()
            while not stop.is_set() and len(finished) != len(writers):
                if os.getppid() != supervisor_pid:
                    return  # Do not keep writing if supervision disappears.
                progress = False
                for name, edge in edges.items():
                    if name in finished or not edge.reader_connection.poll():
                        continue
                    progress = True
                    event, payload, slot = edge.receive()
                    try:
                        assembler = assemblers[name]
                        if event == "batch":
                            assembler.on_batch(payload)
                        elif event == "window_end":
                            assembler.on_window_end(payload["window"])
                            _send(status, {"type": "window", "writer": name, "window": payload["window"]})
                        elif event == "group_end":
                            pass  # Group boundaries are filter-owned; scans belong to the runner.
                        elif event == "scan_end":
                            scan = payload["group"]
                            spec = writers[name]
                            dataset_name = spec.dataset if spec.dataset is not None else f"{spec.prefix}{scan}"
                            assembler.publish(scan, store, spec.group, dataset_name,
                                              label=scan, source=f"{spec.node}.{spec.port}",
                                              input_name=input_name)
                            _send(status, {"type": "scan", "writer": name, "scan": scan,
                                           "group": spec.group, "dataset": dataset_name})
                        elif event == "run_end":
                            if assembler.pending is not None or assembler.rows:
                                raise ValueError(f"Unpublished scan in writer {name}")
                            finished.add(name)
                        else:
                            raise ValueError(f"Unexpected writer event: {event}")
                    finally:
                        if slot is not None:
                            edge.release(slot)
                if not progress:
                    time.sleep(0.01)  # never kill a child while it holds Event's condition lock
            if not stop.is_set():
                _send(status, {"type": "finished"})
    except Exception as exc:
        if not stop.is_set():
            _send(status, {"type": "error", "error": f"Store owner: {type(exc).__name__}: {exc}"})
    finally:
        for assembler in assemblers.values():
            assembler.close()
        status.close()
        for edge in edges.values():
            edge.releases.cancel_join_thread()
