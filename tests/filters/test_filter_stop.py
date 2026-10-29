"""Bound worker-death regressions outside the test runner's process."""
import contextlib
import faulthandler
import json
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from auspex_v2.filters import server


class PausedStop:
    """Expose the death window in a stop check, including legacy Event's lock."""

    def __init__(self, stop, context, notice):
        self.stop = stop
        self.parent = os.getpid()
        self.armed = context.RawValue("b", 0)
        self.notice = notice

    def is_set(self):
        if os.getpid() != self.parent and self.armed.value:
            # Event's real is_set holds this condition while reading its flag.
            # A lock-free flag has no such critical section to strand.
            with getattr(self.stop, "_cond", contextlib.nullcontext()):
                self.notice.send(os.getpid())
                time.sleep(60)
        return self.stop.is_set()

    def set(self):
        self.stop.set()


def probe(names_path, legacy=False):
    """Kill at the stop check; require failure, producer wakeup and full reclamation."""
    from test_filter_server import simple, batch

    faulthandler.dump_traceback_later(10)
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    stop = PausedStop(context.Event() if legacy else server._StopFlag(context), context, writer)
    with patch.object(server._StopFlag, "__new__", lambda cls, context: stop):
        graph = simple(result_capacity=1, max_windows=1)
    names = [slot.name for edge in graph.channels for slot in edge.slots]
    Path(names_path).write_text(json.dumps(names))
    graph.push(batch(0)); graph.end_window(0); graph.wait_window(0)
    graph.push(batch(1)); graph.end_window(1)
    started = threading.Event()
    errors = []

    def submit():
        started.set()
        try:
            graph.push(batch(2))
        except server.PipelineError as exc:
            errors.append(str(exc))

    producer = threading.Thread(target=submit)
    producer.start()
    assert started.wait(2) and producer.is_alive()
    stop.armed.value = 1
    assert reader.poll(5), "worker never reached stop check"
    assert reader.recv() == graph.processes["pass"].pid
    graph.processes["pass"].kill()
    assert graph.failed.wait(5), "worker death was not reported"
    producer.join(5)
    assert not producer.is_alive() and errors, "blocked producer did not fail"
    graph.close()
    graph.close()  # Repeated cleanup remains harmless.
    assert not graph.thread.is_alive()
    assert all(not child.is_alive() for child in graph.processes.values())
    assert graph.lifecycle == "FAILED"
    for name in names:
        try:
            slot = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            continue
        slot.close()
        raise AssertionError(f"shared memory not reclaimed: {name}")
    reader.close(); writer.close()
    faulthandler.cancel_dump_traceback_later()


class FilterStopTest(unittest.TestCase):
    """A dead child must not make cancellation or shared-slot cleanup unbounded."""

    def test_death_during_stop_check_reclaims_graph(self):
        """A process-group watchdog keeps a regression from hanging the suite."""
        with tempfile.TemporaryDirectory() as directory:
            names_path = Path(directory) / "slots.json"
            process = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--probe", str(names_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stdout + stderr)
            finally:
                # Include descendants on failure; never leave a stuck supervisor
                # or child behind. Reclaim slots even if the probe was killed.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                if names_path.exists():
                    for name in json.loads(names_path.read_text()):
                        with contextlib.suppress(FileNotFoundError):
                            slot = shared_memory.SharedMemory(name=name)
                            slot.close()
                            slot.unlink()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        probe(sys.argv[2], legacy="--legacy" in sys.argv)
    else:
        unittest.main()
