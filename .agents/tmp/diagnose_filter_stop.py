"""Temporary deterministic failure injection; production files are unchanged."""
import faulthandler
import multiprocessing as mp
from multiprocessing.synchronize import Event
import os
import threading
import time
import sys
from stress_filter_stop import StopFlag
from unittest.mock import patch

from filters.test_filter_server import simple, batch


class PausedEvent(Event):
    """Pause a child inside the real is_set critical section before SIGKILL."""
    def __init__(self, ctx, gate, notice):
        super().__init__(ctx=ctx)
        self.parent = os.getpid()
        self.gate = gate
        self.notice = notice

    def is_set(self):
        with self._cond:
            if os.getpid() != self.parent and self.gate.value:
                self.notice.send(os.getpid())
                time.sleep(30)
            if self._flag.acquire(False):
                self._flag.release()
                return True
            return False


class PausedFlag(StopFlag):
    def __init__(self, ctx, gate, notice):
        super().__init__(ctx)
        self.parent = os.getpid()
        self.gate = gate
        self.notice = notice

    def is_set(self):
        if os.getpid() != self.parent and self.gate.value:
            self.notice.send(os.getpid())
            time.sleep(30)
        return super().is_set()


def release_orphan(lock):
    # Separate process avoids corrupting the parent's semaphore ownership count.
    lock.release()


def main():
    ctx = mp.get_context('spawn')
    gate = ctx.RawValue('b', 0)
    reader, writer = ctx.Pipe(duplex=False)
    flag = len(sys.argv) > 1 and sys.argv[1] == 'flag'
    stop = (PausedFlag if flag else PausedEvent)(ctx, gate, writer)
    with patch.object(type(ctx), 'Event', lambda self: stop):
        graph = simple(result_capacity=1, max_windows=1)
    graph.push(batch(0)); graph.end_window(0); graph.wait_window(0)
    graph.push(batch(1)); graph.end_window(1)
    errors = []
    def push():
        try:
            graph.push(batch(2))
        except Exception as exc:
            errors.append(str(exc))
    producer = threading.Thread(target=push)
    producer.start()
    gate.value = 1
    assert reader.poll(5), 'worker did not reach is_set'
    print('Worker paused in stop check:', reader.recv(), 'flag =', flag, flush=True)
    graph.processes['pass'].kill()
    graph.processes['pass'].join(2)
    closer = threading.Thread(target=graph.close)
    closer.start()
    closer.join(1)
    print('After SIGKILL: close blocked =', closer.is_alive(),
          'producer blocked =', producer.is_alive(),
          'supervisor blocked =', graph.thread.is_alive(), flush=True)
    faulthandler.dump_traceback()
    # Diagnostic rescue only: knowingly release the lock held by the dead child.
    # This is NOT a proposed production recovery technique.
    if not flag:
        rescue = ctx.Process(target=release_orphan, args=(stop._cond._lock,))
        rescue.start(); rescue.join(5)
        assert rescue.exitcode == 0
    closer.join(5); producer.join(5)
    print('After diagnostic lock release: close blocked =', closer.is_alive(),
          'producer blocked =', producer.is_alive(), 'errors =', errors, flush=True)
    assert not closer.is_alive() and not producer.is_alive()
    assert not graph.thread.is_alive()
    assert all(not p.is_alive() for p in graph.processes.values())


if __name__ == '__main__':
    main()
