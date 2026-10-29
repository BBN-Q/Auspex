"""Temporary in-memory stop-flag experiment; no production source edits."""
import multiprocessing as mp
import sys
import time
import unittest
from unittest.mock import patch
from filters.test_filter_server import FilterServerTest


class StopFlag:
    """Experimental monotonic byte flag; never cleared or read-modify-written."""
    def __init__(self, ctx):
        self.flag = ctx.RawValue('b', 0)
    def is_set(self):
        return bool(self.flag.value)
    def set(self):
        self.flag.value = 1
    def wait(self, timeout):
        # Only the supervisor uses wait, for its existing 2 ms polling interval.
        if not self.is_set():
            time.sleep(timeout)
        return self.is_set()


def main():
    import faulthandler
    faulthandler.dump_traceback_later(20, repeat=True)
    ctx = mp.get_context('spawn')
    tests = ['test_worker_death_unblocks_producer', 'test_worker_death_wakes_waiters',
             'test_cancel_unblocks_producer']
    suite = unittest.TestSuite(FilterServerTest(name) for _ in range(40) for name in tests)
    if sys.argv[1] == 'flag':
        with patch.object(type(ctx), 'Event', lambda self: StopFlag(self)):
            result = unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())

if __name__ == '__main__':
    main()
