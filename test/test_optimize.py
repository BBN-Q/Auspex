import unittest
import os
import time
import numpy as np
import tempfile

pl = None
cl = None

import QGL.config
import auspex.config
auspex.config.auspex_dummy_mode = True

# Set temporary output directories
awg_dir = tempfile.TemporaryDirectory()
kern_dir = tempfile.TemporaryDirectory()
auspex.config.AWGDir = QGL.config.AWGDir = awg_dir.name
auspex.config.KernelDir = kern_dir.name

from QGL import *
from auspex.qubit import *
from auspex.qubit.optimizer import *
import bbndb

def rosenbrock(x=1, y=1):
    z = (1.-x)**2 + 100.*(y - x**2)**2
    return np.tile([z], 4)

def parabola(x=1, y=1):
    z = (x-1)**2 + (y-1)**2
    return np.tile([z], 4)

class OptimizationTestCase(unittest.TestCase):
    """Class for unittests of the auspex optimizer."""

    @classmethod
    def setUpClass(cls):
        global cl, pl
        cl = ChannelLibrary(db_resource_name=":memory:")
        pl = PipelineManager()

    def _setUp(self, averages=4):
        self.averages = 4
        cl.clear()

        q      = cl.new_qubit("q1")
        aps2_1 = cl.new_APS2("APS2-1", address="1.2.3.4")
        aps2_2 = cl.new_APS2("APS2-2", address="1.2.3.5")
        x6     = cl.new_X6("myX6", address="1", record_length=512)
        src1   = cl.new_source("myHolz1", "HolzworthHS9000", "HS9004A-007-1", power=0)
        src2   = cl.new_source("myHolz2", "HolzworthHS9000", "HS9004A-007-2", power=0)
        cl.set_control(q, aps2_1, generator=src1)
        cl.set_measure(q, aps2_2, x6.ch(1), generator=src2)
        cl.set_master(aps2_2, aps2_2.ch("m4"))
        pl.create_default_pipeline()
        pl.reset_pipelines()
        pl["q1"].clear_pipeline()
        pl["q1"].stream_type = "integrated"
        pl["q1"].create_default_pipeline()
        cl.commit()

    @unittest.skip("Very slow test.")
    def test_scipy_optimize(self):
        self._setUp()

        def cost_function(data):
            cost = np.mean(np.real(data))
            return cost

        def sequence_function(qubit, **kwargs):
            return [[X(qubit), MEAS(qubit)] for _ in range(4)]

        opt = QubitOptimizer((cl["q1"],), sequence_function, cost_function,
                            {"x": 1.3, "y": 0.8}, optimizer="scipy", min_cost=0.08,
                            optim_params={"method": "Nelder-Mead", "options": {"disp": True, "maxiter": 100}})
        opt.setup_fake_data(cl["myX6"], parabola)

        opt.recompile = True
        result = opt.optimize()

        self.assertTrue(opt.succeeded)
        self.assertAlmostEqual(result.x[0], 1.0, places=2)
        self.assertAlmostEqual(result.x[1], 1.0, places=2)

if __name__ == '__main__':
    unittest.main()