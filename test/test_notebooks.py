import glob
from Notebook_testcase import NotebooksTestCase

class NotebooksMethods(NotebooksTestCase):
    def _setup(self):
        import QGL.config
        import auspex.config
        import tempfile
        auspex.config.auspex_dummy_mode = True

        # Set temporary output directories
        awg_dir = tempfile.TemporaryDirectory()
        kern_dir = tempfile.TemporaryDirectory()
        auspex.config.AWGDir = QGL.config.AWGDir = awg_dir.name
        auspex.config.KernelDir = kern_dir.name


    # def test_01_Essential_Objects(self):
    #     self._setup()
    #     self.runNotebook_jupyter("../doc/examples/","Example-Config.ipynb")
    #
    # def test_02_Essential_Objects(self):
    #     self._setup()
    #     self.runNotebook_jupyter("../doc/examples/","Example-Channel-Lib.ipynb")
    #
    # def test_03_Essential_Objects(self):
    #     self._setup()
    #     self.runNotebook_jupyter("../doc/examples/","Example-Calibrations.ipynb")
    #
    # def test_04_Essential_Objects(self):
    #     self._setup()
    #     self.runNotebook_jupyter("../doc/examples/","Example-Filter-Pipeline.ipynb")
    #
    # def test_05_Essential_Objects(self):
    #     self._setup()
    #     self.runNotebook_jupyter("../doc/examples/","Example-Experiments.ipynb")
    #
    # def test_06_Essential_Objects(self):
    #     self._setup()
    #     self.runNotebook_jupyter("../doc/examples/","Example-SingleShot-Fid.ipynb")
    #
    # def test_07_Essential_Objects(self):
    #     self._setup()
    #     self.runNotebook_jupyter("../doc/examples/","Example-Sweeps.ipynb")

    def test_All_Notebooks(self):
        files = glob.glob('../doc/examples/*.ipynb')
        for f in files:
            self._setup()
            self.runNotebook_jupyter(".", f)
