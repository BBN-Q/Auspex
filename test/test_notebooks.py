from Notebook_testcase import NotebooksTestCase

class NotebooksMethods(NotebooksTestCase):
    def test_01_Essential_Objects(self):
        self.runNotebook_jupyter("../doc/examples/","Example-Config.ipynb")

    def test_02_Essential_Objects(self):
        self.runNotebook_jupyter("../doc/examples/","Example-Channel-Lib.ipynb")

    def test_03_Essential_Objects(self):
        self.runNotebook_jupyter("../doc/examples/","Example-Calibrations.ipynb")

    def test_04_Essential_Objects(self):
        self.runNotebook_jupyter("../doc/examples/","Example-Filter-Pipeline.ipynb")

    def test_05_Essential_Objects(self):
        self.runNotebook_jupyter("../doc/examples/","Example-Experiments.ipynb")

    def test_06_Essential_Objects(self):
        self.runNotebook_jupyter("../doc/examples/","Example-SingleShot-Fid.ipynb")

    def test_07_Essential_Objects(self):
        self.runNotebook_jupyter("../doc/examples/","Example-Sweeps.ipynb")
