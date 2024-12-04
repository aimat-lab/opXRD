import os.path
import tempfile

from holytools.devtools import Unittest

from opxrd import OpXRD


class TestLoading(Unittest):
    def test_dl(self):
        dl_dirpath = tempfile.mktemp()
        opxrd = OpXRD.load(root_dirpath=dl_dirpath, download=True)
        self.assertTrue(os.path.isdir(dl_dirpath))
        self.assertTrue(len(opxrd.patterns) > 10**4)

        if self.is_manual_mode:
            opxrd.plot_quantity(attr='primary_phase.spacegroup')

if __name__ == "__main__":
    TestLoading.execute_all()