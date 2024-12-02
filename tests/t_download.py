import os.path
from holytools.devtools import Unittest
from opxrd import OpXRD


class TestDownload(Unittest):
    def test_dl(self):
        dl_dir = '../data/opxrd'
        opxrd = OpXRD.load(root_dirpath=dl_dir, download=True)
        self.assertTrue(os.path.isdir(dl_dir))
        self.assertTrue(len(opxrd.patterns) > 1)


if __name__ == "__main__":
    TestDownload.execute_all()