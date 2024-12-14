import os.path
import tempfile

from holytools.devtools import Unittest
from holytools.userIO import TrackedInt
from opxrd import OpXRD
from xrdpattern.pattern import XrdPattern


class TestLoading(Unittest):
    def test_dl(self):
        dl_dirpath = tempfile.mktemp()
        opxrd = OpXRD.load(dirpath=dl_dirpath, download=True)
        print(f'- Checking database loading')
        #opxrd = OpXRD.load(root_dirpath='/home/daniel/aimat/data/opXRD/final/')
        self.assertTrue(os.path.isdir(dl_dirpath))
        self.assertTrue(len(opxrd.patterns) > 10**4)

        # if self.is_manual_mode:
        #     opxrd.plot_quantity(attr='primary_phase.spacegroup')

        print(f'- Checking pattern data ok')
        k = TrackedInt(start_value=0,finish_value=len(opxrd.fpath_dict))
        for fpath, patterns in opxrd.fpath_dict.items():
            for p in patterns:
                self.check_pattern_data(fpath,p)
            k.increment()

    def test_as_database_list(self):
        dl_dirpath = tempfile.mktemp()
        dbs = OpXRD.as_database_list(root_dirpath=dl_dirpath, download=True)
        print(f'- Checking database loading')
        self.assertTrue(os.path.isdir(dl_dirpath))
        self.assertTrue(len(dbs) > 5)

        print(f'- Checking pattern data ok')
        total_num_patterns = sum([len(db.patterns) for db in dbs])
        k = TrackedInt(start_value=0,finish_value=total_num_patterns)
        for db in dbs:
            for p in db.patterns:
                self.check_pattern_data(fpath='None',p=p)
                k.increment()

    @staticmethod
    def check_pattern_data(fpath: str, p: XrdPattern):
        try:
            p.get_pattern_data()
        except Exception as e:
            print(f'Retriving pattern data failed for pattern at fpath = {fpath}')
            raise e

if __name__ == "__main__":
    TestLoading.execute_all()