import os
import shutil
import tempfile

from xrdpattern.xrd import XrayInfo

from processors.internal.final import FinalProcessor

# -------------------------------------------------------------

class ContributionProcessor(FinalProcessor):
    def parse_INT(self):
        db0 = self.get_db(dirname='breitung_schweidler_0', suffixes=['raw'])
        db1 = self.get_db(dirname='breitung_schweidler_1', suffixes=['raw'])
        merged = db0 + db1
        merged.save(dirpath=os.path.join(self.final_dirpath, 'INT'))

    def parse_CNRS(self):
        db = self.get_db(dirname='coudert_hardiagon_0', suffixes=['json'])
        db.save(dirpath=os.path.join(self.final_dirpath, 'CNRS'))

    def parse_USC(self):
        db0 = self.get_db(dirname='hodge_alwen_0', xray_info=self.cu_xray)
        db1 = self.get_db(dirname='hodge_alwen_1', xray_info=self.cu_xray)
        merged = db0 + db1
        merged.save(dirpath=self.get_final_dirpath('USC'))

    def parse_LBNL(self):
        perovskite_db =  self.get_csv_db(dirname='sutter-fella_singh_0', orientation='horizontal')
        perovskite_db += self.get_csv_db(dirname='sutter-fella_kodalle_0', orientation='horizontal', suffixes=['dat','csv'])
        xray_info = XrayInfo(primary_wavelength=1.23984, secondary_wavelength=None)
        perovskite_db.set_xray(xray_info=xray_info)
        perovskite_db.save(dirpath=self.get_final_dirpath('LBNL','LBNL-A'))

        db_B = self.get_csv_db(dirname='sutter-fella_abdelsamie_0', orientation='horizontal')
        db_B.set_xray(xray_info=xray_info)
        db_B.save(dirpath=self.get_final_dirpath('LBNL','LBNL-B'))

        mn_sb_db = self.get_csv_db(dirname='sutter-fella_heymans_0', suffixes=['xlsx'], orientation='vertical')
        mn_sb_db.save(dirpath=self.get_final_dirpath('LBNL','LBNL-C'))

        uio_db = self.get_csv_db(dirname='sutter-fella_hu_0', orientation='horizontal')
        uio_db.save(dirpath=self.get_final_dirpath('LBNL','LBNL-D'))

    def parse_EMPA(self):
        db0 = self.get_db(dirname='siol_wieczorek_0', xray_info=self.cu_xray)
        db1 = self.get_db(dirname='siol_zhuk_0', xray_info=self.cu_xray)
        merged = db0 + db1
        merged.save(dirpath=self.get_final_dirpath('EMPA'))

    def parse_IKFT(self):
        db = self.get_db(dirname='wolf_wolf_0', xray_info=self.cu_xray)
        db.save(dirpath=self.get_final_dirpath('IKFT'))

    def parse_HKUST(self):
        db = self.get_db(dirname='zhang_cao_0', xray_info=self.cu_xray)
        db.save(dirpath=self.get_final_dirpath('HKUST','HKUST-A'))

        db1 = self.get_db(dirname='zhang_cao_1', xray_info=self.cu_xray)
        db1.save(dirpath=self.get_final_dirpath('HKUST', 'HKUST-B'))

    # --------------------------------------------------------------------

    def prepare_zips(self):
        dir_content_names = os.listdir(self.final_dirpath)
        dir_content_paths = [os.path.join(self.final_dirpath, name) for name in dir_content_names]
        dirpaths = [d for d in dir_content_paths if os.path.isdir(d)]

        in_situ_dirs = [d for d in dirpaths if 'LBNL' in d]
        non_situ_dirs = [d for d in dirpaths if d not in in_situ_dirs]

        print(f'In situ dirs: {in_situ_dirs}')
        print(f'Non situ dirs: {non_situ_dirs}')

        in_situ_fpath = os.path.join(self.final_dirpath, 'opxrd_in_situ1.zip')
        self._zip_dirs(in_situ_dirs, output_fpath=in_situ_fpath)

        non_situ_fpath = os.path.join(self.final_dirpath, 'opxrd.zip')
        self._zip_dirs(non_situ_dirs, output_fpath=non_situ_fpath)

    @staticmethod
    def _zip_dirs(dirpaths : list[str], output_fpath : str):
        if len(dirpaths) == 0:
            raise ValueError('No directories to zip')

        tmp_dir = tempfile.mktemp()
        for d in dirpaths:
            print(f'Copying {d} to {tmp_dir}')
            tmp_dirpath = os.path.join(tmp_dir, os.path.basename(d))
            shutil.copytree(d,tmp_dirpath)
        print(f'Zipping {tmp_dir} to {output_fpath}')
        parts = output_fpath.split('.')[:-1]
        output_fpath = '.'.join(parts)
        shutil.make_archive(base_name=output_fpath, format='zip', root_dir=tmp_dir)


if __name__ == "__main__":
    processor = ContributionProcessor(root_dirpath='/media/daniel/mirrors/xrd.aimat.science/local')
    processor.parse_LBNL()