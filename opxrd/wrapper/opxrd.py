import os.path
import tempfile
import time
import zipfile

import requests
from xrdpattern.pattern import PatternDB
from holytools.userIO import TrackedInt

from xrdpattern.pattern.db import patterdb_logger

# -----------------------------


class OpXRD(PatternDB):
    @classmethod
    def load(cls, dirpath : str, download : bool = True, download_in_situ : bool = False, *args, **kwargs) -> PatternDB:
        dirpath = os.path.expanduser(dirpath)
        dirpath = os.path.abspath(dirpath)

        if not os.path.isdir(dirpath) and download:
            cls._prepare_files(root_dirpath=dirpath, include_in_situ=download_in_situ)

        return super().load(dirpath=dirpath, strict=True)


    @classmethod
    def load_project_list(cls, root_dirpath : str, download : bool = True, download_in_situ : bool = False) -> list[PatternDB]:
        if not os.path.isdir(root_dirpath) and download:
            cls._prepare_files(root_dirpath=root_dirpath, include_in_situ=download_in_situ)

        pattern_dbs = []
        print(f'- Loading databases from {root_dirpath}')
        dirpaths = [f.path for f in os.scandir(root_dirpath) if f.is_dir()]
        for d in dirpaths:
            institution_name = os.path.basename(d)
            if any([os.path.isdir(s.path) for s in os.scandir(d)]):
                subdirs = [s.path for s in os.scandir(d) if s.is_dir()]
                new_dbs = [PatternDB.load(dirpath=s, strict=True) for s in subdirs]
                for k, db in enumerate(new_dbs):
                    upper_alphabet = [chr(i) for i in range(65, 91)]
                    db.name = f'{institution_name}-{upper_alphabet[k]}'
                pattern_dbs += new_dbs
            else:
                time.sleep(0.01)
                db = PatternDB.load(dirpath=d, strict=True)
                pattern_dbs.append(db)
        return pattern_dbs

    @classmethod
    def _prepare_files(cls, root_dirpath : str, include_in_situ : bool = False):
        tmp_fpath = tempfile.mktemp(suffix='.zip')
        OpXRD._download_zenodo_opxrd(output_fpath=tmp_fpath)
        OpXRD._unzip_file(tmp_fpath, output_dir=root_dirpath)
        if include_in_situ:
            cls._download_zenodo_opxrd(output_fpath=tmp_fpath, filename='opxrd_in_situ.zip')
            cls._unzip_file(tmp_fpath, output_dir=root_dirpath)


    @classmethod
    def _download_zenodo_opxrd(cls, output_fpath : str, filename : str = 'opxrd.zip'):
        try:
            zenodo_url = f'https://zenodo.org/api/records/{cls.get_latest_record_id()}'
            file_response = requests.get(url=f'{zenodo_url}/files/{filename}/content', stream=True)
        except Exception as e:
            raise ConnectionError(f'Failed to download opXRD database from Zenodo. Reason: {e.__repr__()}')

        total_size = int(file_response.headers.get('content-length', 0))
        total_chunks = (total_size // 1024) + (1 if total_size % 1024 else 0)

        if not file_response.status_code == 200:
            raise ValueError(f'Response returned error status code {file_response.status_code}. Reason: {file_response.reason}')

        patterdb_logger.info(f'Downloading opXRD database from Zenodo ({zenodo_url})')
        patterdb_logger.info(f'Download chunk progress (Chunk size = 1kB):')
        time.sleep(0.01)
        tracked_int = TrackedInt(start_value=0, finish_value=total_chunks)
        with open(output_fpath, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=1024):
                f.write(chunk)
                tracked_int.increment(to_add=1)

    @staticmethod
    def _unzip_file(zip_fpath : str, output_dir : str):
        print(f'- Unziping downloaded files to {output_dir}')
        with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        return f"- Files extracted to {output_dir}"

    @classmethod
    def get_latest_record_id(cls) -> int:
        response = requests.get(url=f'https://zenodo.org/records/14254270')
        url = response.links['linkset']['url']
        record_id = int(url.split('/')[-1])
        return record_id


    @staticmethod
    def merge_databases(dbs : list[PatternDB], common_prefix_length : int = 4) -> list[PatternDB]:
        db_groups = {}
        for db in dbs:
            three_letter_name = db.name[:common_prefix_length]
            if not three_letter_name in db_groups:
                db_groups[three_letter_name] = [db]
            else:
                db_groups[three_letter_name].append(db)

        merged_dbs = []
        for name, g in db_groups.items():
            merged = PatternDB.merge(dbs=g)
            merged.name = name
            merged_dbs.append(merged)
        return merged_dbs


if __name__ == "__main__":
    smoltest_dirpath = '/home/daniel/aimat/data/opXRD/test_smol'
    test_dirpath = '/home/daniel/aimat/data/opXRD/test'
    opxrd_dbs = OpXRD.load_project_list(root_dirpath=test_dirpath)