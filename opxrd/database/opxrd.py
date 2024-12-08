import os.path
import tempfile
import time
import zipfile

import requests
from xrdpattern.pattern import PatternDB
from holytools.userIO import TrackedInt

# -----------------------------


class OpXRD(PatternDB):
    @classmethod
    def load(cls, root_dirpath : str, download : bool = True, *args, **kwargs) -> PatternDB:
        root_dirpath = os.path.expanduser(root_dirpath)
        root_dirpath = os.path.abspath(root_dirpath)

        if not os.path.isdir(root_dirpath) and download:
            cls._prepare_files(root_dirpath=root_dirpath)

        print(f'- Loading patterns from local files')
        return super().load(dirpath=root_dirpath, strict=True)


    @classmethod
    def as_database_list(cls, root_dirpath : str, download : bool = True) -> list[PatternDB]:
        if not os.path.isdir(root_dirpath) and download:
            cls._prepare_files(root_dirpath=root_dirpath)

        pattern_dbs = []
        print(f'- Loading databases from {root_dirpath}')
        for d in os.listdir(path=root_dirpath):
            dirpath = os.path.join(root_dirpath, d)
            time.sleep(0.01)
            #TODO: This is temporary
            db = PatternDB.load(dirpath=dirpath, strict=False)
            #db = PatternDB.load(dirpath=dirpath, strict=True)
            db.name = d

            pattern_dbs.append(db)
        return pattern_dbs

    @classmethod
    def _prepare_files(cls, root_dirpath : str):
        tmp_fpath = tempfile.mktemp(suffix='.zip')
        OpXRD._download_zenodo_opxrd(output_fpath=tmp_fpath)
        OpXRD._unzip_file(tmp_fpath, output_dir=root_dirpath)

    @classmethod
    def _download_zenodo_opxrd(cls, output_fpath : str):
        zenodo_url = f'https://zenodo.org/api/records/{cls.get_record_id()}'
        file_response = requests.get(url=f'{zenodo_url}/files/opxrd.zip/content', stream=True)

        total_size = int(file_response.headers.get('content-length', 0))
        total_chunks = (total_size // 1024) + (1 if total_size % 1024 else 0)

        if not file_response.status_code == 200:
            raise ValueError(f'Response returned error status code {file_response.status_code}. Reason: {file_response.reason}')

        print(f'- Downloading opXRD database from Zenodo ({zenodo_url})')
        print(f'- Download chunk progress (Chunk size = 1kB):')
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
    def get_record_id(cls) -> int:
        return 14289287

if __name__ == "__main__":
    opxrd = OpXRD.load(root_dirpath='../data/opxrd')