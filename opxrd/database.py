import os.path
import tempfile
import zipfile

import requests
from xrdpattern.pattern import PatternDB
from holytools.userIO import TrackedInt


class OpXRD(PatternDB):
    @classmethod
    def load(cls, root_dirpath : str, download : bool = True, *args, **kwargs) -> PatternDB:
        if not os.path.isdir(root_dirpath) and download:
            tmp_fpath = tempfile.mktemp(suffix='.zip')
            OpXRD._download_zenodo_opxrd(output_fpath=tmp_fpath)
            OpXRD._unzip_file(tmp_fpath, output_dir=root_dirpath)

        print(f'- Loading patterns from local files')
        return super().load(dirpath=root_dirpath)

    @staticmethod
    def _download_zenodo_opxrd(output_fpath : str):
        zenodo_url = f'https://zenodo.org/api/records/14254271'
        file_url = f'{zenodo_url}/files/opXRD.zip/content'
        file_response = requests.get(url=file_url, stream=True)

        total_size = int(file_response.headers.get('content-length', 0))
        total_chunks = (total_size // 1024) + (1 if total_size % 1024 else 0)

        tracked_int = TrackedInt(start_value=0, finish_value=total_chunks)
        if not file_response.status_code == 200:
            raise ValueError(f'Response not ok! {file_response.status_code}')

        print(f'- Downloading opXRD database from Zenodo ({zenodo_url})')
        print(f'- Chunk progress (Size = 1kB):')
        with open(output_fpath, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=1024):
                f.write(chunk)
                tracked_int.increment(to_add=1)

    @staticmethod
    def _unzip_file(zip_fpath : str, output_dir : str):
        with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        return f"Files extracted to {output_dir}"


if __name__ == "__main__":
    opxrd = OpXRD.load(root_dirpath='../data/opxrd')