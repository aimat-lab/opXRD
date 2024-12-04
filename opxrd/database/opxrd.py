import os.path
import tempfile
import zipfile

import requests
from xrdpattern.pattern import PatternDB
from holytools.userIO import TrackedInt


class OpXRD(PatternDB):
    @classmethod
    def load(cls, root_dirpath : str, download : bool = True, *args, **kwargs) -> PatternDB:
        root_dirpath = os.path.expanduser(root_dirpath)
        root_dirpath = os.path.abspath(root_dirpath)

        if not os.path.isdir(root_dirpath) and download:
            tmp_fpath = tempfile.mktemp(suffix='.zip')
            OpXRD._download_zenodo_opxrd(output_fpath=tmp_fpath)
            OpXRD._unzip_file(tmp_fpath, output_dir=root_dirpath)


        print(f'- Loading patterns from local files')
        return super().load(dirpath=root_dirpath, strict=True)

    @staticmethod
    def _download_zenodo_opxrd(output_fpath : str):
        zenodo_url = f'https://zenodo.org/records/14278656'
        file_url = f'{zenodo_url}/files/opXRD.zip?download=1'
        file_response = requests.get(url=file_url)

        file_response = requests.get(url=f'https://zenodo.org/records/14278656/files/opxrd.zip?download=1', stream=True)

        total_size = int(file_response.headers.get('content-length', 0))
        total_chunks = (total_size // 1024) + (1 if total_size % 1024 else 0)

        if not file_response.status_code == 200:
            raise ValueError(f'Response returned error status code {file_response.status_code}. Reason: {file_response.reason}')

        tracked_int = TrackedInt(start_value=0, finish_value=total_chunks)
        print(f'- Downloading opXRD database from Zenodo ({zenodo_url})')
        print(f'- Chunk progress (Size = 1kB):')
        with open(output_fpath, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=1024):
                f.write(chunk)
                tracked_int.increment(to_add=1)

    @staticmethod
    def _unzip_file(zip_fpath : str, output_dir : str):
        print(f'- Unziping downloaded files to {output_dir}')
        with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        return f"Files extracted to {output_dir}"


if __name__ == "__main__":
    opxrd = OpXRD.load(root_dirpath='../data/opxrd')