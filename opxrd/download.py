import os.path
import tempfile
import zipfile

import requests
from xrdpattern.pattern import PatternDB


class OpXRD(PatternDB):
    @classmethod
    def load(cls, root_dirpath : str, download : bool = True, *args, **kwargs) -> PatternDB:
        if not os.path.isdir(root_dirpath) and download:
            tmp_fpath = tempfile.mktemp(suffix='.zip')
            OpXRD._download_zenodo_opxrd(output_fpath=tmp_fpath)
            OpXRD._unzip_file(tmp_fpath, output_dir=root_dirpath)

        return super().load(dirpath=root_dirpath)

    @staticmethod
    def _download_zenodo_opxrd(output_fpath : str):
        file_url = f'https://zenodo.org/api/records/14254271/files/opXRD.zip/content'
        file_response = requests.get(url=file_url, stream=True)

        if file_response.status_code == 200:
            print(f'Response ok!')
            with open(output_fpath, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=1024):
                    print(f'Writing chunk...')
                    f.write(chunk)
        print(f'attained response')


    @staticmethod
    def _unzip_file(zip_fpath : str, output_dir : str):
        with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        return f"Files extracted to {output_dir}"

if __name__ == "__main__":
    opxrd = OpXRD.load(root_dirpath='../data/opxrd')