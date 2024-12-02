# from xrdpattern.pattern import PatternDB
#
#
# class OpXRD(PatternDB):
#     def __init__(self):
import requests


def download_zenodo_opxrd(output_path : str = '../data/opxrd_data.zip'):
    file_url = f'https://zenodo.org/api/records/14254271/files/opXRD.zip/content'
    file_response = requests.get(url=file_url, stream=True)

    if file_response.status_code == 200:
        print(f'Response ok!')
        with open(output_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=1024):
                print(f'Writing chunk...')
                f.write(chunk)
    print(f'attained response')

if __name__ == "__main__":
    download_zenodo_opxrd()