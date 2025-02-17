import json
import os
import tempfile
import uuid

from xrdpattern.pattern import XrdPattern

icsd_json_path = '/home/daniel/Drive/data/workspace/icsd/dataset.json'
keywords = ['fullprof', 'rietveld', 'powder']
save_dir = '/home/daniel/Drive/data/workspace/icsd_extracted'

with open(icsd_json_path, 'r') as f:
    the_dict = json.loads(f.read())
print(f'Loaded icsd dataset')


for num, content in the_dict.items():
    lower_content = content.lower()
    if not any([word in lower_content for word in keywords]):
        print(f'Could not find any matching keywords in cif file number {num}')
        continue
    else:
        print(f'Found keyword matching powder diffraction in cif file number {num}! \n'
              f'Starting pattern extraction')

    try:
        tmp_fpath = tempfile.mktemp(suffix='.cif')
        lines = content.split('\n')
        content = '\n'.join(lines[1:])

        with open(tmp_fpath, 'w') as f:
            f.write(content)
        pattern = XrdPattern.load(fpath=tmp_fpath)
        fpath = os.path.join(save_dir, str(uuid.uuid4()))
        pattern.save(fpath=fpath)
        print(f'Successfullly extracted pattern from cif and wrote information to {fpath}')
    except Exception as e:
        print(f'An error occured during extraction: {e}')

