import json
import os
import tempfile

import numpy as np
import requests

from xrdpattern.crystal import CrystalPhase, CrystalBase
from xrdpattern.pattern import XrdPattern
from xrdpattern.xrd import PowderExperiment


# -------------------------------------------------

def retrieve_cod_data(json_fpath : str, out_dirpath : str):
    with open(json_fpath, 'r') as f:
        content = f.read()

    the_dict = json.loads(content)
    print(f'done reading json. Contains {len(the_dict)} entries')

    for cod_id, data_dict in the_dict.items():
        num = cod_id.split('/')[-1]
        fname = f"COD_{num}"
        save_fpath = os.path.join(out_dirpath, f'{fname}.json')
        try:
            pattern = parse_cod_cif(num=num)
            print(f'Successfully parsed structure number {num} and saved file at {save_fpath}')
        except BaseException as e:
            a,b,c = data_dict['cell_a'], data_dict['cell_b'], data_dict['cell_c']
            a,b,c = (10*a,10*b,10*c)
            alpha, beta, gamma = data_dict['cell_alpha'], data_dict['cell_beta'], data_dict['cell_gamma']
            spg_num = data_dict['sg_number']

            x, y = data_dict['x'], data_dict['y']
            phase = CrystalPhase(lengths=(a,b,c), angles=(alpha,beta,gamma), spacegroup=spg_num, base=CrystalBase())
            powder_experiment = PowderExperiment.from_single_phase(phase=phase)
            pattern = XrdPattern(two_theta_values=np.array(x), intensities=np.array(y), powder_experiment=powder_experiment)

            print(f'Failed to extract COD pattern {num} due to error {e}. Falling back on provided data')

        pattern.save(fpath=save_fpath, force_overwrite=True)

def parse_cod_cif(num : int) -> XrdPattern:
    base_url = 'https://www.crystallography.net/cod'
    cif_request_url = f'{base_url}/{num}.cif'
    cif_content = requests.get(url=cif_request_url).content.decode()

    try:
        hkl_request_url = f'{base_url}/{num}.hkl'
        hkl_content = requests.get(url=hkl_request_url).content.decode()
        loops = hkl_content.split(f'loop_')
        xfields = ["_pd_proc_2theta_corrected", "_pd_meas_2theta_scan", "_pd_meas_2theta"]

        for l in loops:
            l = l.strip()
            if any([x in l for x in xfields]):
                cif_content += f'loop_\n{l}'
    except:
        pass

    temp_fpath = tempfile.mktemp(suffix='.cif')
    with open(temp_fpath, 'w') as f:
        f.write(cif_content)

    return XrdPattern.load(fpath=temp_fpath, mute=True)

if __name__ == "__main__":
    # cod_int = 1508528
    # pattern = parse_cod_cif(num=cod_int)
    # pattern.save(fpath=f'./thisjson.json', force_overwrite=True)

    j_fpath = '/home/daniel/aimat/data/opXRD/processed/coudert_hardiagon_0/data/extracted_data.json'
    the_out_dirpath = '/home/daniel/aimat/data/opXRD/processed/coudert_hardiagon_0/data/'
    retrieve_cod_data(json_fpath=j_fpath, out_dirpath=the_out_dirpath)
    print(f'done')
