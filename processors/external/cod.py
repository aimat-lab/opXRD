import json
import os
import tempfile

import numpy as np
import requests
from pymatgen.core import Lattice

from xrdpattern.crystal import CrystalStructure
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
            print(f'Failed to extract COD pattern {num} due to error {e}. Falling back on provided data')

            a,b,c = data_dict['cell_a'], data_dict['cell_b'], data_dict['cell_c']
            a,b,c = (10*a,10*b,10*c)
            alpha, beta, gamma = data_dict['cell_alpha'], data_dict['cell_beta'], data_dict['cell_gamma']
            spg_num = data_dict['sg_number']

            x, y = data_dict['x'], data_dict['y']
            lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            phase = CrystalStructure(lattice=lattice, spacegroup=spg_num, basis=None)
            powder_experiment = PowderExperiment.from_single_phase(phase=phase)
            try:
                pattern = XrdPattern(two_theta_values=np.array(x), intensities=np.array(y), powder_experiment=powder_experiment)
            except:
                print(f'Failed to create pattern from data {data_dict}')
                continue

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

    return XrdPattern.load(fpath=temp_fpath)

if __name__ == "__main__":
    extracted_fpath = '/media/daniel/mirrors/xrd.aimat.science/local/prepared/coudert_hardiagon_0/original/extracted_data.json'
    target_dirpath = '/media/daniel/mirrors/xrd.aimat.science/local/prepared/coudert_hardiagon_0/ready'
    retrieve_cod_data(json_fpath=extracted_fpath, out_dirpath=target_dirpath)
    print(f'done')
