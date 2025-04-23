import os

import numpy as np

from xrdpattern.crystal import CrystalStructure
from xrdpattern.pattern import XrdPattern
from xrdpattern.xrd import PowderExperiment

source_dirpath = '/media/daniel/mirrors/xrd.aimat.science/local/prepared/zhang_cao_0/original/caobin_pxrd_xy'
target_dirpath = '/media/daniel/mirrors/xrd.aimat.science/local/prepared/zhang_cao_0/ready'

patterns = []
subdirs = [x for x in os.listdir(source_dirpath) if os.path.isdir(os.path.join(source_dirpath, x))]


for j, dirname in enumerate(subdirs):
    subdir_path = os.path.join(source_dirpath, dirname)
    if not os.path.isdir(subdir_path):
        continue

    filenames = os.listdir(subdir_path)
    fpaths = [os.path.join(subdir_path, filename) for filename in filenames]

    phases = []
    for cif_fpath in [x for x in fpaths if x.endswith('.cif')]:
        with open(cif_fpath, 'r') as f:
            try:
                phases.append(CrystalStructure.from_cif(cif_content=f.read()))
            except:
                print(f'Failed to read cif file {cif_fpath}')
    experiment = PowderExperiment.from_multi_phase(phases)

    xy_fpath = [x for x in fpaths if x.endswith('.txt')][0]
    with open(xy_fpath, 'r') as f:
        xylines = f.readlines()
        x,y = [], []
        for l in xylines:
            xval, yval = l.split(',')
            xval, yval = float(xval), float(yval)
            x.append(xval)
            y.append(yval)

    pattern = XrdPattern(two_theta_values=np.array(x), intensities=np.array(y), powder_experiment=experiment)
    fpath = os.path.join(target_dirpath, f'pattern_{j}.json')
    pattern.save(fpath=fpath)


