import os

import numpy as np

from xrdpattern.pattern import XrdPattern
from xrdpattern.xrd import PowderExperiment

root_dirpath = '/home/daniel/aimat/data/full/zhang_cao_0/data/caobin_pxrd_xy'
target_dirpath = '/home/daniel/aimat/data/final/HKUST-A'

patterns = []
subdirs = [x for x in os.listdir(root_dirpath) if os.path.isdir(os.path.join(root_dirpath, x))]


for j, dirname in enumerate(subdirs):
    subdir_path = os.path.join(root_dirpath, dirname)
    if not os.path.isdir(subdir_path):
        continue

    filenames = os.listdir(subdir_path)
    fpaths = [os.path.join(subdir_path, filename) for filename in filenames]

    cif_fpath = [x for x in fpaths if x.endswith('.cif')][0]
    xy_fpath = [x for x in fpaths if x.endswith('.txt')][0]

    with open(cif_fpath, 'r') as f:
        experiment = PowderExperiment.from_cif(f.read())
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


