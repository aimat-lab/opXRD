import os.path

import numpy as np
from ase.db import connect
from ase.db.core import Database
from numpy.typing import NDArray
from pymatgen.core import Lattice
from xrdpattern.crystal import CrystalStructure, CrystalBasis, AtomSite
from xrdpattern.pattern import XrdPattern
from xrdpattern.xrd import PowderExperiment, XrayInfo


# -------------------------------------

def get_xrdpattern(database: Database, index: int, add_labels : bool = True) -> XrdPattern:
    row = database.get(id=index)
    two_theta_values = get_as_float_arr('angle', row=row)
    intensities = get_as_float_arr('intensity', row=row)

    if add_labels:
        atom = database.get_atoms(id=index)
        basis = make_base(chemical_symbols=atom.get_chemical_symbols(), fract_positions=atom.get_positions())
        a, b, c, alpha, beta, gamma = atom.get_cell_lengths_and_angles().tolist()
        lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        phase = CrystalStructure(lattice=lattice, basis=basis)
        experiment = PowderExperiment(phases=[phase], xray_info=XrayInfo.copper_xray())
        p = XrdPattern(two_theta_values=np.array(two_theta_values), intensities=np.array(intensities), powder_experiment=experiment)
    else:
        p = XrdPattern.make_unlabeled(two_theta_values=two_theta_values, intensities=intensities)

    return p

def get_as_float_arr(name : str, row) -> list[float]:
    return eval(getattr(row, name))

def make_base(chemical_symbols : list[str], fract_positions : NDArray) -> CrystalBasis:
    if not len(chemical_symbols) == len(fract_positions):
        raise ValueError('The number of chemical symbols and positions must be equal')

    atoms : list[AtomSite] = []
    for symbol, (x,y,z) in zip(chemical_symbols, fract_positions):
        atoms.append(AtomSite(species_str=symbol, x=x, y=y, z=z, occupancy=1))

    return CrystalBasis(atoms)

if __name__ == "__main__":
    processing_dirpath = '/home/daniel/aimat/data/opXRD/processed/zhang_cao_1'
    database_fpath = os.path.join(processing_dirpath,'caobin.db')
    print(f'Reading form database at {database_fpath}')
    data = connect(database_fpath)

    print(f'Reading data from database containing {data.count()} entries')
    for idx in range(1, data.count()+1):
        xrdpattern = get_xrdpattern(data, index=idx, add_labels=False)
        xrdpattern.save(fpath=os.path.join(processing_dirpath, 'data', f'pattern_{idx}.json'), force_overwrite=True)
        print(f'Saved pattern {idx} to file')

