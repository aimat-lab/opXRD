from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from numpy._typing import NDArray
from pymatgen.core import Species
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from xrdpattern.crystal import CrystalStructure

# --------------------------------------------------------------------------

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}


class VAECrystal(object):
    def __init__(self, frac_coords, atom_types, lengths, angles):
        self.frac_coords : list  = frac_coords
        self.atom_types : list[Species] = atom_types
        self.lengths : tuple[float, float, float] = lengths
        self.angles : tuple[float, float, float] = angles

        self.get_structure()
        self.get_fingerprints()

    @classmethod
    def from_custom_structure(cls, structure : CrystalStructure) -> VAECrystal:
        lengths = structure.lengths
        angles = structure.angles

        frac_coords = []
        atom_types = []
        for site in structure.basis.atom_sites:
            x,y,z = site.x, site.y, site.z
            frac_coords.append((x,y,z))
            atom_types.append(site.atom.as_pymatgen)

        return VAECrystal(lengths=lengths, angles=angles, frac_coords=frac_coords, atom_types=atom_types)

    def get_structure(self):
        if min(self.lengths) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:

                self.structure = Structure(
                    lattice=Lattice.from_parameters(*self.lengths, *self.angles),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_fingerprints(self):
        # print(f'- Atom types: {self.atom_types}')
        # print(f'- Frac coords: {self.frac_coords}')

        start_time = time.time()
        try:
            site_fps = [CrystalNNFP.featurize(self.structure, i) for i in range(len(self.structure))]
            print(f'- Finished fingerprint creation')
            print(f'- Size of structure = {len(self.structure)}')
            print(f'- Number of atoms: {len(self.structure.sites)}')

        except Exception as e:
            print(f'- CrystalNNFP failed:{e}')
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.struct_fp = None
            return
        self.struct_fp : NDArray = np.array(site_fps).mean(axis=0)
        print(f'time taken = {time.time()-start_time}')


class FingerprintProcessor:
    @staticmethod
    def load_fingerprint_map(fpath : str) -> dict:
        with open(fpath, 'r') as f:
            content = f.read()
            fingerprint = json.loads(content)
            fingerprint['done'] = set(fingerprint['done'])

            return fingerprint

    @staticmethod
    def save_fingerprint_map(fpath : str, state : dict):
        with open(fpath, 'w') as f:
            state['done'] = list(state['done'])
            content = json.dumps(state)
            f.write(content)

    @staticmethod
    def process_icsd_fingerprint(dirpath : str):
        map_fpath = '/home/daniel/aimat/data/fingerprint_map.txt'
        if os.path.isfile(map_fpath):
            fingerprint_map = FingerprintProcessor.load_fingerprint_map(fpath=map_fpath)
        else:
            fingerprint_map = {'done' : set()}

        for j, fname in enumerate(os.listdir(dirpath)):
            if fname in fingerprint_map['done']:
                print(f'- Skipping {fname} as already processed')
                continue

            fpath = os.path.join(dirpath, fname)
            with open(fpath, 'r') as f:
                content = f.read()
                try:
                    crystal_structure = CrystalStructure.from_cif(cif_content=content)
                except:
                    print(f'- Failed to parse {fname} as cif')
                    continue

            crystal = VAECrystal.from_custom_structure(crystal_structure)
            fingerprint = crystal.struct_fp

            if fingerprint is None:
                continue

            fingerprint_map[fname] = fingerprint.tolist()
            fingerprint_map['done'].add(fname)

            FingerprintProcessor.save_fingerprint_map(fpath=map_fpath, state=copy.deepcopy(fingerprint_map))
            print(f'- Successfully processed ICSD crystal. Loop counter = {j}')

    @staticmethod
    def get_fingerprints(structures : list[CrystalStructure]):
        vae_crystals = [VAECrystal.from_custom_structure(struct) for struct in structures]
        fingerprints = [c.struct_fp for c in vae_crystals]

        return fingerprints


if __name__ == '__main__':
    icsd_dirpath = '/home/daniel/aimat/data/cif'
    FingerprintProcessor.process_icsd_fingerprint(dirpath=icsd_dirpath)