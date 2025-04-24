from __future__ import annotations

from collections import Counter

import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.core import Species
from pymatgen.core.composition import Composition
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
        self.get_composition()
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

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


if __name__ == '__main__':
    from xrdpattern.crystal import CrystalExamples
    example_crystal = CrystalExamples.get_crystal()
    crystal = VAECrystal.from_custom_structure(example_crystal)