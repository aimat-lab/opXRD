from __future__ import annotations
import warnings
import copy
from tqdm import tqdm
import json
import os
import time
import argparse
from typing import Optional

import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from numpy.typing import NDArray
from pymatgen.core import Species
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from xrdpattern.crystal import CrystalStructure
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------------------------------------------------------

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")

Percentiles = {
    "mp20": np.array([-3.17562208, -2.82196882, -2.52814761]),
    "carbon": np.array([-154.527093, -154.45865733, -154.44206825]),
    "perovskite": np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    "mp20": {"struc": 0.4, "comp": 10.0},
    "carbon": {"struc": 0.2, "comp": 4.0},
    "perovskite": {"struc": 0.2, "comp": 4},
}


class VAECrystal(object):
    def __init__(self, frac_coords, atom_types, lengths, angles):
        self.frac_coords: list = frac_coords
        self.atom_types: list[Species] = atom_types
        self.lengths: tuple[float, float, float] = lengths
        self.angles: tuple[float, float, float] = angles

        self.structure: Structure = self.get_structure()

    @classmethod
    def from_custom_structure(cls, structure: CrystalStructure) -> VAECrystal:
        lengths = structure.lengths
        angles = structure.angles

        frac_coords = []
        atom_types = []
        for site in structure.basis.atom_sites:
            x, y, z = site.x, site.y, site.z
            frac_coords.append((x, y, z))
            atom_types.append(site.atom.as_pymatgen)

        return VAECrystal(
            lengths=lengths,
            angles=angles,
            frac_coords=frac_coords,
            atom_types=atom_types,
        )

    def get_structure(self) -> Structure:
        lattice = Lattice.from_parameters(*self.lengths, *self.angles)
        return Structure(
            lattice=lattice,
            species=self.atom_types,
            coords=self.frac_coords,
            coords_are_cartesian=False,
        )

    def get_fingerprint(self) -> Optional[NDArray]:
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
            # print(f"- Finished fingerprint creation")
            # print(f"- Size of structure = {len(self.structure)}")
            # print(f"- Number of atoms: {len(self.structure.sites)}")

        except Exception as e:
            print(f"- CrystalNNFP failed:{e}")
            return None

        # print(f"time taken = {time.time()-start_time}")
        return np.array(site_fps).mean(axis=0)


class FingerprintProcessor:
    @staticmethod
    def load_fingerprint_map(fpath: str) -> dict:
        with open(fpath, "r") as f:
            content = f.read()
            fingerprint = json.loads(content)
            fingerprint["done"] = set(fingerprint["done"])

            return fingerprint

    @staticmethod
    def save_fingerprint_map(fpath: str, state: dict):
        with open(fpath, "w") as f:
            state["done"] = list(state["done"])
            content = json.dumps(state)
            f.write(content)

    @staticmethod
    def process_icsd_fingerprint(fpath: str):
        try:
            with open(fpath, "r") as f:
                content = f.read()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    crystal_structure = CrystalStructure.from_cif(cif_content=content)
        except Exception as e:
            print(f"- Failed to read cif file {fpath}: {e}")
            return None

        try:
            crystal = VAECrystal.from_custom_structure(crystal_structure)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fingerprint = crystal.get_fingerprint()
        except Exception as e:
            print(f"- Failed to create fingerprint for {fpath}: {e}")
            return None

        return fingerprint

    @staticmethod
    def process_icsd_fingerprints(
        dirpath: str,
        map_fpath: str | None = None,
        n_workers: int = 8,
        save_every: int = 1000,
    ):
        if map_fpath is not None and os.path.isfile(map_fpath):
            fingerprint_map = FingerprintProcessor.load_fingerprint_map(fpath=map_fpath)
        else:
            fingerprint_map = {"done": set()}

        fpaths = [
            os.path.join(dirpath, fname)
            for fname in os.listdir(dirpath)
            if fname not in fingerprint_map["done"]
        ]

        print(
            f"Using {n_workers} workers to process {len(fpaths)} remaining cif files, saving every {save_every} cifs"
        )

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    FingerprintProcessor.process_icsd_fingerprint, fpath
                ): fpath
                for fpath in fpaths
            }

            with tqdm(total=len(fpaths), desc="Processing structures") as pbar:
                i = 0
                for future in as_completed(futures.keys()):
                    fpath = futures[future]
                    try:
                        fingerprint = future.result()
                    except Exception as e:
                        print(f"- Error processing {fpath}: {e}")
                        fingerprint = None

                    fname = os.path.basename(fpath)
                    fingerprint_map[fname] = (
                        fingerprint.tolist() if fingerprint is not None else None
                    )
                    fingerprint_map["done"].add(fname)

                    i += 1

                    if i % save_every == 0:
                        print("Saving fingerprint map...")
                        FingerprintProcessor.save_fingerprint_map(
                            fpath=map_fpath, state=copy.deepcopy(fingerprint_map)
                        )

                    pbar.update(1)

    @staticmethod
    def get_fingerprints(structures: list[CrystalStructure]):
        vae_crystals = [
            VAECrystal.from_custom_structure(struct) for struct in structures
        ]
        fingerprints = [c.get_fingerprint() for c in vae_crystals]

        return fingerprints


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--icsd_dirpath",
        type=str,
        required=True,
        help="Path to the directory containing ICSD cif files.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        required=True,
        help="Number of workers to use for processing.",
    )

    args = parser.parse_args()

    map_fpath = "./fingerprint_map.txt"
    save_every = 1000

    FingerprintProcessor.process_icsd_fingerprints(
        dirpath=args.icsd_dirpath,
        map_fpath=map_fpath,
        n_workers=args.n_workers,
        save_every=save_every,
    )
