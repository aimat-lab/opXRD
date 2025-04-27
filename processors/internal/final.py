import logging
import os
from logging import Logger
from typing import Optional

import pandas as pd
from xrdpattern.crystal import CrystalStructure
from xrdpattern.parsing.path_tools import PathTools
from xrdpattern.pattern import PatternDB
from xrdpattern.xrd import PowderExperiment, XrayInfo

from processors.internal.csv_label import get_powder_experiment, get_label_mapping
from processors.internal.methods import ModuleInspector


# -------------------------------------------

class FinalProcessor:
    def __init__(self, root_dirpath : str):
        self.root_dirpath : str = root_dirpath
        self.prepared_dirpath : str = os.path.join(root_dirpath, 'prepared')
        self.final_dirpath : str = os.path.join(root_dirpath, 'final')
        self.cu_xray : XrayInfo = XrayInfo.copper_xray()
        self.logger : Logger = logging.getLogger(name=__name__)
        self.logger.level = logging.INFO

    # ---------------------------------------
    # Parsing individual contributions

    def parse_all(self):
        methods = ModuleInspector.get_methods(self)
        parse_methods = [m for m in methods if not m.__name__.endswith('all') and 'parse' in m.__name__]

        for mthd in parse_methods:
            print(f'- Running method {mthd.__name__}')
            mthd()
            print(f'+--------------------------------------+\n')

    def get_csv_db(self, dirname: str, orientation : str, suffixes: Optional[list[str]] = None) -> PatternDB:
        return self.get_db(dirname=dirname, csv_orientation=orientation, suffixes=suffixes)

    def get_db(self, dirname: str,
               suffixes : Optional[list[str]] = None,
               xray_info : Optional[XrayInfo] = None,
               csv_orientation : Optional[str] = None,
               strict : bool = False) -> PatternDB:
        print(f'Started processing contribution {dirname}')
        contrib_dirpath = os.path.join(self.prepared_dirpath, dirname)
        contrib_data = os.path.join(self.prepared_dirpath, dirname, 'ready')
        pattern_db = PatternDB.load(dirpath=contrib_data, suffixes=suffixes, csv_orientation=csv_orientation, strict=strict)

        self.attach_metadata(pattern_db, dirname=dirname)
        self.attach_csv_labels(pattern_db=pattern_db, contrib_dirpath=contrib_dirpath)
        if xray_info:
            pattern_db.set_xray(xray_info=xray_info)
        for p in pattern_db.patterns:
            p.metadata.remove_filename()
            standardized_phases = [phase.get_standardized() for phase in p.powder_experiment.phases]
            p.powder_experiment.phases = standardized_phases
        print(f'Finished processing contribution {dirname}')

        return pattern_db

    # ---------------------------------------
    # Parsing steps

    def attach_metadata(self, pattern_db : PatternDB, dirname : str):
        form_dirpath = os.path.join(self.prepared_dirpath, dirname, 'form.txt')
        with open(form_dirpath, "r") as file:
            lines = file.readlines()
        form_data = {}
        for line in lines:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                form_data[key] = value

        for p in pattern_db.patterns:
            p.metadata.contributor_name = form_data["name_of_advisor"]
            p.metadata.institution = form_data["contributing_institution"]


    @staticmethod
    def attach_cif_labels(pattern_db : PatternDB):
        for fpath, patterns in pattern_db.fpath_dict.items():
            dirpath = os.path.dirname(fpath)
            cif_fnames = [fname for fname in os.listdir(dirpath) if PathTools.get_suffix(fname) == 'cif']

            phases = []
            for fname in cif_fnames:
                cif_fpath = os.path.join(dirpath, fname)
                attached_cif_content = FinalProcessor.read_file(fpath=cif_fpath)
                crystal_phase = FinalProcessor.safe_cif_read(cif_content=attached_cif_content)
                phases.append(crystal_phase)

            phases = [p for p in phases if not p is None]
            powder_experiment = PowderExperiment.from_multi_phase(phases=phases)
            for p in patterns:
                p.powder_experiment = powder_experiment


    @staticmethod
    def attach_csv_labels(pattern_db : PatternDB, contrib_dirpath : str):
        csv_fpath = os.path.join(contrib_dirpath, 'labels.csv')

        if not os.path.isfile(csv_fpath):
            print(f'- Detected no CSV labels for contribution {os.path.basename(contrib_dirpath)}')
            return

        for p in pattern_db.patterns:
            if p.is_labeled:
                raise ValueError(f"Pattern {p.get_name()} is already labeled")

        data = pd.read_csv(csv_fpath, skiprows=1)
        phases = [get_label_mapping(data=data, phase_num=num) for num in range(2)]
        for pattern_fpath, file_patterns in pattern_db.fpath_dict.items():
            powder_experiment = get_powder_experiment(pattern_fpath=pattern_fpath, contrib_dirpath=contrib_dirpath, phases=phases)

            for p in file_patterns:
                p.powder_experiment = powder_experiment


    def save(self, pattern_db : PatternDB, dirname : str):
        out_dirpath = os.path.join(self.final_dirpath, dirname)
        if not os.path.isdir(out_dirpath):
            os.makedirs(out_dirpath)
        pattern_db.save(dirpath=out_dirpath)

    # -----------------------------
    # Helper methods

    @staticmethod
    def read_file(fpath: str) -> str:
        with open(fpath, 'r') as file:
            cif_content = file.read()
        return cif_content

    @staticmethod
    def safe_cif_read(cif_content: str) -> Optional[CrystalStructure]:
        try:
            extracted_phase = CrystalStructure.from_cif(cif_content)
        except:
            extracted_phase = None
        return extracted_phase

    def get_final_dirpath(self, *path_elements : str):
        return os.path.join(self.final_dirpath, *path_elements)