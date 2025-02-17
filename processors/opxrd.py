import os
from logging import Logger
from typing import Optional

import pandas as pd

from holytools.logging import LoggerFactory
from special.tools.csv_label import get_powder_experiment, get_label_mapping
from holytools.devtools import ModuleInspector
from holytools.fsys import PathTools
from holytools.logging.tools import log_execution
from xrdpattern.crystal import CrystalPhase
from xrdpattern.pattern import PatternDB
from xrdpattern.xrd import PowderExperiment, XrayInfo, XrdAnode

# -------------------------------------------

class OpXRDProcessor:
    def __init__(self, root_dirpath : str):
        self.root_dirpath : str = root_dirpath
        self.processed_dirpath : str = os.path.join(root_dirpath, 'processed')
        self.final_dirpath : str = os.path.join(root_dirpath, 'final')
        self.cu_xray : XrayInfo = XrdAnode.Cu.get_xray_info()
        self.logger : Logger = LoggerFactory.get_logger(name=__name__)

    # ---------------------------------------
    # Parsing individual contributions


    def parse_all(self):
        methods = ModuleInspector.get_methods(self)
        parse_methods = [m for m in methods if not m.__name__.endswith('all') and 'parse' in m.__name__]

        for mthd in parse_methods:
            print(f'mthd name = {mthd.__name__}')
            mthd()

    def get_db(self, dirname: str,
               suffixes : Optional[list[str]] = None,
               use_cif_labels : bool = False,
               xray_info : Optional[XrayInfo] = None,
               csv_orientation : Optional[str] = None,
               strict : bool = False) -> PatternDB:
        self.logger.info(f'Started processing contribution {dirname}')
        data_dirpath = os.path.join(self.processed_dirpath, dirname, 'data')
        contrib_dirpath = os.path.join(self.processed_dirpath, dirname)
        pattern_db = PatternDB.load(dirpath=data_dirpath, suffixes=suffixes, csv_orientation=csv_orientation, strict=strict)

        self.attach_metadata(pattern_db, dirname=dirname)
        self.attach_labels(pattern_db=pattern_db, contrib_dirpath=contrib_dirpath, use_cif_labels=use_cif_labels)
        if xray_info:
            pattern_db.set_xray(xray_info=xray_info)
        for p in pattern_db.patterns:
            p.metadata.remove_filename()
        self.logger.info(f'Finished processing contribution {dirname}')

        return pattern_db

    def get_csv_db(self, dirname: str, orientation : str, suffixes: Optional[list[str]] = None) -> PatternDB:
        return self.get_db(dirname=dirname, csv_orientation=orientation, suffixes=suffixes, strict=False)

    # ---------------------------------------
    # Parsing steps

    @log_execution
    def attach_metadata(self, pattern_db : PatternDB, dirname : str):
        form_dirpath = os.path.join(self.processed_dirpath, dirname, 'form.txt')
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


    def attach_labels(self, pattern_db : PatternDB, contrib_dirpath: str, use_cif_labels : bool):
        if use_cif_labels:
            self.attach_cif_labels(pattern_db)
        else:
            self.attach_csv_labels(pattern_db, contrib_dirpath=contrib_dirpath)

    @log_execution
    def attach_cif_labels(self, pattern_db : PatternDB):
        for fpath, patterns in pattern_db.fpath_dict.items():
            dirpath = os.path.dirname(fpath)
            cif_fnames = [fname for fname in os.listdir(dirpath) if PathTools.get_suffix(fname) == 'cif']

            phases = []
            for fname in cif_fnames:
                cif_fpath = os.path.join(dirpath, fname)
                attached_cif_content = OpXRDProcessor.read_file(fpath=cif_fpath)
                crystal_phase = OpXRDProcessor.safe_cif_read(cif_content=attached_cif_content)
                phases.append(crystal_phase)

            phases = [p for p in phases if not p is None]
            powder_experiment = PowderExperiment.from_multi_phase(phases=phases)
            for p in patterns:
                p.powder_experiment = powder_experiment


    @log_execution
    def attach_csv_labels(self, pattern_db : PatternDB, contrib_dirpath : str):
        csv_fpath = os.path.join(contrib_dirpath, 'labels.csv')

        if not os.path.isfile(csv_fpath):
            print(f'No labels available for contribution {os.path.basename(contrib_dirpath)}')
            return

        for p in pattern_db.patterns:
            if p.powder_experiment.is_nonempty():
                raise ValueError(f"Pattern {p.get_name()} is already labeled")

        data = pd.read_csv(csv_fpath, skiprows=1)
        phases = [get_label_mapping(data=data, phase_num=num) for num in range(2)]
        for pattern_fpath, file_patterns in pattern_db.fpath_dict.items():
            powder_experiment = get_powder_experiment(pattern_fpath=pattern_fpath, contrib_dirpath=contrib_dirpath, phases=phases)

            for p in file_patterns:
                p.powder_experiment = powder_experiment


    @log_execution
    def save(self, pattern_db : PatternDB, dirname : str):
        out_dirpath = os.path.join(self.final_dirpath, dirname)
        if not os.path.isdir(out_dirpath):
            os.makedirs(out_dirpath)
        pattern_db.save(dirpath=out_dirpath, force_overwrite=True)

    # -----------------------------
    # Helper methods

    @staticmethod
    def read_file(fpath: str) -> str:
        with open(fpath, 'r') as file:
            cif_content = file.read()
        return cif_content

    @staticmethod
    def safe_cif_read(cif_content: str) -> Optional[CrystalPhase]:
        try:
            extracted_phase = CrystalPhase.from_cif(cif_content)
        except:
            extracted_phase = None
        return extracted_phase

    def get_final_dirpath(self, *path_elements : str):
        return os.path.join(self.final_dirpath, *path_elements)