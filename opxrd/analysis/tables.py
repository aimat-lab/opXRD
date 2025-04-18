import os
import random

from tabulate import tabulate

from opxrd.analysis.tools import print_text
from xrdpattern.pattern import PatternDB
from xrdpattern.xrd import LabelType


# ---------------------------------------------------

class TableAnalyser:
    def __init__(self, databases: list[PatternDB], output_dirpath: str):
        if len(databases) == 0:
            raise ValueError('No databases provided')
        self.databases: list[PatternDB] = databases
        self.joined_db: PatternDB = PatternDB.merge(databases)
        self.output_dirpath: str = output_dirpath
        os.makedirs(self.output_dirpath, exist_ok=True)
        random.seed(42)


    def show_label_fractions(self):
        print_text(f'---> Overview of label fractions per contribution')
        table_data = []
        for d in self.databases:
            label_counts = {l: 0 for l in LabelType}
            patterns = d.patterns
            for l in LabelType:
                for p in patterns:
                    if p.powder_experiment.has_label(label_type=l):
                        label_counts[l] += 1
            row = [len(d.patterns)] + [label_counts[l] / len(patterns) for l in LabelType]
            table_data.append(row)

        col_headers = ['No. patterns'] +  [label.name for label in LabelType]
        row_headers = [db.name for db in self.databases]


        table = tabulate(table_data, headers=col_headers, showindex=row_headers, tablefmt='psql')
        print(table)
        print(f'total patterns = {sum([len(d.patterns) for d in self.databases])}')


    def print_total_counts(self):
        print_text(f'---> Total pattern counts in opXRD')
        num_total = len(self.joined_db.patterns)

        labeled_patterns = [p for p in self.joined_db.patterns if p.powder_experiment.is_labeled()]
        num_labelel = len(labeled_patterns)
        print(f'Total number of patterns = {num_total}')
        print(f'Number of labeled patterns = {num_labelel}')
