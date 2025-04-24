import os
import random

from IPython.core.display import Markdown
from IPython.core.display_functions import display
from tabulate import tabulate

from xrdpattern.pattern import PatternDB, XrdPattern
from xrdpattern.xrd import LabelType


# ---------------------------------------------------

class TableAnalyser:
    def __init__(self, databases: list[PatternDB], output_dirpath: str):
        if len(databases) == 0:
            raise ValueError('No databases provided')
        self.databases: list[PatternDB] = databases
        self.joined_db: PatternDB = PatternDB.merge(databases)
        self.output_dirpath: str = output_dirpath
        self.patterns : list[XrdPattern] = self.joined_db.patterns

        self.unlabeled : list[XrdPattern] = []
        self.labeled : list[XrdPattern] = []
        self.fully_labeled : list[XrdPattern] = []

        for p in self.patterns:
            if p.is_labeled:
                self.labeled.append(p)
                if p.has_label(label_type=LabelType.basis):
                    self.fully_labeled.append(p)
            else:
                self.unlabeled.append(p)


        os.makedirs(self.output_dirpath, exist_ok=True)
        random.seed(42)

    def show_label_fractions(self):
        self.print_text(f'---> Overview of label fractions per contribution')
        table_data = []
        for d in self.databases:
            row = self.get_label_row(d=d)
            table_data.append(row)

        col_headers = ['No. patterns'] +  [label.name for label in LabelType.get_main_labels()]
        row_headers = [db.name for db in self.databases]

        labeled_db = PatternDB(patterns=self.labeled, fpath_dict={})
        unlabeled_db = PatternDB(patterns=self.unlabeled, fpath_dict={})

        table_data.append(self.get_label_row(d=labeled_db))
        table_data.append(self.get_label_row(d=unlabeled_db))
        row_headers += ['Σ Labeled', 'Σ Unlabeled']

        table = tabulate(table_data, headers=col_headers, showindex=row_headers, tablefmt='psql')
        print(table)

    @staticmethod
    def get_label_row(d : PatternDB):
        label_counts = {l: 0 for l in LabelType.get_main_labels()}
        patterns = d.patterns
        for l in LabelType.get_main_labels():
            for p in patterns:
                if p.has_label(label_type=l):
                    label_counts[l] += 1
        return [len(d.patterns)] + [label_counts[l] / len(patterns) if len(patterns) > 0 else '#' for l in LabelType.get_main_labels()]


    @staticmethod
    def print_text(msg: str):
        try:
            display(Markdown(msg))
        except:
            print(msg)
