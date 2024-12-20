import math
import os
import random
import sys

import numpy as np
from IPython.core.display import Markdown
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from tabulate import tabulate

from holytools.devtools import Profiler
from xrdpattern.pattern import PatternDB
from xrdpattern.pattern import XrdPattern
from xrdpattern.xrd import LabelType

# -----------------------------------------

class DatabaseAnalyser:
    def __init__(self, databases: list[PatternDB], output_dirpath: str):
        if len(databases) == 0:
            raise ValueError('No databases provided')
        self.databases: list[PatternDB] = databases
        self.joined_db: PatternDB = PatternDB.merge(databases)
        self.output_dirpath: str = output_dirpath
        os.makedirs(self.output_dirpath, exist_ok=True)

        random.seed(42)

    def plot_in_single(self, limit_patterns: int):
        lower_alphabet = [chr(i) for i in range(97, 123)]
        explanation = [f'{letter}:{db.name}' for letter, db in zip(lower_alphabet, self.databases)]
        self.print_text(f'---> Combined pattern plot for databaes {explanation} | No. patterns = {limit_patterns}')

        lower_alphabet = [chr(i) for i in range(97, 123)]
        save_fpath = os.path.join(self.output_dirpath, f'ALL_pattern_multiplot.png')

        cols = 3
        rows = math.ceil(len(self.databases) / cols)
        num_plots = len(self.databases)
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        axes = []
        for i in range(num_plots):
            if i != 0:
                ax = fig.add_subplot(rows, cols, i + 1, sharex=axes[0], sharey=axes[0])
            else:
                ax = fig.add_subplot(rows, cols, i + 1)
            axes.append(ax)

        for letter, ax, database in zip(lower_alphabet, axes, self.databases):
            patterns = database.patterns[:limit_patterns]
            data = [p.get_pattern_data() for p in patterns]

            for x, y in data:
                ax.plot(x, y, linewidth=0.25, alpha=0.75, linestyle='--')
            title = f'{letter})'

            if title:
                ax.set_title(title, loc='left')

        fig.supylabel('Standardized relative intensity (a.u.)')
        fig.supxlabel(r'$2\theta$ [$^\circ$]', ha='center')

        plt.tight_layout()
        plt.savefig(f'{save_fpath}')
        plt.show()


    def plot_reference_fourier(self, b1: float = 0.3, b2: float = 0.5, c = 2):
        x = np.linspace(0, 180, num=1000)
        self.print_text(r'---> Fourier transform of a pair of gaussians $I(x) = e^{{-0.5(x-b)^2/c}$')
        c1, c2 = 0.1, 0.2

        y = 5* np.exp(-1 / 2 * (x - b1) ** 2 / c1) + np.exp(-1 / 2 * (x - b2) ** 2 / c2)
        xf, yf = self.compute_fourier(x, y)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Gaussian plot
        ax1.plot(x, y, label='Original Gaussian')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Original Gaussian')

        # Fourier Transform plot
        ax2.plot(xf, yf, label='Fourier Transform', color='r')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('|F(k)|')
        ax2.set_title('Fourie|r Transform')

        plt.tight_layout()
        plt.show()


    def plot_opxrd_fourier(self):
        for db in self.databases:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

            db_intensities = [p.get_pattern_data()[1] for p in db.patterns]

            intensity_sum = np.sum(db_intensities, axis=0)
            x, _ = db.patterns[0].get_pattern_data()
            xf, yf = self.compute_fourier(x, intensity_sum)

            plt.plot(xf, yf)
            ax.set_title(f'{db.name} patterns summed up fourier transform ' + r'$F(k)=\int d(2\theta) I(2\theta) e^{-ik2\theta}$')
            ax.set_xlabel(r'k [deg$^{−1}$]')
            ax.set_ylabel('l|F($k$)| (a.u.)')

            plt.savefig(os.path.join(self.output_dirpath, f'{db.name}_fourier.png'))
            plt.show()


    def plot_effective_components(self, use_fractions : bool = True):
        self.print_text(r'Cumulative explained variance ratio $v$ over components '
                        r'|  $v =  \frac{\sum_i \lambda_i}{\sum^n_{j=1} \lambda_j}$')

        for db_num, db in enumerate(self.databases):
            max_components = min(len(db.patterns), XrdPattern.std_num_entries())
            standardized_intensities = np.array([p.get_pattern_data()[1] for p in db.patterns])
            print(f'[Debug]: Performing PCA for {db.name} | No. patterns = {len(standardized_intensities)}')
            pca = PCA(n_components=max_components)
            pca.fit_transform(standardized_intensities)

            accuracies = []
            print(f'Max components for {db.name} = {max_components}')
            components_list = range(0, max_components)

            x_axis = components_list if not use_fractions else np.linspace(0,1, num=max_components)
            for x in x_axis:
                if use_fractions:
                    n_comp = int(x * max_components)
                else:
                    n_comp = x
                explained_variance = np.sum(pca.explained_variance_ratio_[:n_comp])
                accuracies.append(explained_variance)
            plt.plot(x_axis, accuracies, label=db.name)


        if use_fractions:
            plt.xlabel(f'Fraction of max. No. components')
        else:
            plt.xlabel(f'No. components')
        plt.xscale(f'log')
        plt.ylabel(f'Cumulative explained variance ratio $V$')

        # locator = LogLocator(base=10.0, subs=(1.0,), numticks=10)
        # plt.gca().xaxis.set_major_locator(locator)

        # plt.xlim(1, 300)
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.output_dirpath, f'ALL_effective_components.png'))

        plt.show()


    def plot_histogram(self, attach_colorbar : bool = False):
        self.print_text(f'---> Histogram of general information on opXRD')
        self.joined_db.show_histograms(save_fpath=os.path.join(self.output_dirpath, 'ALL_histogram.png'), attach_colorbar=attach_colorbar)


    def show_label_fractions(self):
        self.print_text(f'---> Overview of label fractions per contribution')
        table_data = []
        for d in self.databases:
            label_counts = {l: 0 for l in LabelType}
            patterns = d.patterns
            for l in LabelType:
                for p in patterns:
                    if p.has_label(label_type=l):
                        label_counts[l] += 1
            db_percentages = [label_counts[l] / len(patterns) for l in LabelType]
            table_data.append(db_percentages)

        col_headers = [label.name for label in LabelType]
        row_headers = [db.name for db in self.databases]

        table = tabulate(table_data, headers=col_headers, showindex=row_headers, tablefmt='psql')
        print(table)


    def print_total_counts(self):
        self.print_text(f'---> Total pattern counts in opXRD')
        num_total = len(self.joined_db.patterns)

        labeled_patterns = [p for p in self.joined_db.patterns if p.is_labeled()]
        num_labelel = len(labeled_patterns)
        print(f'Total number of patterns = {num_total}')
        print(f'Number of labeled patterns = {num_labelel}')















    # -----------------------
    # tools

    @staticmethod
    def print_text(msg: str):
        if 'ipykernel' in sys.modules:
            display(Markdown(msg))
        else:
            print(msg)


    @staticmethod
    def compute_fourier(x: NDArray, y : NDArray):
        N = len(y)
        T = (x[-1] - x[0]) / (N - 1)

        yf = np.fft.fft(y)
        xf = np.fft.fftfreq(N, T)[:N // 2]
        yf = 2.0 / N * np.abs(yf[:N // 2])
        return xf, yf


if __name__ == "__main__":
    from opxrd.wrapper import OpXRD
    smoltest_dirpath = '/home/daniel/aimat/data/opXRD/test_smol'
    bigtest_dirpath = '/home/daniel/aimat/data/opXRD/test'
    test_databases = OpXRD.load_project_list(root_dirpath=smoltest_dirpath)

    analyser = DatabaseAnalyser(databases=test_databases, output_dirpath='/tmp/opxrd_analysis')
    analyser.plot_reference_fourier(b1=60, b2=80)