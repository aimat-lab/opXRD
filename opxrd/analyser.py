import os
import random

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from tabulate import tabulate

from xrdpattern.pattern import PatternDB
from xrdpattern.pattern import XrdPattern
from matplotlib import pyplot as plt

from opxrd import OpXRD
from xrdpattern.xrd import LabelType

from holytools.devtools import Profiler

profiler = Profiler()

# ----------------------------------------------------------

class DatabaseAnalyser:
    def __init__(self, databases : list[PatternDB], output_dirpath : str):
        if len(databases) == 0:
            raise ValueError('No databases provided')
        self.databases : list[PatternDB] = databases
        self.joined_db : PatternDB = PatternDB.merge(databases)
        self.output_dirpath : str = output_dirpath
        os.makedirs(self.output_dirpath, exist_ok=True)

        random.seed(42)

    def plot_databases_in_single(self):
        for database in self.databases:
            database.show_all(single_plot=True, limit_patterns=10)

    def plot_fourier(self, max_freq=2):
        for db in self.databases:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
            ref_x, _ = db.patterns[0].get_pattern_data()
            ref_y = np.zeros(shape=len(ref_x))

            size = len(ref_y)
            mean = size // 2
            std_dev = 200

            x = np.linspace(0, size - 1, size)
            gaussian = np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))
            noise = np.random.normal(0, 0.025, size)
            noisy_gaussian = gaussian + noise
            ref_y += noisy_gaussian


            for p in db.patterns[:10]:
                x,y = p.get_pattern_data()
                plt.plot(x, y, linewidth=0.75, linestyle='--', alpha=0.75)
            plt.plot(ref_x, ref_y, alpha=0.75)
            plt.show()

            for p in db.patterns[:10]:
                x,y = p.get_pattern_data()
                xf, yf = self.compute_fourier_transform(x, y, max_freq)

                self._set_ax_properties(ax, title='Fourier Transform', xlabel='Frequency (Hz)', ylabel='Magnitude')
                ax.grid(True)
                plt.plot(xf, yf, linewidth=0.75, linestyle='--', alpha=0.75)
            ft_ref_x, ft_ref_y = self.compute_fourier_transform(ref_x, ref_y, max_freq)
            plt.plot(ft_ref_x, ft_ref_y, alpha=0.75)

            plt.show()

    def plot_pattern_dbs(self, title : str):
        combined_pattern_list = self.get_all_patterns()
        combined_intensities_list = [p.get_pattern_data()[1] for p in combined_pattern_list]

        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(combined_intensities_list)

        rand_indices = [np.random.randint(low=0, high=len(combined_intensities_list)) for _ in range(10)]
        example_xy_list = [combined_pattern_list[idx].get_pattern_data() for idx in rand_indices]
        example_pca_coords =  [transformed_data[idx] for idx in rand_indices]

        self._plot_pca_scatter(transformed_data, title=title)
        self._plot_pca_basis(pca, title=title)
        self._plot_reconstructed(pca, example_xy_list, example_pca_coords, title=title)
        print('done')

    def plot_effective_components(self):
        for db in self.databases:
            max_components = len(db.patterns)
            standardized_intensities = [p.get_pattern_data()[1] for p in db.patterns]
            pca = PCA(n_components=max_components)
            db_pca_coords = pca.fit_transform(standardized_intensities)

            plot_components = min(10, max_components)
            accuracies = []
            x = np.linspace(0,plot_components/max_components, num=plot_components)
            for n_comp in range(plot_components):
                mismatches = []
                limited_pca = db_pca_coords[:,:n_comp]
                zero_padded_comp = np.pad(limited_pca, ((0, 0), (0, max_components - n_comp)))
                reconstructions = pca.inverse_transform(zero_padded_comp)
                for i1, i2 in zip(standardized_intensities, reconstructions):
                    mismatch = self.compute_mismatch(i1, i2)
                    mismatches.append(mismatch)

                mismatch  = np.mean(mismatches)
                accuracies.append(mismatch)
                print(f'Computed accuracy for {db.name} with {n_comp} components = {mismatch}')

            plt.plot(x,accuracies, label=db.name)

        plt.xlabel(f'Fraction of maximum components')
        plt.ylabel(f'Average relative mismatch $/lange \delta /rangle$')
        plt.legend(loc='lower right')

        plt.show()

    # -----------------------
    # tools

    def _plot_pca_scatter(self, transformed_data, title : str):
        db_lens = [len(db.patterns) for db in self.databases]
        max_points = 50
        for j, l in enumerate(db_lens):
            partial = transformed_data[:l]
            if l > max_points:
                indices = np.random.choice(len(partial), size=max_points, replace=False)
                partial = partial[indices]
            plt.scatter(partial[:, 0], partial[:, 1], label=f'db number {j}')
            transformed_data = transformed_data[l:]

        plt.title(f'(1): Two Component PCA Scatter Plot for {title}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.savefig(f'{self.output_dirpath}pca_scatter_{title}.png')
        plt.show()

    def _plot_pca_basis(self, pca, title : str):
        b1, b2 = pca.inverse_transform(np.array([1,0])), pca.inverse_transform(np.array([0,1]))
        x = np.linspace(start=0,stop=180, num=len(b1))
        plt.plot(x, b1)
        plt.plot(x, b2)
        plt.title(f'(2): Principal Components for {title}')
        plt.savefig(f'{self.output_dirpath}pca_basis_{title}.png')
        plt.show()

    def _plot_reconstructed(self, pca, example_xy_list, example_pca_coords, title):
        fig, axs = plt.subplots(len(example_xy_list), 2, figsize=(10, 5 * len(example_xy_list)))
        fig.suptitle(f'(3): Comparison of Original and Reconstructed Patterns for {title}', fontsize=16)
        for index, ((x1, y1), pca_coords) in enumerate(zip(example_xy_list, example_pca_coords)):
            axs[index, 0].plot(x1, y1, 'b-')
            self._set_ax_properties(axs[index, 0], title='Original pattern', xlabel='x', ylabel='Relative intensity')

            reconstructed = pca.inverse_transform(pca_coords)
            x = np.linspace(start=0, stop=180, num=len(reconstructed))
            axs[index, 1].plot(x, reconstructed, 'r-')
            self._set_ax_properties(axs[index, 1], title='Reconstructed pattern', xlabel='x',
                                    ylabel='Relative intensity')

        plt.tight_layout()
        plt.savefig(f'{self.output_dirpath}reconstructed_{title}.png')
        plt.show()

    @staticmethod
    def _set_ax_properties(ax : Axes, title : str, xlabel : str, ylabel : str):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def get_all_patterns(self) -> list[XrdPattern]:
        return self.joined_db.patterns

    @staticmethod
    def compute_fourier_transform(x,y, max_freq : float):
        N = len(y)  # Number of sample points
        T = (x[-1] - x[0]) / (N - 1)  # Sample spacing
        yf = np.fft.fft(y)  # Perform FFT
        xf = np.fft.fftfreq(N, T)[:N // 2]  # Frequency axis

        magnitude = 2.0 / N * np.abs(yf[:N // 2])
        valid_indices = xf <= max_freq

        xf = xf[valid_indices]
        yf = magnitude[valid_indices]
        return xf, yf

    @staticmethod
    def compute_mismatch(i1 : NDArray, i2 : NDArray) -> float:
        norm_original = np.linalg.norm(i1) / len(i1)
        delta_norm = np.linalg.norm(i1 - i2)/len(i1)
        mismatch = delta_norm / norm_original
        return mismatch

    def show_label_fractions(self):
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
        num_total = len(self.get_all_patterns())

        labeled_patterns = [p for p in self.get_all_patterns() if p.is_labeled()]
        num_labelel = len(labeled_patterns)
        print(f'Total number of patterns = {num_total}')
        print(f'Number of labeled patterns = {num_labelel}')


if __name__ == "__main__":
    test_dirpath = '/tmp/opxrd_test'
    full_dirpath = '/home/daniel/aimat/data/opXRD/test'
    opxrd_databases = OpXRD.as_database_list(root_dirpath=test_dirpath)
    analyser = DatabaseAnalyser(databases=opxrd_databases, output_dirpath='/tmp/opxrd_analysis')
    analyser.plot_effective_components()

    print(profiler.make_report())