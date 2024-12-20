import math
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from opxrd.analysis.tables import TableAnalyser
from opxrd.analysis.tools import print_text, compute_fourier
from xrdpattern.pattern import XrdPattern

# -----------------------------------------

class DatabaseAnalyser(TableAnalyser):
    def plot_in_single(self, limit_patterns: int):
        lower_alphabet = [chr(i) for i in range(97, 123)]
        explanation = [f'{letter}:{db.name}' for letter, db in zip(lower_alphabet, self.databases)]
        print_text(f'---> Combined pattern plot for databaes {explanation} | No. patterns = {limit_patterns}')

        cols = 3
        rows = math.ceil(len(self.databases) / cols)

        axes = []
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(len(self.databases)):
            ax = fig.add_subplot(rows, cols, i + 1)
            axes.append(ax)

        for letter, ax, database in zip(lower_alphabet, axes, self.databases):
            patterns = database.patterns[:limit_patterns]
            data = [p.get_pattern_data() for p in patterns]

            for x, y in data:
                ax.plot(x, y, linewidth=0.25, alpha=0.75, linestyle='--')
                ax.set_title(f'{letter})', loc='left')

        fig.supylabel('Standardized relative intensity (a.u.)')
        fig.supxlabel(r'$2\theta$ [$^\circ$]', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirpath, f'ALL_pattern_multiplot.png'))
        plt.show()


    @staticmethod
    def plot_reference_fourier(b1: float = 0.3, b2: float = 0.5, c = 2):
        print_text(r'---> Fourier transform of a pair of gaussians $I(x) = e^{{-0.5(x-b)^2/c}$')

        c1, c2 = 0.1, 0.2
        x = np.linspace(0, 180, num=1000)
        y = 5 * np.exp(-1 / 2 * (x - b1) ** 2 / c1) + np.exp(-1 / 2 * (x - b2) ** 2 / c2)
        xf, yf = compute_fourier(x, y)
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
        ax2.set_title('Fourier Transform')

        plt.tight_layout()
        plt.show()


    def plot_opxrd_fourier(self):
        for db in self.databases:
            fig, ax = plt.subplots(figsize=(10, 4), dpi=300)

            db_intensities = [p.get_pattern_data()[1] for p in db.patterns]
            intensity_sum = np.sum(db_intensities, axis=0)
            x, _ = db.patterns[0].get_pattern_data()
            xf, yf = compute_fourier(x, intensity_sum)
            plt.plot(xf, yf)

            ax.set_title(f'{db.name} patterns summed up fourier transform ' +
                         r'$F(k)=\int d(2\theta) I(2\theta) e^{-ik2\theta}$')
            ax.set_xlabel(r'k [deg$^{−1}$]')
            ax.set_ylabel('l|F($k$)| (a.u.)')

            plt.savefig(os.path.join(self.output_dirpath, f'{db.name}_fourier.png'))
            plt.show()


    def plot_effective_components(self, use_fractions : bool = True):
        print_text(r'Cumulative explained variance ratio $v$ over components '
                        r'|  $v =  \frac{\sum_i \lambda_i}{\sum^n_{j=1} \lambda_j}$')

        for db_num, db in enumerate(self.databases):
            print(f'[Debug]: Performing PCA for {db.name} | No. patterns = {len(db.patterns)}')

            max_components = min(len(db.patterns), XrdPattern.std_num_entries())
            standardized_intensities = np.array([p.get_pattern_data()[1] for p in db.patterns])
            pca = PCA(n_components=max_components)
            pca.fit_transform(standardized_intensities)

            cumulative_explained_var = []
            x_axis = np.linspace(0, 1, num=max_components) if use_fractions else range(0, max_components)
            for x in x_axis:
                n_comp = int(x * max_components) if use_fractions else x
                cvar = np.sum(pca.explained_variance_ratio_[:n_comp])
                cumulative_explained_var.append(cvar)
            plt.plot(x_axis, cumulative_explained_var, label=db.name)


        if use_fractions:
            plt.xlabel(f'Fraction of max. No. components')
        else:
            plt.xlabel(f'No. components')

        plt.xscale(f'log')
        plt.ylabel(f'Cumulative explained variance ratio $V$')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.output_dirpath, f'ALL_effective_components.png'))

        plt.show()


    def plot_histogram(self, attach_colorbar : bool = False):
        print_text(f'---> Histogram of general information on opXRD')
        self.joined_db.show_histograms(save_fpath=os.path.join(self.output_dirpath, 'ALL_histogram.png'), attach_colorbar=attach_colorbar)


if __name__ == "__main__":
    from opxrd.wrapper import OpXRD
    smoltest_dirpath = '/home/daniel/aimat/data/opXRD/test_smol'
    bigtest_dirpath = '/home/daniel/aimat/data/opXRD/test'
    test_databases = OpXRD.load_project_list(root_dirpath=smoltest_dirpath)

    analyser = DatabaseAnalyser(databases=test_databases, output_dirpath='/tmp/opxrd_analysis')
    analyser.plot_reference_fourier(b1=60, b2=80)