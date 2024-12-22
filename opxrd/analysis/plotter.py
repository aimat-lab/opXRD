import math
import os
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from opxrd.analysis.tables import TableAnalyser
from opxrd.analysis.tools import compute_fourier, print_text
from xrdpattern.pattern import XrdPattern
from numpy.typing import NDArray
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
            data = [p.get_pattern_data(apply_standardization=False) for p in patterns]

            for x, y in data:
                y = y/np.max(y)
                ax.plot(x, y, linewidth=0.25, alpha=0.75, linestyle='--')
                ax.set_title(f'{letter})', loc='left')

        fig.supylabel('Standardized relative intensity (a.u.)')
        fig.supxlabel(r'$2\theta$ [$^\circ$]', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirpath, f'ALL_pattern_multiplot.png'))
        plt.show()

    def plot_reference_fourier(self, b1: float, b2: float, b3 : float, add_noise : bool):
        msg = r'---> Fourier transform of gaussians of the form $I(x) = e^{-0.5(x-b)^2/c}$'
        if add_noise:
            msg += ' with added noise'

        c1, c2 = 0.1, 0.2
        x = np.linspace(0, 180, num=500)

        y = 5 * np.exp(-1 / 2 * (x - b1) ** 2 / c1) + 0.75*np.exp(-1 / 2 * (x - b2) ** 2 / c2) + 0* np.exp(-1 / 2 * (x - b3) ** 2 / 0.1)
        if add_noise:
            y += 0.2* np.random.normal(0, 1, x.shape)

        self._fourier_plots(x, [y], msg=msg, figname='reference_fourier.png')


    def plot_opxrd_fourier(self, combine_plots : bool = True, filter_dbs : Optional[str] = None, n_entries : int = 512):
        y_list = []

        databases = self.databases
        if filter_dbs:
            databases = [db for db in self.databases if filter_dbs.lower() in db.name.lower()]
        for db in databases:
            x, _ = db.patterns[0].get_pattern_data(num_entries=n_entries)
            db_intensities = [p.get_pattern_data(num_entries=n_entries)[1] for p in db.patterns]
            summed_intensities = np.sum(db_intensities, axis=0)
            normalized_sums = summed_intensities / np.max(summed_intensities)

            if not combine_plots:
                self._fourier_plots(x, [normalized_sums], msg=f'---> Fourier transform of summed {db.name} patterns',figname=f'{db.name}_fourier.png')
            else:
                y_list.append(normalized_sums)

        if combine_plots:
            x = np.linspace(0, 180, num=n_entries)
            self._fourier_plots(x, y_list, msg='---> Fourier transform of summed up opXRD patterns',
                                y_names=[db.name for db in self.databases],
                                figname='ALL_fourier.png')


    def _fourier_plots(self, x, y_list: list[NDArray], msg: str, figname: str, y_names: Optional[list[str]] = None):
        if len(y_list) == 0:
            raise ValueError('No y data provided for Fourier Transform plot')

        print_text(msg)
        xf, _ = compute_fourier(x, y_list[0])
        yf_list = []
        for y in y_list:
            xf, yf = compute_fourier(x, y)
            yf_list.append(yf)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        for y in y_list:
            ax1.plot(x, y)
        ax1.set_xlabel(r'$2\theta$')
        ax1.set_ylabel('I(x)')
        ax1.set_title('Original')

        for yf in yf_list:
            ax2.plot(xf, yf, label='Fourier Transform magnitude')
        ax2.set_xlabel('Frequency k')
        ax2.set_ylabel('Magnitude |F(k)|')
        # ax2.set_yscale(f'log')
        ax2.set_xlim(0, 2.5)
        ax2.set_title('Fourier Transform')

        if y_names:
            ax1.legend(y_names, ncol=2)
            ax2.legend(y_names, ncol=2)

        if figname:
            plt.savefig(os.path.join(self.output_dirpath, figname))

        plt.tight_layout()
        plt.show()


    def plot_effective_components(self, use_fractions : bool = True):
        print_text(r'Cumulative explained variance ratio $v$ over components '
                        r'|  $v =  \frac{\sum_i \lambda_i}{\sum^n_{j=1} \lambda_j}$')

        for db_num, db in enumerate(self.databases):
            print(f'[Debug]: Performing PCA for {db.name} | No. patterns = {len(db.patterns)}')
            std_num_entries = 512

            max_components = min(len(db.patterns), std_num_entries)
            standardized_intensities = np.array([p.get_pattern_data(num_entries=std_num_entries)[1] for p in db.patterns])
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


