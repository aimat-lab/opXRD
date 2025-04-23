import math
import os
from typing import Optional

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from opxrd.analysis.tables import TableAnalyser
from opxrd.analysis.tools import compute_standardized_fourier, print_text
from .visualization import define_angle_start_stop_ax, define_recorded_angles_ax, define_spg_ax


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
                y = y/np.max(y)
                ax.plot(x, y, linewidth=0.25, alpha=0.75, linestyle='--')
                ax.set_title(f'{letter})', loc='left')

        fig.supylabel('Standardized relative intensity (a.u.)')
        fig.supxlabel(r'$2\theta$ [$^\circ$]', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dirpath, f'combined.png'))
        plt.show()


    def plot_reference_fourier(self, b1: float, b2: float, A : float = 5, B : float = 0.75, c1 : float = 0.04, add_noise : bool = False):
        msg = r'---> Fourier transform of gaussians of the form $I(x) = e^{-0.5(x-b)^2/c}$'

        if add_noise:
            msg += ' with added noise'

        c2 = 0.04
        x = np.linspace(0, 100, num=1000)

        y = A * np.exp(-1 / 2 * (x - b1) ** 2 / c1) + B*np.exp(-1 / 2 * (x - b2) ** 2 / c2)
        if add_noise:
            y += 0.2* np.random.normal(0, 1, x.shape)
        y = y/np.max(y)

        self._fourier_plots(x, [y], msg=msg, figname='reference_fourier.png')
        print_text(r'$f(x) = \delta(x-a_1)+\delta(x-a_2) \implies \hat{f}(k) = e^{ika_1} + e^{ika_2}$  <br />'
                   r'$|\hat{f}(k)| = 2 | \sin(k\Delta a/2)|$, $\Delta a = a_1 - a_2$')


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
                msg = f'---> Fourier transform of summed {db.name} patterns'
                self._fourier_plots(x, [normalized_sums], msg=msg, figname=f'{db.name}_fourier.png')
            else:
                y_list.append(normalized_sums)

        if combine_plots:
            x = np.linspace(0, 90, num=n_entries)
            msg = f'---> Fourier transform of summed up patterns combined in single figure'
            y_names = [db.name for db in databases]
            self._fourier_plots(x, y_list, msg=msg, y_names=y_names, figname='ALL_fourier.png')


    def _fourier_plots(self, x, y_list: list[NDArray], msg: str, figname: str, y_names: Optional[list[str]] = None):
        if len(y_list) == 0:
            raise ValueError('No y data provided for Fourier Transform plot')

        xf, _ = compute_standardized_fourier(x, y_list[0])
        yf_list = []
        for y in y_list:
            xf, yf = compute_standardized_fourier(x, y)
            yf_list.append(yf)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
        kwargs = {} if len(y_list) <= 1 else {'linewidth': 0.5, 'alpha': 0.75}


        for y in y_list:
            ax1.plot(x, y, **kwargs)
        ax1.set_xlabel(r'$2\theta$')
        ax1.set_ylabel(r'$I(2\theta)$')
        ax1.set_title('Original')

        for yf in yf_list:
            xf, yf = xf[np.where(xf < 2.5)], yf[np.where(xf < 2.5)]
            ax2.plot(xf, yf, label='Fourier Transform magnitude', **kwargs)
        ax2.set_xlabel('Frequency k Value/$deg^{-1}$')
        ax2.set_ylabel('Magnitude |F(k)|')
        ax2.set_yscale(f'log')
        ax2.set_title('Fourier Transform')

        msg += f'| k-spacing: {xf[1] - xf[0]:.2f}, sampling rate= {len(x)/(x[-1] - x[0]):.0f} values/deg$^{-1}$'
        fig.suptitle(msg)

        if y_names:
            ax1.legend(y_names, ncol=2, loc='upper right',fontsize=8)
            ax2.legend(y_names, ncol=2, loc='upper right',fontsize=8)

        if figname:
            plt.savefig(os.path.join(self.output_dirpath, figname))

        plt.tight_layout()
        plt.show()


    def plot_effective_components(self, use_fractions : bool = True):
        print_text(r'---> Cumulative explained variance ratio $v$ over components '
                        r'|  $v =  \frac{\sum_i \lambda_i}{\sum^n_{j=1} \lambda_j}$')

        half = len(self.databases) // 2
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        letters = ['a', 'b']

        for i, ax in enumerate(axes):
            dbs = self.databases[half * i: half * (i + 1)]
            for db in dbs:
                print(f'[Debug]: Performing PCA for {db.name} | No. patterns = {len(db.patterns)}')
                std_num_entries = 512

                max_components = min(len(db.patterns), std_num_entries)
                standardized_intensities = np.array(
                    [p.get_pattern_data(num_entries=std_num_entries)[1] for p in db.patterns])
                pca = PCA(n_components=max_components)
                pca.fit_transform(standardized_intensities)

                cumulative_explained_var = []
                x_axis = np.linspace(0, 1, num=max_components) if use_fractions else range(0, max_components)
                for x in x_axis:
                    n_comp = int(x * max_components) if use_fractions else x
                    cvar = np.sum(pca.explained_variance_ratio_[:n_comp])
                    cumulative_explained_var.append(cvar)

                ax.plot(x_axis, cumulative_explained_var, label=db.name)
            ax.legend(loc='lower right', ncols=2, fontsize='small')
            ax.set_title(f"{letters[i]})", loc='left')
            ax.set_xscale(f'log')
            xlabel = 'Fraction of maximal No. components' if use_fractions else 'No. components'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'Cumulative explained variance ratio')

        plt.tight_layout()
        figname = 'component_fractions.png' if use_fractions else 'components.png'
        plt.savefig(os.path.join(self.output_dirpath,figname))
        plt.show()

    def show_histograms(self, save_fpath: Optional[str] = None, attach_colorbar: bool = True):
        fig = plt.figure(figsize=(12, 8))

        figure = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace=0.35)
        figure.update(top=0.96, bottom=0.075)
        upper_half = figure[0].subgridspec(1, 3)
        ax2 = fig.add_subplot(upper_half[:, :])
        define_spg_ax(patterns=self.patterns, ax=ax2)

        lower_half = figure[1].subgridspec(1, 2)
        ax3 = fig.add_subplot(lower_half[:, 0])
        define_recorded_angles_ax(patterns=self.patterns, ax=ax3)

        if attach_colorbar:
            lower_half_right = lower_half[1].subgridspec(nrows=3, ncols=3, width_ratios=[3, 3, 4], hspace=0, wspace=0)
            ax4 = fig.add_subplot(lower_half_right[1:, :2])  # scaatter
            ax5 = fig.add_subplot(lower_half_right[:1, :2], sharex=ax4)  # Above
            ax6 = fig.add_subplot(lower_half_right[1:, 2:], sharey=ax4)  # Right
            ax7 = fig.add_subplot(lower_half_right[:1, 2:])
            ax7.axis('off')
        else:
            lower_half_right = lower_half[1].subgridspec(nrows=3, ncols=4, width_ratios=[2.75, 3, 3, 3], hspace=0,
                                                         wspace=0)
            ax4 = fig.add_subplot(lower_half_right[1:, 1:3])  # scatter
            ax5 = fig.add_subplot(lower_half_right[:1, 1:3], sharex=ax4)  # Above
            ax6 = fig.add_subplot(lower_half_right[1:, 3:4], sharey=ax4)  # Right
            ax7 = fig.add_subplot(lower_half_right[:4, :1])

        define_angle_start_stop_ax(patterns=self.patterns, density_ax=ax4, top_marginal=ax5, right_marginal=ax6,
                                   cmap_ax=ax7, attach_colorbar=attach_colorbar)

        if save_fpath:
            plt.savefig(save_fpath)
        plt.show()

    def labels_histogram(self, save_fpath: Optional[str] = None):
        fig = plt.figure(figsize=(12, 8))

        figure = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace=0.35)
        figure.update(top=0.96, bottom=0.075)
        upper_half = figure[0].subgridspec(1, 3)
        ax2 = fig.add_subplot(upper_half[:, :])
        define_spg_ax(patterns=self.patterns, ax=ax2)

        if save_fpath:
            plt.savefig(save_fpath)
        plt.show()

