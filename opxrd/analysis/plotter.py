import math
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from opxrd.analysis.tables import TableAnalyser
from opxrd.analysis.tools import compute_fourier, print_text
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


    @classmethod
    def plot_reference_fourier(cls, b1: float, b2: float, b3 : float, add_noise : bool):
        msg = r'---> Fourier transform of gaussians of the form $I(x) = e^{{-0.5(x-b)^2/c}$'
        if add_noise:
            msg += ' with added noise'

        c1, c2 = 0.1, 0.2
        x = np.linspace(0, 180, num=500)

        y = 5 * np.exp(-1 / 2 * (x - b1) ** 2 / c1) + np.exp(-1 / 2 * (x - b2) ** 2 / c2) + 2* np.exp(-1 / 2 * (x - b3) ** 2 / 0.1)
        if add_noise:
            y += 0.2* np.random.normal(0, 1, x.shape)

        cls.fourier_plots(x, [y], msg=msg, names=[])


    def plot_opxrd_fourier(self):
        x, _ = self.databases[0].patterns[0].get_pattern_data()
        y_list = []
        for db in self.databases:
            db_intensities = [p.get_pattern_data()[1] for p in db.patterns]
            summed_intensities = np.sum(db_intensities, axis=0)
            normalized_sums = summed_intensities / np.max(summed_intensities)
            y_list.append(normalized_sums)
        self.fourier_plots(x, y_list, msg='---> Fourier transform of summed up opXRD patterns', names=[db.name for db in self.databases])

            #
            #
            #
            # xf, yf = compute_fourier(x, summed_intensities)
            # plt.plot(xf, yf)
            #
            # ax.set_title(f'{db.name} patterns summed up fourier transform ' +
            #              r'$F(k)=\int d(2\theta) I(2\theta) e^{-ik2\theta}$')
            # ax.set_xlabel(r'k [deg$^{−1}$]')
            # ax.set_ylabel('l|F($k$)| (a.u.)')
            #
            # plt.savefig(os.path.join(self.output_dirpath, f'{db.name}_fourier.png'))
            # plt.show()


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


    @classmethod
    def fourier_plots(cls, x, y_list : list, msg : str, names : list[str]):
        if not y_list:
            raise ValueError('No y data provided for Fourier Transform plot')

        print_text(msg)
        xf, _ = compute_fourier(x, y_list[0])
        yf_list = []
        for y in y_list:
            xf, yf = compute_fourier(x, y)
            yf_list.append(yf)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Gaussian plot
        for y in y_list:
            ax1.plot(x, y)
        ax1.set_xlabel(r'$2\theta$')
        ax1.set_ylabel('I(x)')
        ax1.set_title('Original')

        # Fourier Transform plot
        for yf in yf_list:
            ax2.plot(xf, yf, label='Fourier Transform magnitude')
        ax2.set_xlabel('Frequency k')
        ax2.set_ylabel('Magnitude |F(k)|')
        # ax2.set_yscale('log')
        ax2.set_title('Fourier Transform')

        if names:
            ax1.legend(names)
            ax2.legend(names)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    from opxrd.wrapper import OpXRD
    smoltest_dirpath = '/home/daniel/aimat/data/opXRD/test_smol'
    bigtest_dirpath = '/home/daniel/aimat/data/opXRD/test'
    test_databases = OpXRD.load_project_list(root_dirpath=smoltest_dirpath)

    analyser = DatabaseAnalyser(databases=test_databases, output_dirpath='/tmp/opxrd_analysis')
    analyser.plot_reference_fourier(b1=60, b2=80)