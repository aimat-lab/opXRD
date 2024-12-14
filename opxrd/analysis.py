import os

import numpy as np
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from xrdpattern.pattern import PatternDB
from xrdpattern.pattern import XrdPattern
from matplotlib import pyplot as plt

from opxrd import OpXRD
# ----------------------------------------------------------

class DatabaseAnalyser:
    def __init__(self, databases : list[PatternDB], output_dirpath : str):
        if len(databases) == 0:
            raise ValueError('No databases provided')
        self.databases : list[PatternDB] = databases
        self.joined_db : PatternDB = PatternDB.merge(databases)
        self.output_dirpath : str = output_dirpath
        os.makedirs(self.output_dirpath, exist_ok=True)

    def plot_databases_in_single(self):
        for database in self.databases:
            database.show_all(single_plot=True)

    def plot_fourier(self, x, y, max_freq=10):
        N = len(y)  # Number of sample points
        T = (x[-1] - x[0]) / (N - 1)  # Sample spacing
        yf = np.fft.fft(y)  # Perform FFT
        xf = np.fft.fftfreq(N, T)[:N // 2]  # Frequency axis

        magnitude = 2.0 / N * np.abs(yf[:N // 2])
        fig, ax = plt.subplots(figsize=(10, 4))
        self._set_ax_properties(ax, title='Fourier Transform', xlabel='Frequency (Hz)', ylabel='Magnitude')
        ax.grid(True)

        if max_freq is not None:
            valid_indices = xf <= max_freq
            plt.plot(xf[valid_indices], magnitude[valid_indices])
        else:
            plt.plot(xf, magnitude)

        plt.show()

    def plot_pattern_dbs(self, title : str):
        combined_pattern_list = self.get_all_patterns()
        xy_list = [p.get_pattern_data() for p in combined_pattern_list]
        combined_y_list = [y for x, y in xy_list]

        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(combined_y_list)

        rand_indices = [np.random.randint(low=0, high=len(combined_y_list)) for _ in range(10)]
        example_xy_list = [combined_pattern_list[idx].get_pattern_data() for idx in rand_indices]
        example_pca_coords =  [transformed_data[idx] for idx in rand_indices]

        self._plot_pca_scatter(transformed_data, title=title)
        self._plot_pca_basis(pca, title=title)
        self._plot_reconstructed(pca, example_xy_list, example_pca_coords, title=title)
        print('done')

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


if __name__ == "__main__":
    test_dirpath = '/tmp/opxrd_test'
    full_dirpath = '/tmp/opxrd'
    opxrd_databases = OpXRD.as_database_list(root_dirpath=test_dirpath)
    analyser = DatabaseAnalyser(databases=opxrd_databases, output_dirpath='/tmp/opxrd_analysis')
    analyser.plot_databases_in_single()