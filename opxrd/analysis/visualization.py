from collections import Counter, defaultdict

import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from xrdpattern.crystal.spgs import SpacegroupConverter
from xrdpattern.pattern import XrdPattern
from xrdpattern.xrd import LabelType


# -----------------------------------------

class AxesDefiner:
    @staticmethod
    def define_elements_ax(patterns : list[XrdPattern], ax : Axes, letter : str):
        element_map = defaultdict(int)
        basis_labeled = []
        other = []

        for p in patterns:
            if p.has_label(label_type=LabelType.basis):
                basis_labeled.append(p)
            else:
                other.append(p)

        for p in basis_labeled:
            if p.primary_phase is None:
                raise ValueError(f'Pattern {p.get_name()} has no primary phase')
            if p.primary_phase.num_atoms == 0:
                raise ValueError(f'Pattern {p.get_name()} has no atoms')

            primary_phase = p.primary_phase
            for element in primary_phase.to_pymatgen().elements:
                element = element.symbol
                element_map[element] += 1

        from pymatgen.core.periodic_table import Element
        element_symbols = set([el.symbol for el in Element])

        for p in other:
            compositions = [ph.chemical_composition for ph in p.powder_experiment.phases if not ph.chemical_composition is None]
            joined_comp = ''.join(compositions)
            for element in element_symbols:
                element_map[element] += int(element in joined_comp)

        keys, counts = list(element_map.keys()), list(element_map.values())
        zipped = zip(keys, counts)
        sorted_zipped = sorted(zipped, key=lambda x : -x[1])
        keys, counts = [], []

        for k,c in sorted_zipped[:30]:
            keys.append(k)
            counts.append(c)
        ax.bar(keys, counts)

        ax.tick_params(labelbottom=True, labelleft=True)  # Enable labels
        ax.set_title(f'({letter})', loc='left')
        ax.set_ylabel(f'No. patterns')

    @staticmethod
    def define_spg_ax(patterns: list[XrdPattern], ax: Axes, letter : str):
        keys, counts = get_keys_counts(patterns=patterns, attr='primary_phase.spacegroup')
        keys, counts = keys[:30], counts[:30]

        spgs = [int(k) for k in keys]
        spg_formulas = [f'${SpacegroupConverter.to_formula(spg, mathmode=True)}$' for spg in spgs]
        ax.bar(spg_formulas, counts)
        ax.tick_params(labelbottom=True, labelleft=True)  # Enable labels
        ax.set_title(f'({letter})', loc='left')
        ax.set_ylabel(f'No. patterns')
        ax.set_xticklabels(spg_formulas, rotation=90)

    @staticmethod
    def define_volume_ax(patterns : list[XrdPattern], ax : Axes, letter : str):
        keys, counts = get_keys_counts(patterns=patterns, attr='primary_phase.volume_uc')

        basis = 2
        exponent_range = range(6,16)
        order_counts_map = {i : 0 for i in exponent_range}
        forgotten_counts = 0

        for k, c in zip(keys, counts):
            k = int(k)
            exponent = 1
            while basis**exponent < k:
                exponent += 1
            try:
                order_counts_map[exponent] += c
            except:
                forgotten_counts += c
        print(forgotten_counts)

        labels = []
        for j in exponent_range:
            labels.append(f'$\\leq {basis}^{{{j}}} \\AA$')

        counts = list(order_counts_map.values())
        ax.bar(labels, counts)
        ax.tick_params(labelbottom=True, labelleft=True)  # Enable labels
        ax.set_title(f'({letter})', loc='left')
        ax.set_ylabel(f'No. patterns')


    @staticmethod
    def define_no_atoms_ax(patterns : list[XrdPattern], ax : Axes, letter : str):
        keys, counts = get_keys_counts(patterns=patterns, attr='primary_phase.num_atoms')

        order_counts_map = {'10' : 0, '100' : 0, '1000' : 0, 'BIG' : 0}
        for k, c in zip(keys, counts):
            k = int(k)
            if k <= 10:
                order = '10'
            elif k <= 100:
                order = '100'
            elif k <= 1000:
                order = '1000'
            else:
                order = 'BIG'
            order_counts_map[order] += c

        no_atoms_str = r'N_{\text{atom}}'
        labels = [f'${no_atoms_str} \\leq 10$',
                  f'$10 < {no_atoms_str} \\leq 10^2$',
                  f'$10^2 < {no_atoms_str} \\leq 10^3$',
                  f'${no_atoms_str} > 10^3$']
        counts = list(order_counts_map.values())
        ax.bar(labels, counts)
        ax.tick_params(labelbottom=True, labelleft=True)  # Enable labels
        ax.set_title(f'({letter})', loc='left')
        ax.set_ylabel(f'No. patterns')

    @staticmethod
    def define_recorded_angles_ax(patterns: list[XrdPattern], ax: Axes):
        values = get_values(patterns=patterns, attr='angular_resolution')
        ax.set_title(f'(a)', loc='left')
        ax.hist(values, bins=10, range=(0, 0.1), edgecolor='black')
        ax.set_xlabel(r'Angular resolution $\Delta(2\theta)$ [$^\circ$]')
        ax.set_yscale('log')
        ax.set_ylabel(f'No. patterns')

    @staticmethod
    def define_angle_start_stop_ax(patterns: list[XrdPattern], density_ax: Axes, top_marginal: Axes, right_marginal: Axes, cmap_ax: Axes, attach_colorbar: bool):
        start_data = get_values(patterns=patterns, attr='startval')
        end_data = get_values(patterns=patterns, attr='endval')
        start_angle_range = (0, 60)
        end_angle_range = (0, 180)

        # noinspection PyTypeChecker
        h = density_ax.hist2d(start_data, end_data, bins=(10, 10), range=[list(start_angle_range), list(end_angle_range)],
                              norm=matplotlib.colors.LogNorm())
        density_ax.set_xlabel(r'Smallest recorded $2\theta$ [$^\circ$]')
        density_ax.set_ylabel(r'Largest recorded $2\theta$ [$^\circ$]')
        density_ax.set_xlim(start_angle_range)
        density_ax.set_ylim(end_angle_range)

        density_ax.xaxis.set_major_locator(MaxNLocator(4))
        xticks = density_ax.get_xticks()
        density_ax.set_xticks(xticks[:-1])



        if attach_colorbar:
            divider = make_axes_locatable(density_ax)
            cax = divider.append_axes('right', size='5%', pad=0.0)
            plt.colorbar(h[3], cax=cax, orientation='vertical')

        else:
            plt.colorbar(h[3], cax=cmap_ax, orientation='vertical', location='left')
            cmap_ax.set_ylabel(f'No. patterns')

        top_marginal.hist(start_data, bins=np.linspace(*start_angle_range, num=10), edgecolor='black')
        top_marginal.set_title(f'(b)', loc='left')
        top_marginal.set_yscale('log')
        top_marginal.tick_params(axis="x", labelbottom=False, which='both', bottom=False)

        if attach_colorbar:
            divider = make_axes_locatable(top_marginal)
            cax = divider.append_axes('right', size='5%', pad=0.0)
            cax.axis('off')

            divider = make_axes_locatable(right_marginal)
            cax = divider.append_axes('left', size='15%', pad=0.0)
            cax.axis('off')

        else:
            divider = make_axes_locatable(cmap_ax)
            cax = divider.append_axes('right', size=0.8, pad=0.0)
            cax.axis('off')

        right_marginal.hist(end_data, bins=np.linspace(*end_angle_range, num=10), orientation='horizontal',edgecolor='black')
        right_marginal.set_xscale('log')
        right_marginal.tick_params(axis="y", labelleft=False, which='both', left=False)


def get_keys_counts(patterns : list[XrdPattern], attr : str, sort_by_keys : bool = False):
    values = get_values(patterns, attr)
    count_map = Counter(values)
    if sort_by_keys:
        sorted_counts = sorted(count_map.items(), key=lambda x: x[0])
    else:
        sorted_counts = sorted(count_map.items(), key=lambda x: x[1], reverse=True)
    keys, counts = zip(*sorted_counts)

    return keys, counts


def get_values(patterns : list[XrdPattern], attr : str) -> (list[str], list[int]):
    def nested_getattr(obj: object, attr_string):
        attr_names = attr_string.split('.')
        for name in attr_names:
            obj = getattr(obj, name)
        return obj

    values = []
    for pattern in patterns:
        try:
            v = nested_getattr(pattern, attr)
            values.append(v)
        except Exception as e:
            print(f'Could not extract attribute "{attr}" from pattern {pattern.get_name()}\n- Reason: {e}')

    if not values:
        raise ValueError(f'No data found for attribute {attr}')
    if any([v is None for v in values]):
        raise ValueError(f'Attribute {attr} contains None values')

    return values
