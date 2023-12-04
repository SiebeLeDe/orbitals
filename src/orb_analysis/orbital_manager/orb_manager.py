from abc import ABC
import attrs
from orb_analysis.custom_types import Array2D
import numpy as np
from orb_analysis.orbital.orbital import MO, SFO
import pandas as pd
from itertools import zip_longest

ENERGY_FORMAT = "+.4f"
GROSSPOP_FORMAT = ".3f"


class OrbitalManager(ABC):
    """ This class contains methods for accessing information about orbitals which can be SFOs (symmetrized fragment orbitals) and MOs (molecular orbitals) """


@attrs.define
class MOManager(OrbitalManager):
    """ This class contains methods for accessing information about molecular orbitals (MOs). """
    complex_mos: list[MO]

    def __str__(self):
        """
        Returns a string with the molecular orbital amsview label, homo_lumo label, energy, and grosspop in the formatted way.
        """
        mos_info = [f"{orb.amsview_label :13s} ({orb.homo_lumo_label :7s}): E (Ha): {orb.energy:+.4f}" for orb in self.complex_mos]

        max_len_frag1 = max(len(info) for info in mos_info)

        header = f"{'Molecular Orbitals':^{max_len_frag1}}"
        mos_info_str = '\n'.join([header] + mos_info)

        return mos_info_str


@attrs.define
class SFOManager(OrbitalManager):
    """ This class contains methods for accessing information about symmetrized fragment orbitals (SFOs). """
    frag1_orbs: list[SFO]
    frag2_orbs: list[SFO]
    overlap_matrix: Array2D[np.float64]

    def get_overlap_matrix_table(self):
        """ Returns a table (str) of the overlap matrix """
        row_labels = [orb.homo_lumo_label for orb in self.frag1_orbs]
        column_labels = [orb.homo_lumo_label for orb in self.frag2_orbs][::-1]

        # Create a DataFrame from the overlap matrix
        df = pd.DataFrame(self.overlap_matrix, index=row_labels, columns=column_labels)

        # Convert the DataFrame to a string and return it
        return df.to_string(float_format="{:0.4f}".format)

    def __str__(self):
        longest_orb_label_frag1 = max([len(orb.amsview_label) for orb in self.frag1_orbs]) + 1
        longest_orb_label_frag2 = max([len(orb.amsview_label) for orb in self.frag2_orbs]) + 1
        longest_homo_lumo_label_frag1 = max([len(orb.homo_lumo_label) for orb in self.frag1_orbs]) + 1
        longest_homo_lumo_label_frag2 = max([len(orb.homo_lumo_label) for orb in self.frag2_orbs]) + 1

        frag1_orb_info = [
            f"{orb.amsview_label:{longest_orb_label_frag1}} ({orb.homo_lumo_label :{longest_homo_lumo_label_frag1}}): "
            f"E (Ha): {orb.energy:{ENERGY_FORMAT}}, "
            f"Grosspop: {orb.gross_pop:{GROSSPOP_FORMAT}}"
            for orb in self.frag1_orbs
        ]

        frag2_orb_info = [
            f"{orb.amsview_label:{longest_orb_label_frag2}} ({orb.homo_lumo_label :{longest_homo_lumo_label_frag2}}): "
            f"E (Ha): {orb.energy:{ENERGY_FORMAT}}, "
            f"Grosspop: {orb.gross_pop:{GROSSPOP_FORMAT}}"
            for orb in self.frag2_orbs
        ]

        max_len_frag1 = max(len(info) for info in frag1_orb_info)
        max_len_frag2 = max(len(info) for info in frag2_orb_info)

        combined_info = [f"{frag1:<{max_len_frag1}} | {frag2:<{max_len_frag2}}" for frag1, frag2 in zip_longest(frag1_orb_info, frag2_orb_info, fillvalue="")]

        header = f"{'Fragment 1':^{max_len_frag1}} | {'Fragment 2':^{max_len_frag2}}"
        combined_info_str = '\n'.join([header] + combined_info)

        overlap_matrix_table = self.get_overlap_matrix_table()
        note = "overlap is guaranteed 0.0 when the irreps do not match and when both are unoccupied [non-physical]"

        return f"{combined_info_str}\n\nOverlap Matrix ({note}):\n{overlap_matrix_table}"
