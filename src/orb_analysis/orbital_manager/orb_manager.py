from abc import ABC
import attrs
from orb_analysis.custom_types import Array2D
import numpy as np
from orb_analysis.orbital.orbital import MO, SFO
import pandas as pd


class OrbitalManager(ABC):
    """ This class contains methods for accessing information about orbitals which can be SFOs (symmetrized fragment orbitals) and MOs (molecular orbitals) """


@attrs.define
class MOManager(OrbitalManager):
    """ This class contains methods for accessing information about molecular orbitals (MOs). """
    complex_mos: list[MO]


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
        """
        Returns a string with the fragment orbital amsview label, homo_lumo label, energy, and grosspop in the formatted way.
        It also includes the formatted overlap matrix
        """
        orbital_info = ""
        for orb in self.frag1_orbs:
            orbital_info += f"Fragment 1 SFO: {orb.amsview_label :10s}, {orb.homo_lumo_label :8s}, E (Ha): {orb.energy :.3f}, Grosspop: {orb.gross_pop :.4f}\n"
        for orb in self.frag2_orbs:
            orbital_info += f"Fragment 2 SFO: {orb.amsview_label :10s}, {orb.homo_lumo_label :8s}, E (Ha): {orb.energy :.3f}, Grosspop: {orb.gross_pop :.4f}\n"

        overlap_matrix_table = self.get_overlap_matrix_table()

        return f"{orbital_info}\nOverlap Matrix (LUMO-LUMO overlap is set to 0.0):\n{overlap_matrix_table}"
