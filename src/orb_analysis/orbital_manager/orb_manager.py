from abc import ABC
from typing import Any
import attrs
from tabulate import tabulate
from orb_analysis.custom_types import Array2D
import numpy as np
from orb_analysis.log_messages import INTERACTION_MATRIX_NOTE, OVERLAP_MATRIX_NOTE, format_message
from orb_analysis.orbital.orbital import MO, SFO
import pandas as pd
from itertools import zip_longest
from scm.plams import Units

# Used for formatting the tables in the __str__ methods using the tabulate package
TABLE_FORMAT_OPTIONS: dict[str, Any] = {
    "numalign": "left",
    "stralign": "left",
    "floatfmt": "+.3f",
    "intfmt": "d",
    "tablefmt": "simple_outline",
}


class OrbitalManager(ABC):
    """This class contains methods for accessing information about orbitals which can be SFOs (symmetrized fragment orbitals) and MOs (molecular orbitals)"""


@attrs.define
class MOManager(OrbitalManager):
    """This class contains methods for accessing information about molecular orbitals (MOs)."""

    complex_mos: list[MO]

    def __str__(self):
        """
        Returns a string with the molecular orbital amsview label, homo_lumo label, energy, and grosspop in the formatted way.
        """
        mos_info = [[orb.amsview_label, orb.homo_lumo_label, orb.energy] for orb in self.complex_mos]

        headers = ["Molecular Orbitals", "", "Energy (Ha)"]
        table = tabulate(tabular_data=mos_info, headers=headers, **TABLE_FORMAT_OPTIONS)

        return table


@attrs.define
class SFOManager(OrbitalManager):
    """This class contains methods for accessing information about symmetrized fragment orbitals (SFOs)."""

    frag1_orbs: list[SFO]  # Is sorted by energy from HOMO-x -> LUMO+x
    frag2_orbs: list[SFO]  # Is reversed: LUMO+x -> HOMO-x
    overlap_matrix: Array2D[np.float64]

    def __str__(self):
        sfo_overview_table = self.get_sfo_overview_table()

        overlap_matrix_table = self.get_overlap_matrix_table()
        overlap_str = "\n".join(["Overlap Matrix", format_message(OVERLAP_MATRIX_NOTE), overlap_matrix_table])

        sfo_interaction_matrix_table = self.get_sfo_interaction_matrix()
        sfo_interaction_str = "\n".join(["SFO Interaction Matrix", format_message(INTERACTION_MATRIX_NOTE), sfo_interaction_matrix_table])

        return f"{sfo_overview_table}\n\n{overlap_str})\n\n{sfo_interaction_str}"

    def get_sfo_overview_table(self):
        frag1_orb_info = [[orb.amsview_label, orb.homo_lumo_label, orb.energy, orb.gross_pop] for orb in self.frag1_orbs]

        frag2_orb_info = [
            [orb.amsview_label, orb.homo_lumo_label, orb.energy, orb.gross_pop]
            for orb in self.frag2_orbs[::-1]  # Reversed because best format is from bottom to top: HOMO-x -> LUMO+x
        ]

        combined_info = [frag1 + frag2 for frag1, frag2 in zip_longest(frag1_orb_info, frag2_orb_info, fillvalue=["", "", "", ""])]

        headers = ["Fragment 1", "", "E (Ha)", "Gross population"] + ["Fragment 2", "", "E (Ha)", "Gross population"]
        table = tabulate(tabular_data=combined_info, headers=headers, **TABLE_FORMAT_OPTIONS)
        return table

    def get_overlap_matrix_table(self):
        """Returns a table (str) of the overlap matrix"""
        row_labels = [orb.homo_lumo_label for orb in self.frag1_orbs]
        column_labels = [orb.homo_lumo_label for orb in self.frag2_orbs]

        # Create a DataFrame from the overlap matrix
        df = pd.DataFrame(self.overlap_matrix, index=row_labels, columns=column_labels)
        table = tabulate(df, headers="keys", **TABLE_FORMAT_OPTIONS)  # type: ignore # df is accepted as argument
        return table

    def get_sfo_interaction_matrix(self):
        """
        Calculates interaction matrix which is composed of:
            1. The stabilization matrix (S^2 / energy_gap) * 100 in units (a.u.^2 / eV) for HOMO-LUMO interactions
            2. The Pauli repulsion matrix (S^2) in units (a.u.^2) for HOMO-HOMO interactions
            3. Zero matrix for LUMO-LUMO interactions

        """

        def calculate_matrix_element(sfo1: SFO, sfo2: SFO, overlap: float) -> float:
            """Checks if the interaction is HOMO-HOMO / HOMO-LUMO / LUMO-LUMO and returns the correct value (see parent function docstring"""
            # LUMO-LUMO non-physical
            if not sfo1.is_occupied and not sfo2.is_occupied:
                return 0.0

            # HOMO-HOMO Pauli repulsion
            if sfo1.is_occupied and sfo2.is_occupied:
                return overlap**2 * 100

            # HOMO-LUMO / LUMO-HOMO favorable orbital interactions (SCF process)
            energy_gap: float = Units.convert(abs(frag1_orb.energy - frag2_orb.energy), "ha", "eV")  # type: ignore
            if np.isclose(energy_gap, 0):
                return overlap * 100
            else:
                return (overlap**2 / energy_gap) * 100

        row_labels = [orb.homo_lumo_label for orb in self.frag1_orbs]
        column_labels = [orb.homo_lumo_label for orb in self.frag2_orbs]

        stabilization_matrix = np.zeros_like(self.overlap_matrix)
        for i, frag1_orb in enumerate(self.frag1_orbs):
            for j, frag2_orb in enumerate(self.frag2_orbs):
                overlap = self.overlap_matrix[i, j]
                stabilization_matrix[i, j] = calculate_matrix_element(sfo1=frag1_orb, sfo2=frag2_orb, overlap=overlap)

        df = pd.DataFrame(stabilization_matrix, index=row_labels, columns=column_labels)
        table = tabulate(df, headers="keys", **TABLE_FORMAT_OPTIONS)  # type: ignore # df is accepted as argument
        return table
