from abc import ABC
from itertools import zip_longest
from typing import Any

import attrs
import numpy as np
import pandas as pd
from tabulate import tabulate

from orb_analysis.custom_types import Array2D, SFOInteractionTypes
from orb_analysis.log_messages import OVERLAP_MATRIX_NOTE, SFO_ORDER_NOTE, format_message, interaction_matrix_message
from orb_analysis.orbital.orbital import MO, SFO

# Used for formatting the tables in the __str__ methods using the tabulate package
TABLE_FORMAT_OPTIONS: dict[str, Any] = {
    "numalign": "left",
    "stralign": "left",
    "floatfmt": "+.3f",
    "tablefmt": "simple_outline",
}

# ------------------------------------------------------------ #
# --------------------- Helper Functions --------------------- #
# ------------------------------------------------------------ #


def filter_sfos_by_interaction_type(frag1_sfos: list[SFO], frag2_sfos: list[SFO], interaction_type: SFOInteractionTypes) -> tuple[list[int], list[int]]:
    """Returns the indices of relevant frag1 and frag 2 SFOs depending on the interaction type"""

    if interaction_type == SFOInteractionTypes.HOMO_HOMO:
        frag1_filtered_indices = [index for index, sfo in enumerate(frag1_sfos) if sfo.is_occupied]
        frag2_filtered_indices = [index for index, sfo in enumerate(frag2_sfos) if sfo.is_occupied]

    elif interaction_type == SFOInteractionTypes.HOMO_LUMO:
        frag1_filtered_indices = [index for index, sfo in enumerate(frag1_sfos) if sfo.is_occupied]
        frag2_filtered_indices = [index for index, sfo in enumerate(frag2_sfos) if not sfo.is_occupied]

    elif interaction_type == SFOInteractionTypes.LUMO_HOMO:
        frag1_filtered_indices = [index for index, sfo in enumerate(frag1_sfos) if not sfo.is_occupied]
        frag2_filtered_indices = [index for index, sfo in enumerate(frag2_sfos) if sfo.is_occupied]

    else:
        frag1_filtered_indices = [index for index in range(len(frag1_sfos))]
        frag2_filtered_indices = [index for index in range(len(frag2_sfos))]

    return frag1_filtered_indices, frag2_filtered_indices


def calculate_matrix_element(sfo1: SFO, sfo2: SFO, overlap: float) -> float:
    """Checks if the interaction is HOMO-HOMO / HOMO-LUMO / LUMO-LUMO and returns the correct value (see parent function docstring"""
    # LUMO-LUMO: non-physical
    if not sfo1.is_occupied and not sfo2.is_occupied:
        return 0.0

    # HOMO-HOMO: Pauli repulsion
    if sfo1.is_occupied and sfo2.is_occupied:
        return overlap**2 * 100

    # HOMO-LUMO / LUMO-HOMO: favorable orbital interactions (SCF process)
    energy_gap: float = abs(sfo1.energy - sfo2.energy) if sfo1.energy > sfo2.energy else abs(sfo2.energy - sfo1.energy)
    # print(f"{sfo1.energy :.2f} {sfo2.energy :.2f} {energy_gap:.2f} {energy_gap2:.2f}")
    if np.isclose(energy_gap, 0):
        return -overlap * 100
    else:
        return (overlap**2 / energy_gap) * 100


# ------------------------------------------------------------ #
# --------------------- Classes Functions -------------------- #
# ------------------------------------------------------------ #


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

        table_headers = ["Molecular Orbitals", "", "Energy (eV)"]
        table = tabulate(tabular_data=mos_info, headers=table_headers, **TABLE_FORMAT_OPTIONS)

        return "\n\n".join(["Molecular Orbitals", table])


@attrs.define
class SFOManager(OrbitalManager):
    """This class contains methods for accessing information about symmetrized fragment orbitals (SFOs)."""

    frag1_sfos: list[SFO]  # Is sorted by energy from HOMO-x -> LUMO+x
    frag2_sfos: list[SFO]  # Is reversed: LUMO+x -> HOMO-x
    overlap_matrix: Array2D[np.float64]

    def __str__(self):
        sfo_overview_table = self.get_sfo_overview_table()

        overlap_matrix_table = self.get_overlap_matrix_table()
        overlap_str = "\n".join(["Overlap Matrix", format_message(OVERLAP_MATRIX_NOTE + SFO_ORDER_NOTE), overlap_matrix_table])

        interaction_matrices_str = ""
        for sfo_interaction in SFOInteractionTypes:
            interaction_table = self.get_sfo_interaction_matrix(sfo_interaction)
            header, note = interaction_matrix_message(sfo_interaction)
            interaction_matrices_str += "\n".join([header, format_message(note), interaction_table])

        return f"{sfo_overview_table}\n\n{overlap_str})\n\n{interaction_matrices_str}"

    def get_sfo_overview_table(self):
        frag1_orb_info = [[orb.amsview_label, orb.homo_lumo_label, orb.energy, orb.gross_pop] for orb in self.frag1_sfos]

        frag2_orb_info = [
            [orb.amsview_label, orb.homo_lumo_label, orb.energy, orb.gross_pop]
            for orb in self.frag2_sfos[::-1]  # Reversed because best format is from bottom to top: HOMO-x -> LUMO+x
        ]

        combined_info = [frag1 + frag2 for frag1, frag2 in zip_longest(frag1_orb_info, frag2_orb_info, fillvalue=["", "", "", ""])]

        headers = ["Fragment 1", "", "E (eV)", "Gross population"] + ["Fragment 2", "", "E (eV)", "Gross population"]
        table = tabulate(tabular_data=combined_info, headers=headers, **TABLE_FORMAT_OPTIONS)
        return table

    def get_overlap_matrix_table(self):
        """Returns a table (str) of the overlap matrix"""
        row_labels = [orb.homo_lumo_label for orb in self.frag1_sfos]
        column_labels = [orb.homo_lumo_label for orb in self.frag2_sfos]

        # Create a DataFrame from the overlap matrix
        df = pd.DataFrame(self.overlap_matrix, index=row_labels, columns=column_labels)
        table = tabulate(df, headers="keys", **TABLE_FORMAT_OPTIONS)  # type: ignore # df is accepted as argument
        return table

    def get_sfo_interaction_matrix(self, interaction_type: SFOInteractionTypes):
        """
        Calculates interaction matrix which is composed of:
            1. The stabilization matrix (S^2 / energy_gap) * 100 in units (a.u.^2 / eV) for HOMO-LUMO interactions or -S * 100 for degenerate SFOs
            2. The Pauli repulsion matrix (S^2) in units (a.u.^2) for HOMO-HOMO interactions

        """
        frag1_filtered_indices, frag2_filtered_indices = filter_sfos_by_interaction_type(self.frag1_sfos, self.frag2_sfos, interaction_type)
        relevant_frag1_sfos = [self.frag1_sfos[i] for i in frag1_filtered_indices]
        relevant_frag2_sfos = [self.frag2_sfos[i] for i in frag2_filtered_indices]

        row_labels = [orb.homo_lumo_label for orb in relevant_frag1_sfos]
        column_labels = [orb.homo_lumo_label for orb in relevant_frag2_sfos]

        stabilization_matrix = np.zeros(shape=(len(relevant_frag1_sfos), len(relevant_frag2_sfos)))
        for new_index1, (old_index1, frag1_orb) in enumerate(zip(frag1_filtered_indices, relevant_frag1_sfos)):
            for new_index2, (old_index2, frag2_orb) in enumerate(zip(frag2_filtered_indices, relevant_frag2_sfos)):
                overlap = self.overlap_matrix[old_index1, old_index2]
                stabilization_matrix[new_index1, new_index2] = calculate_matrix_element(sfo1=frag1_orb, sfo2=frag2_orb, overlap=overlap)
        df = pd.DataFrame(stabilization_matrix, index=row_labels, columns=column_labels)
        table = tabulate(df, headers="keys", **TABLE_FORMAT_OPTIONS)  # type: ignore # df is accepted as argument
        return table
