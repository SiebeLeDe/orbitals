from abc import ABC
from itertools import zip_longest
from typing import Any

import attrs
import numpy as np
import pandas as pd
from orb_analysis.custom_types import Array2D, SFOInteractionTypes
from orb_analysis.log_messages import OVERLAP_MATRIX_NOTE, SFO_ORDER_NOTE, format_message, interaction_matrix_message
from orb_analysis.orbital.orbital import MO, SFO
from orb_analysis.orbital.orbital_pair import OrbitalPair
from orb_analysis.orbital_manager.shared_functions import calculate_matrix_element, filter_sfos_by_interaction_type
from tabulate import tabulate

# Used for formatting the tables in the __str__ methods using the tabulate package
TABLE_FORMAT_OPTIONS: dict[str, Any] = {
    "numalign": "left",
    "stralign": "left",
    "floatfmt": "+.3f",
    "tablefmt": "simple",
}


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

    @property
    def stabilization_matrix(self) -> Array2D[np.float64]:
        """Creates the stabilization matrix, which is S^2 / epsilon * 100 with S being the overlap, epsilon the energy gap, and the 100 factor for scaling"""
        stabilization_matrix = np.zeros(self.overlap_matrix.shape)
        for index1, frag1_orb in enumerate(self.frag1_sfos):
            for index2, frag2_orb in enumerate(self.frag2_sfos):
                overlap = self.overlap_matrix[index1, index2]

                if frag1_orb.is_occupied and frag2_orb.is_occupied:
                    continue  # Pauli repulsion is not included in the stabilization matrix
                stabilization_matrix[index1, index2] = calculate_matrix_element(sfo1=frag1_orb, sfo2=frag2_orb, overlap=overlap)
        return stabilization_matrix

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

    def get_most_destabilizing_pauli_pairs(self, n_pairs: int = 4) -> list[OrbitalPair]:
        """
        Determines which SFO pairs have the strongest repulsion by searching for the indices where the Pauli repulsion matrix is the largest.
        Returns a list with the user-defined number of pairs with its first entry the pair that has the most destabilizing effect, and so on.
        """

        # First, get the overlap matrix of only the HOMO-HOMO pairs by checking if the both SFOs are occupied. If not, make the overlap 0
        pauli_overlap_matrix = self.overlap_matrix.copy()
        for i, orb1 in enumerate(self.frag1_sfos):
            for j, orb2 in enumerate(self.frag2_sfos):
                if not (orb1.is_occupied and orb2.is_occupied):
                    pauli_overlap_matrix[i, j] = 0
                # magnitude of the overlap is most important; so make every value positive.
                else:
                    pauli_overlap_matrix[i, j] = abs(pauli_overlap_matrix[i, j])

        # Determine indices in the matrix (corresponding to SFO1-SFO2 pairs) sorted by the largest overlap
        indices = np.unravel_index(np.argsort(-pauli_overlap_matrix, axis=None)[:n_pairs], pauli_overlap_matrix.shape)
        pairs = [OrbitalPair(self.frag1_sfos[i], self.frag2_sfos[j], float(self.overlap_matrix[i, j])) for i, j in zip(*indices)]

        # select the n_pairs most destabilizing pairs
        return [pair for pair in pairs if pair.orb1.is_occupied and pair.orb2.is_occupied]

    def get_most_stabilizing_oi_pairs(self, n_pairs: int = 4) -> list[OrbitalPair]:
        """
        Determines which SFO pairs have the most favorable orbital interactions.
        Returns a list with the user-defined number of pairs with its first entry the pair that has the most stabilizing effect, and so on.
        """
        stabilization_matrix = self.stabilization_matrix

        # Determine indices in the matrix (corresponding to SFO1-SFO2 pairs) sorted by the largest stabilization
        indices = np.unravel_index(np.argsort(-stabilization_matrix, axis=None)[:n_pairs], stabilization_matrix.shape)
        pairs = [OrbitalPair(self.frag1_sfos[i], self.frag2_sfos[j], float(self.overlap_matrix[i, j])) for i, j in zip(*indices)]

        # select the n_pairs most destabilizing pairs
        return [pair for pair in pairs if (pair.orb1.is_occupied and not pair.orb2.is_occupied) or (not pair.orb1.is_occupied and pair.orb2.is_occupied)]
