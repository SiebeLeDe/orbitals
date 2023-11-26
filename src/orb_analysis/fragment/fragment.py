"""
Module containing the :Fragment: class. It stores information about fragments present in fragment analysis calculation.
"""
from abc import ABC, abstractmethod
from functools import lru_cache

import attrs
import numpy as np
from scm.plams import KFFile

from orb_analysis.fragment.fragmentdata import FragmentData, create_fragment_data


# --------------------Interface Function(s)-------------------- #

def create_fragment(frag_index: int, kf_file: KFFile, restricted_calc: bool):
    """
    Creates a fragment object from the kf_file. The type of fragment object depends on the calculation type (restricted or unrestricted).
    """
    fragment_data = create_fragment_data(restricted_calc, frag_index, kf_file)

    if restricted_calc:
        return RestrictedFragment(fragment_data)
    else:
        return UnrestrictedFragment(fragment_data)


# ------------------- Helper Functions -------------------- #

@lru_cache(maxsize=1)
def get_overlap_matrix(kf_file: KFFile, irrep: str):
    """
    Returns the overlap matrix from the kf file as a numpy array.
    Note that this is a seperate function due to memory considerations as the matrix can be quite large.
    For that reason, @lru_cache is used here.
    """
    np.array(kf_file.read(irrep, 'S-CoreSFO'))


@lru_cache(maxsize=2)
def get_frag_sfo_index_mapping_to_total_sfo_index(kf_file: KFFile, frozen_cores_per_irrep_tuple: tuple[str, int], use_symmetry: bool) -> dict[int, dict[str, list[int]]]:
    """
    Function that creates a mapping (in the form of a nested dictionary) between the SFO indices of the fragments and the total SFO indices.
    The dict looks like this for a c3v calculation with two fragments:
    {
        1: {
            "A1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "E1:1": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            "E1:2": [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        },
        2: {
            "A1": [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
            "B2": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            "E1:1": [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
            "E1:2": [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
        },
    }

    This function is used in the get_overlap method in the Fragment class and makes sure that the indices of fragment 2 are shifted by the number of SFOs in fragment 1.
    It also takes into account the different irreps, such as 15_A1 may be 15 in fragment 1 and 41 in fragment 2.

    """
    sfo_indices: list[int] = kf_file.read("SFOs", "isfo", return_as_list=True)  # type: ignore
    frag_indices: list[int] = kf_file.read("SFOs", "fragment", return_as_list=True)  # type: ignore
    irreps_each_sfo = kf_file.read("SFOs", "subspecies", return_as_list=True).split()  # type: ignore
    frozen_cores_per_irrep: dict[str, int] = dict(frozen_cores_per_irrep_tuple)  # type: ignore # frozen_cores_per_irrep is a tuple, but we want a dict

    mapping_dict = {}

    # if the calculation did not use symmetry, we can skip the symmetry part
    if not use_symmetry:
        for sfo_index, frag_index in zip(sfo_indices, frag_indices):
            if frag_index not in mapping_dict:
                mapping_dict[frag_index] = {"A": []}

            frozen_core_shift = frozen_cores_per_irrep["A"]
            mapping_dict[frag_index]["A"].append(sfo_index + frozen_core_shift)
        return mapping_dict

    # Otherwise, we have to take into account the irreps
    for sfo_index, frag_index, irrep in zip(sfo_indices, frag_indices, irreps_each_sfo):
        if frag_index not in mapping_dict:
            mapping_dict[frag_index] = {}

        if irrep not in mapping_dict[frag_index]:
            mapping_dict[frag_index][irrep] = []

        # Note: index is shifted also by the frozen core orbitals
        frozen_core_shift = frozen_cores_per_irrep[irrep] if irrep in frozen_cores_per_irrep else 0
        mapping_dict[frag_index][irrep].append(sfo_index + frozen_core_shift)

    return mapping_dict

# --------------------Fragment Classes-------------------- #


@attrs.define
class Fragment(ABC):
    """
    Interface class for fragments. This class contains methods that are shared between restricted and unrestricted fragments.
    """
    fragment_data: FragmentData

    @property
    def name(self):
        return self.fragment_data.name

    @abstractmethod
    def get_overlap(self, symmetry: bool, kf_file: KFFile, irrep1: str, index1: int, irrep2: str, index2: int) -> float:
        """ Returns the overlap between two orbitals in a.u. """
        pass

    @abstractmethod
    def get_orbital_energy(self, irrep: str, index: int) -> float:
        """ Returns the orbital energy """
        pass

    @abstractmethod
    def get_gross_population(self, irrep: str, index: int) -> float:
        """ Returns the gross population """
        pass

    @abstractmethod
    def get_occupation(self, irrep: str, index: int) -> float:
        pass


class RestrictedFragment(Fragment):

    def get_overlap(self, symmetry: bool, kf_file: KFFile, irrep1: str, index1: int, irrep2: str, index2: int):

        if not symmetry:
            irrep1 = "A"
            irrep2 = "A"

        # Note: the overlap matrix is stored in the rkf file as a lower triangular matrix. Thus, the index is calculated as follows:
        # index = max_index * (max_index - 1) // 2 + min_index - 1

        frozen_cores_per_irrep = tuple(sorted(self.fragment_data.n_frozen_cores_per_irrep.items()))
        index_mapping = get_frag_sfo_index_mapping_to_total_sfo_index(kf_file, frozen_cores_per_irrep, symmetry)
        index1 = index_mapping[1][irrep1][index1-1]
        index2 = index_mapping[2][irrep2][index2-1]

        min_index, max_index = sorted([index1, index2])
        overlap_index = max_index * (max_index - 1) // 2 + min_index - 1
        overlap_matrix = np.array(kf_file.read(irrep1, 'S-CoreSFO'))  # type: ignore
        return abs(overlap_matrix[overlap_index])

    def get_orbital_energy(self, irrep: str, index: int):
        return self.fragment_data.orb_energies[irrep][index-1]

    def get_gross_population(self, irrep: str, index: int):
        return self.fragment_data.gross_populations[irrep][index-1]

    def get_occupation(self, irrep: str, index: int):
        return self.fragment_data.occupations[irrep][index-1]


class UnrestrictedFragment(Fragment):
    def get_overlap(self, kf_file: KFFile, irrep1: str, index1: int, irrep2: str, index2: int):
        raise NotImplementedError("Unrestricted fragments are currently not supported.")

    def get_orbital_energy(self, irrep: str, index: int):
        raise NotImplementedError("Unrestricted fragments are currently not supported.")

    def get_gross_population(self, irrep: str, index: int):
        raise NotImplementedError("Unrestricted fragments are currently not supported.")

    def get_occupation(self, irrep: str, index: int):
        raise NotImplementedError("Unrestricted fragments are currently not supported.")
