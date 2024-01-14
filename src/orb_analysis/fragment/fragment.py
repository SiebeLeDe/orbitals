"""
Module containing the :Fragment: class. It stores information about fragments present in fragment analysis calculation.
"""
from abc import ABC, abstractmethod
from functools import lru_cache

import attrs
import numpy as np
from scm.plams import KFFile
from orb_analysis.analyzer.calc_info import CalcInfo

from orb_analysis.custom_types import RestrictedProperty, SpinTypes

from orb_analysis.fragment.fragmentdata import FragmentData, create_restricted_fragment_data, create_unrestricted_fragment_data, RestrictedFragmentData, UnrestrictedFragmentData
from orb_analysis.orb_functions.orb_functions import filter_orbitals
from orb_analysis.orbital.orbital import SFO


# --------------------Interface Function(s)-------------------- #


def create_restricted_fragment(frag_index: int, kf_file: KFFile, calc_info: CalcInfo):
    """
    Creates a fragment object from the kf_file. The type of fragment object depends on the calculation type (restricted or unrestricted).
    """
    fragment_data = create_restricted_fragment_data(kf_file, frag_index)
    return RestrictedFragment(fragment_data=fragment_data, calc_info=calc_info)


def create_unrestricted_fragment(frag_index: int, kf_file: KFFile, calc_info: CalcInfo):
    """
    Creates a fragment object from the kf_file. The type of fragment object depends on the calculation type (restricted or unrestricted).
    """
    fragment_data = create_unrestricted_fragment_data(kf_file, frag_index)
    return UnrestrictedFragment(fragment_data=fragment_data, calc_info=calc_info)


# ------------------- Helper Functions -------------------- #


@lru_cache(maxsize=1)
def get_overlap_matrix(kf_file: KFFile, irrep: str):
    """
    Returns the overlap matrix from the kf file as a numpy array.
    Note that this is a seperate function due to memory considerations as the matrix can be quite large.
    For that reason, @lru_cache is used here.
    """
    np.array(kf_file.read(irrep, "S-CoreSFO"))


@lru_cache(maxsize=2)
def get_frag_sfo_index_mapping_to_total_sfo_index(kf_file: KFFile, frozen_cores_per_irrep_tuple: tuple[str, int], uses_symmetry: bool) -> dict[int, dict[str, list[int]]]:
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

    # if the calculation did not use uses_symmetry, we can skip the uses_symmetry part
    if not uses_symmetry:
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
    calc_info: CalcInfo

    @property
    def name(self):
        return self.fragment_data.name

    def _get_overlap(self, irrep: bool, kf_file: KFFile, irrep1: str, index1: int, irrep2: str, index2: int, spin: str = SpinTypes.A) -> float:
        # Note: the overlap matrix is stored in the rkf file as a lower triangular matrix. Thus, the index is calculated as follows:
        # index = max_index * (max_index - 1) // 2 + min_index - 1
        frozen_cores_per_irrep = tuple(sorted(self.fragment_data.n_frozen_cores_per_irrep.items()))
        index_mapping = get_frag_sfo_index_mapping_to_total_sfo_index(kf_file, frozen_cores_per_irrep, irrep)
        index1 = index_mapping[1][irrep1][index1 - 1]
        index2 = index_mapping[2][irrep2][index2 - 1]

        min_index, max_index = sorted([index1, index2])
        overlap_index = max_index * (max_index - 1) // 2 + min_index - 1
        variable = f"S-CoreSFO_{spin}" if spin == SpinTypes.B else "S-CoreSFO"
        overlap_matrix = np.array(kf_file.read(irrep1, variable))
        return overlap_matrix[overlap_index]

    @abstractmethod
    def get_overlap(self, irrep: bool, kf_file: KFFile, irrep1: str, index1: int, irrep2: str, index2: int) -> float:
        """Returns the overlap between two orbitals in a.u."""
        pass

    @abstractmethod
    def get_orbital_energy(self, irrep: str, index: int, spin: str = SpinTypes.A) -> float:
        """Returns the orbital energy an active SFO"""
        pass

    @abstractmethod
    def get_gross_population(self, irrep: str, index: int, spin: str = SpinTypes.A) -> float:
        """Returns the gross population an active SFO"""
        pass

    @abstractmethod
    def get_occupation(self, irrep: str, index: int, spin: str = SpinTypes.A) -> float:
        """Returns the occupation of an active SFO"""
        pass

    def _get_sfos(self, orb_range: tuple[int, int], orb_irrep: str | None, spin: str, orb_energies: RestrictedProperty, occupations: RestrictedProperty, gross_pop: RestrictedProperty) -> list[SFO]:
        max_occupied_orbitals, max_unoccupied_orbitals = orb_range
        irreps = [orb_irrep.upper()] if orb_irrep is not None else self.fragment_data.frag_irreps
        sfos: list[SFO] = []

        # Then, flatten the data to a list of SFOs
        for irrep in self.fragment_data.frag_irreps:
            for index, energy, occ in zip(range(1, len(orb_energies[irrep]) + 1), orb_energies[irrep], occupations[irrep]):
                pop = self.get_gross_population(irrep if self.calc_info.symmetry else "A", index, spin)
                sfos.append(SFO(index=index, irrep=irrep, spin=spin, energy=energy, gross_pop=pop, occupation=occ))

        return filter_orbitals(sfos, max_occupied_orbitals, max_unoccupied_orbitals, irreps)

    @abstractmethod
    def get_sfos(self, irrep: str | None = None, spin: str = SpinTypes.A) -> int:
        """Returns the index of the highest occupied (molecular) orbital (HOMO) of which a subset can be returned bij specifying the irrep ands spin"""


@attrs.define
class RestrictedFragment(Fragment):
    fragment_data: RestrictedFragmentData

    def get_overlap(self, uses_symmetry: bool, kf_file: KFFile, irrep1: str, index1: int, irrep2: str, index2: int) -> float:
        if irrep1 != irrep2:
            return 0.0

        if not uses_symmetry:
            irrep1, irrep2 = "A", "A"

        return self._get_overlap(uses_symmetry, kf_file, irrep1, index1, irrep2, index2, SpinTypes.A)

    def get_orbital_energy(self, irrep: str, index: int, spin: str = SpinTypes.A) -> float:
        return self.fragment_data.orb_energies[irrep][index - 1]

    def get_gross_population(self, irrep: str, index: int, spin: str = SpinTypes.A) -> float:
        return self.fragment_data.gross_populations[irrep][index - 1]

    def get_occupation(self, irrep: str, index: int, spin: str = SpinTypes.A):
        return self.fragment_data.occupations[irrep][index - 1]

    def get_sfos(self, orbital_range: tuple[int, int], orb_irrep: str | None = None, spin: str | None = SpinTypes.A) -> list[SFO]:
        # First get the data
        orb_energies = self.fragment_data.orb_energies
        gross_pop = self.fragment_data.gross_populations
        occupations = self.fragment_data.occupations

        return self._get_sfos(orbital_range, orb_irrep, SpinTypes.A, orb_energies, occupations, gross_pop)


@attrs.define
class UnrestrictedFragment(Fragment):
    fragment_data: UnrestrictedFragmentData

    def get_overlap(self, uses_symmetry: bool, kf_file: KFFile, irrep1: str, index1: int, irrep2: str, index2: int, spin: str) -> float:
        if irrep1 != irrep2:
            return 0.0

        if not uses_symmetry:
            irrep1, irrep2 = "A", "A"
        return self._get_overlap(uses_symmetry, kf_file, irrep1, index1, irrep2, index2, spin)

    def get_orbital_energy(self, irrep: str, index: int, spin: str) -> float:
        return self.fragment_data.orb_energies[spin][irrep][index - 1]

    def get_gross_population(self, irrep: str, index: int, spin: str) -> float:
        return self.fragment_data.gross_populations[spin][irrep][index - 1]

    def get_occupation(self, irrep: str, index: int, spin: str) -> float:
        return self.fragment_data.occupations[spin][irrep][index - 1]

    def get_sfos(self, orbital_range: tuple[int, int], orb_irrep: str | None = None, spin: str = SpinTypes.A) -> list[SFO]:
        # First get the data
        orb_energies = self.fragment_data.orb_energies[spin]
        gross_pop = self.fragment_data.gross_populations[spin]
        occupations = self.fragment_data.occupations[spin]

        return self._get_sfos(orbital_range, orb_irrep, spin, orb_energies, occupations, gross_pop)
