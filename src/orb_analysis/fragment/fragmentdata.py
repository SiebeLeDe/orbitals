from abc import ABC
import attrs
import numpy as np
from orb_analysis.custom_types import Array1D
from scm.plams import KFFile
from typing import Sequence
from orb_analysis.orbital_functions import get_frozen_cores_per_irrep, get_gross_populations, get_restricted_fragment_properties


def create_fragment_data(restricted_frag: bool, frag_index: int, kf_file: KFFile):
    """
    Creates a fragment data object from the kf_file. The type of fragment object depends on the calculation type (restricted or unrestricted).
    """
    # There are two important terms required for reading the fragment data: the indices of the SFOs and their corresponding symlabels (=irreps)
    # These can be passed to the get_fragment_properties function to get the data for the current fragment
    sfo_indices_one_frag = [i for i, sfo_frag_index in enumerate(kf_file.read("SFOs", "fragment")) if sfo_frag_index == frag_index]  # type: ignore

    # Here we select the symlabels that belong to the current fragment (see self.frag_index)
    frag_symlabels_each_sfo = list(kf_file.read("SFOs", "subspecies").split())  # type: ignore
    frag_symlabels_each_sfo = [frag_symlabels_each_sfo[i] for i in sfo_indices_one_frag]

    if restricted_frag:
        return _create_restricted_fragment_data(kf_file, frag_index, sfo_indices_one_frag, frag_symlabels_each_sfo)
    else:
        return _create_restricted_fragment_data(kf_file, frag_index, sfo_indices_one_frag, frag_symlabels_each_sfo)


def _create_restricted_fragment_data(kf_file: KFFile, frag_index: int, sfo_indices_one_frag: Sequence[int], frag_symlabels_each_sfo: Sequence[str]):

    frag_name = kf_file.read("SFOs", "fragtype").split()[sfo_indices_one_frag[0]]  # type: ignore

    # Get the number of frozen cores per irrep because the SFO indices are shifted by the number of frozen cores for gross populations and overlap analyses
    n_frozen_cores_per_irrep = get_frozen_cores_per_irrep(kf_file, frag_index)

    # Get regular properties such as occupations and orbital energies
    data_dic_to_be_unpacked = get_restricted_fragment_properties(kf_file, sfo_indices_one_frag, frag_symlabels_each_sfo)

    # Gross populations is special due to the frozen cores, so we have to do some extra work here
    data_dic_to_be_unpacked["gross_populations"] = get_gross_populations(kf_file, frag_index)

    new_fragment_data = FragmentData(name=frag_name, frag_index=frag_index, n_frozen_cores_per_irrep=n_frozen_cores_per_irrep, **data_dic_to_be_unpacked)
    return new_fragment_data


@attrs.define
class FragmentData(ABC):
    """
    Stores the symmetrized fragment orbital (SFO) data from the rkf files. The data includes:
        - Gross Populations
        - Orbital Energies
        - Occupations
        - Number of frozen cores per irrep

    See the specific fragment classes for more information about the format of data stored in the dictionaries.
    """
    name: str
    frag_index: int  # 1 or 2
    orb_energies: dict[str, Array1D[np.float64]]
    occupations: dict[str, Array1D[np.float64]]
    gross_populations: dict[str, Array1D[np.float64]]
    n_frozen_cores_per_irrep: dict[str, int]


@attrs.define
class RestrictedFragmentData(FragmentData):
    """
    The data is stored in dictionaries with the symlabels as keys. For example:
        - self.occupations[IRREP1] returns an array with the occupations of all IRREP1 orbitals.
        - self.occupations[IRREP2] returns an array with the occupations of all IRREP2 orbitals.
        - self.orb_energies[IRREP1] returns an array with the orbital energies of all IRREP1 orbitals.
        - self.gross_populations[IRREP1] returns an array with the gross populations of all IRREP1 orbitals.

    Examples:
        orb_energies = {
            "A1": [-1.0, -2.0, 3.0],
            "A2": [-4.0, -5.0, 6.0],
            "E1": [-7.0, -8.0, 9.0],
            "E2": [-10.0, -11.0, 12.0],
        }

        n_frozen_cores_per_irrep = {
            "A1": 4,
            "A2": 0,
            "E1": 1,
            "E2": 1,
        }
        etc. for occupations and gross populations
    """


@attrs.define
class UnrestrictedFragmentData(FragmentData):
    """
    The data is stored in dictionaries with the symlabels as keys. For example:
        - self.occupations[SPIN][IRREP1] returns an array with the occupations of all IRREP1 orbitals.
        - self.occupations[SPIN][IRREP2] returns an array with the occupations of all IRREP2 orbitals.
        - self.orb_energies[SPIN][IRREP1] returns an array with the orbital energies of all IRREP1 orbitals.
        - self.gross_populations[SPIN][IRREP1] returns an array with the gross populations of all IRREP1 orbitals.

    Examples:
        orb_energies = {
            "A": {
                "A1": [-1.0, -2.0, 3.0],
                "A2": [-4.0, -5.0, 6.0],
                "E1": [-7.0, -8.0, 9.0],
                "E2": [-10.0, -11.0, 12.0],
            },
            "B": {
                "A1": [-1.0, -2.0, 3.0],
                "A2": [-4.0, -5.0, 6.0],
                "E1": [-7.0, -8.0, 9.0],
                "E2": [-10.0, -11.0, 12.0],
        }
        etc. for occupations and gross populations

        The frozen cores per irrep format remains the same
    """
    orb_energies: dict[str, dict[str, Array1D[np.float64]]]
    gross_populations: dict[str, dict[str, Array1D[np.float64]]]
    n_frozen_cores_per_irrep: dict[str, dict[str, int]]
