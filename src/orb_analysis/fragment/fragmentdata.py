from abc import ABC

import attrs
import numpy as np
from scm.plams import KFFile

from orb_analysis.custom_types import Array1D, RestrictedProperty, SpinTypes, UnrestrictedProperty
from orb_analysis.orb_functions.sfo_functions import get_frag_name, get_fragment_properties, get_frozen_cores_per_irrep, get_gross_populations, get_ordered_irreps_of_one_frag

# --------------------Helper Functions-------------------- #


def flatten_data(data: RestrictedProperty, ordered_irreps: list[str]) -> Array1D[np.float64]:
    """Flattens the data from a dictionary with {irrep: [data]} to a list with [data]."""
    try:
        return np.concatenate([data[irrep] for irrep in ordered_irreps])
    except KeyError:
        return data["A"]  # When the complex has has "nosym"


# --------------------Interface Function(s)-------------------- #


def create_restricted_fragment_data(kf_file: KFFile, frag_index: int):
    """
    Creates a restricted fragment data object from the kf_file.
    """
    data_dic_to_be_unpacked = {}

    # Get the fragment name
    frag_name = get_frag_name(kf_file, frag_index)

    # Get the number of frozen cores per irrep because the SFO indices are shifted by the number of frozen cores for gross populations and overlap analyses
    n_frozen_cores_per_irrep = get_frozen_cores_per_irrep(kf_file, frag_index)

    # Get regular properties such as occupations and orbital energies
    # the returned data is a dictionary with {property: {spin: {irrep: [data]}}} with property being either "orb_energies" or "occupations" and spin being either "A" or "B" (see `SpinTypes`)
    data_dic_to_be_unpacked = get_fragment_properties(kf_file, frag_index)
    data_dic_to_be_unpacked = {key: value[SpinTypes.A] for key, value in data_dic_to_be_unpacked.items()}  # Here we want to get rid of the spin key because restricted fragments don't have spin

    # Gross populations is special due to the frozen cores, so we have to do some extra work here
    data_dic_to_be_unpacked["gross_populations"] = get_gross_populations(kf_file, frag_index)[SpinTypes.A]

    frag_irreps = get_ordered_irreps_of_one_frag(kf_file, frag_index)

    new_fragment_data = RestrictedFragmentData(name=frag_name, frag_index=frag_index, n_frozen_cores_per_irrep=n_frozen_cores_per_irrep, **data_dic_to_be_unpacked, frag_irreps=frag_irreps)
    return new_fragment_data


def create_unrestricted_fragment_data(kf_file: KFFile, frag_index: int):
    """
    Creates an unrestricted fragment data object from the kf_file.
    """
    data_dic_to_be_unpacked = {}

    # Get the fragment name
    frag_name = get_frag_name(kf_file, frag_index)

    n_frozen_cores_per_irrep = get_frozen_cores_per_irrep(kf_file, frag_index)

    data_dic_to_be_unpacked = get_fragment_properties(kf_file, frag_index)

    data_dic_to_be_unpacked["gross_populations"] = get_gross_populations(kf_file, frag_index)

    frag_irreps = get_ordered_irreps_of_one_frag(kf_file, frag_index)

    new_fragment_data = UnrestrictedFragmentData(name=frag_name, frag_index=frag_index, n_frozen_cores_per_irrep=n_frozen_cores_per_irrep, **data_dic_to_be_unpacked, frag_irreps=frag_irreps)
    return new_fragment_data


# --------------------Fragment Data Classes -------------------- #


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
    orb_energies: RestrictedProperty
    occupations: RestrictedProperty
    gross_populations: RestrictedProperty
    n_frozen_cores_per_irrep: dict[str, int]
    frag_irreps: list[str]

    @property
    def uses_frag_symmetry(self) -> bool:
        return len(self.frag_irreps) > 1


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

    orb_energies: UnrestrictedProperty
    gross_populations: UnrestrictedProperty
    occupations: UnrestrictedProperty


def main():
    import pathlib as pl

    from numpy import set_printoptions

    set_printoptions(precision=3, suppress=True)

    current_dir = pl.Path(__file__).parent
    rkf_dir = current_dir.parent.parent.parent / "test" / "fixtures" / "rkfs" / "different_sym"

    rkf_file = "restricted_largecore_differentfragsym_c4v_full_full.adf.rkf"
    # rkf_file = "unrestricted_largecore_fragsym_nosym_full.adf.rkf"
    kf_file = KFFile(str(rkf_dir / rkf_file))

    data = create_restricted_fragment_data(frag_index=2, kf_file=kf_file)
    print(data.frag_irreps)

    # grospop = get_gross_populations(kf_file, frag_index=2)
    # print(grospop)


if __name__ == "__main__":
    main()
