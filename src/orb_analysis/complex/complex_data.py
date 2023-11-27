"""
Module containing classes that stores information of the complex calculation in fragment analysis calculations.
"""
from __future__ import annotations
from scm.plams import KFFile
import attrs
from orb_analysis.orb_functions.mo_functions import get_frozen_cores_per_irrep, get_complex_properties
from orb_analysis.custom_types import RestrictedProperty, SpinTypes, UnrestrictedProperty


# --------------------Interface Function(s)-------------------- #

def create_complex_data(name: str, kf_file: KFFile, restricted_calc: bool) -> ComplexData:
    """
    Main function that the user could use to create a :ComplexData: object.

    Args:
        name (str): Name of the complex calculation.
        kf_file (KFFile): KFFile object of the complex calculation.
        restricted_calc (bool): Whether the calculation is restricted or unrestricted.

    Returns:
        ComplexData: A :ComplexData: object that contains information about the complex calculation.
    """
    data_to_be_unpacked = get_complex_properties(kf_file, restricted_calc)
    n_frozen_cores_per_irrep = get_frozen_cores_per_irrep(kf_file)

    if restricted_calc:
        data_to_be_unpacked = {key: value[SpinTypes.A] for key, value in data_to_be_unpacked.items()}  # Here we want to get rid of the spin key because restricted fragments don't have spin
        return RestrictedComplexData(name=name, **data_to_be_unpacked, n_frozen_cores_per_irrep=n_frozen_cores_per_irrep)

    return UnrestrictedComplexData(name=name, **data_to_be_unpacked, n_frozen_cores_per_irrep=n_frozen_cores_per_irrep)


# --------------------Calc Info classes-------------------- #


@attrs.define
class ComplexData:
    """
    Stores data about molecular orbitals (MOs) from the rkf files. The data includes:
        - Orbital Energies
        - Occupations
        - Number of frozen cores per irrep

    See the specific :Complex: classes for more information about the format of data stored in the dictionaries.
    """
    name: str
    orb_energies: RestrictedProperty
    occupations: RestrictedProperty
    n_frozen_cores_per_irrep: dict[str, int]


class RestrictedComplexData(ComplexData):
    """ Stores data about molecular orbitals (MOs) from the rkf files. The data includes:
        - Orbital Energies
        - Occupations
        - Number of frozen cores per irrep

    in the format:
        - Orbital Energies: {"IRREP1": [orb_energies], "IRREP2": [orb_energies], ...}
        - Occupations: {"IRREP1": [occupations], "IRREP2": [occupations], ...}
        - Number of frozen cores per irrep: {"IRREP1": n_frozen_cores, "IRREP2": n_frozen_cores, ...}
    """


@attrs.define
class UnrestrictedComplexData(ComplexData):
    """ Stores data about molecular orbitals (MOs) in unrestricted format from the rkf files. The data includes:
        - Orbital Energies
        - Occupations
        - Number of frozen cores per irrep

    in the format:
        - Orbital Energies:
            {
                "A": {"IRREP1": [orb_energies], "IRREP2": [orb_energies]},
                "B": {"IRREP1": [orb_energies], "IRREP2": [orb_energies]},
            }
        - Occupations:
            {
                "A": {"IRREP1": [occupations], "IRREP2": [occupations]},
                "B": {"IRREP1": [occupations], "IRREP2": [occupations]}
            }
        - Number of frozen cores per irrep: {"IRREP1": n_frozen_cores, "IRREP2": n_frozen_cores, ...}
    """
    name: str
    orb_energies: UnrestrictedProperty
    occupations: UnrestrictedProperty
    n_frozen_cores_per_irrep: dict[str, int]
