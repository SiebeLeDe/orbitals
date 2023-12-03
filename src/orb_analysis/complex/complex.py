"""
Module containing classes that stores information of the complex calculation in fragment analysis calculations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import attrs
from scm.plams import KFFile
from orb_analysis.complex.complex_data import ComplexData, RestrictedComplexData, UnrestrictedComplexData, create_complex_data
from orb_analysis.custom_types import SpinTypes
from orb_analysis.orb_functions.orb_functions import filter_orbitals
from orb_analysis.orbital.orbital import MO


# --------------------Interface Function(s)-------------------- #


def create_complex(name: str, kf_file: KFFile, restricted_calc: bool) -> Complex:
    """
    Main function that the user could use to create a :Complex: object.

    Args:
        name (str): Name of the complex calculation.
        kf_file (KFFile): KFFile object of the complex calculation.
        complex_data (ComplexData): ComplexData object that contains information about the complex calculation.

    Returns:
        Complex: A :Complex: object that contains information about the complex calculation.
    """

    # Create complex data
    complex_data = create_complex_data(name, kf_file, restricted_calc)

    # Create complex instance
    if restricted_calc:
        return RestrictedComplex(name=name, kf_file=kf_file, complex_data=complex_data)

    return UnrestrictedComplex(name=name, kf_file=kf_file, complex_data=complex_data)


# --------------------Calc Info classes-------------------- #

@attrs.define
class Complex(ABC):
    """
    This class contains information about the complex calculation.
    """
    name: str
    kf_file: KFFile
    complex_data: ComplexData

    @abstractmethod
    def get_orbital_energy(self, irrep: str, index: int, spin: str) -> float:
        """ Returns the orbital energy """
        pass

    @abstractmethod
    def get_occupation(self, irrep: str, index: int, spin: str) -> float:
        pass

    @abstractmethod
    def get_mos(self, orb_range: tuple[int, int], orb_irrep: str | None = None, spin: str | None = SpinTypes.A) -> list[MO]:
        pass


class RestrictedComplex(Complex):
    complex_data: RestrictedComplexData

    """ This class contains methods for accessing information about the restricted molecular orbitals. """

    def get_orbital_energy(self, irrep: str, index: int, spin: str):
        return self.complex_data.orb_energies[irrep][index-1]

    def get_occupation(self, irrep: str, index: int, spin: str):
        return self.complex_data.occupations[irrep][index-1]

    def get_mos(self, orb_range: tuple[int, int], orb_irrep: str | None = None, spin: str | None = SpinTypes.A) -> list[MO]:
        max_occupied_orbitals, max_unoccupied_orbitals = orb_range
        irreps = [orb_irrep.upper()] if orb_irrep is not None else self.complex_data.irreps
        mos: list[MO] = []

        # First get the data
        orb_energies = self.complex_data.orb_energies
        occupations = self.complex_data.occupations

        # Then, flatten the data to a list of mos
        for irrep in self.complex_data.irreps:
            for index, energy, occ in zip(range(1, len(orb_energies[irrep])+1), orb_energies[irrep], occupations[irrep]):
                mos.append(MO(index=index, irrep=irrep, spin=SpinTypes.A, energy=energy, occupation=occ))

        return filter_orbitals(mos, max_occupied_orbitals, max_unoccupied_orbitals, irreps)


class UnrestrictedComplex(Complex):
    """ This class contains methods for accessing information about the unrestricted molecular orbitals. """
    complex_data: UnrestrictedComplexData

    def get_orbital_energy(self, irrep: str, index: int, spin: str):
        return self.complex_data.orb_energies[spin][irrep][index-1]

    def get_occupation(self, irrep: str, index: int, spin: str):
        return self.complex_data.occupations[spin][irrep][index-1]

    def get_mos(self, irrep: str, spin: str) -> list[MO]:
        raise NotImplementedError("This method is not implemented yet.")
