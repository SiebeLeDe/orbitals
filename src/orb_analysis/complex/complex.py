"""
Module containing classes that stores information of the complex calculation in fragment analysis calculations.
"""
from __future__ import annotations

from abc import ABC

import attrs
from scm.plams import KFFile
from orb_analysis.complex.complex_data import ComplexData, create_complex_data


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


class RestrictedComplex(Complex):
    """ This class contains methods for accessing information about the restricted molecular orbitals. """


class UnrestrictedComplex(Complex):
    """ This class contains methods for accessing information about the unrestricted molecular orbitals. """
