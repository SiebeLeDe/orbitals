"""
Module containing classes that stores information of the complex calculation in fragment analysis calculations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
import pathlib as pl

import attrs
from scm.plams import KFFile

from orb_analysis.fragment.fragment import Fragment, create_fragment
from orb_analysis.complex.complex import Complex, create_complex

from orb_analysis.sfo.sfo import SFO

# --------------------Interface Function(s)-------------------- #


def create_calc_analyser(path_to_rkf_file: str | pl.Path, n_fragments: int = 2) -> CalcAnalyzer:
    """
    Main function that the user should use to create a :FACalcAnalyser: object. The function will automatically detect whether the calculation is restricted or unrestricted.

    Args:
        path_to_rkf_file (str): Path to the rkf file of the complex calculation.
        n_fragments (int, optional): Number of fragments in the calculation. Defaults to 2.

    Returns:
        FACalcAnalyser: A :FACalcAnalyser: object that contains information about the complex calculation.
    """
    path_to_rkf_file = pl.Path(path_to_rkf_file)
    kf_file = KFFile(str(path_to_rkf_file))

    if not kf_file.sections():  # type: ignore
        raise ValueError(f"The KFFile is empty. Please check the path to the KFFile. Current path is: {path_to_rkf_file}")

    # Here, we create necessary instances that store all the relevant information about orbitals. It includes
    # - :CalcInfo: An instance that contains general information about the calculation such as restricted or unrestricted, relativistic or non-relativistic, etc.
    # - :Complex: An instance that contains information about the complex calculation (Molecular Orbitals)
    # - A list of :Fragment: objects that contain information about the fragment calculation and the fragments respectively (Symmetrized Fragment Orbitals).
    name = path_to_rkf_file.parent.name
    calc_info = CalcInfo(kf_file=kf_file)
    complex = create_complex(name=name, kf_file=kf_file, restricted_calc=calc_info.restricted)
    fragments = [create_fragment(kf_file=kf_file, frag_index=i+1, restricted_calc=calc_info.restricted) for i in range(n_fragments)]

    if calc_info.restricted:
        return RestrictedCalcAnalyser(name=name, kf_file=kf_file, calc_info=calc_info, complex=complex, fragments=fragments)

    return UnrestrictedCalcAnalyser(name=name, kf_file=kf_file, calc_info=calc_info, complex=complex, fragments=fragments)


# --------------------Classes-------------------- #

@attrs.define
class CalcInfo:
    """
    This class contains information about the orbitals present in the complex calculation
    """
    kf_file: KFFile
    restricted: bool = True
    relativistic: bool = False
    symmetry: bool = False

    def __attrs_post_init__(self):
        # First, get relevant terms such as symmetry group label, unrestricted, relativistic, etc.
        if str(self.kf_file.read("Symmetry", "grouplabel")).split()[0].lower() not in ["nosym"]:
            self.symmetry = True

        if int(self.kf_file.read("General", "nspin") != 1):
            self.restricted = False

        if int(self.kf_file.read("General", "ioprel") != 0):
            self.relativistic = True


@attrs.define
class CalcAnalyzer(ABC):
    """
    This class contains information about the orbitals present in the complex calculation
    """
    name: str
    calc_info: CalcInfo
    kf_file: KFFile
    complex: Complex
    fragments: Sequence[Fragment] = attrs.field(default=list)

    @abstractmethod
    def get_sfo_overlap(self, sfo1: str | SFO, sfo2: str | SFO) -> float:
        pass

    @abstractmethod
    def get_sfo_gross_population(self, fragment: int, sfo: str | SFO) -> float:
        pass

    @abstractmethod
    def get_sfo_orbital_energy(self, fragment: int, sfo: str | SFO) -> float:
        pass

    @abstractmethod
    def get_sfo_occupation(self, fragment: int, sfo: str | SFO) -> float:
        pass


class RestrictedCalcAnalyser(CalcAnalyzer):
    """
    This class contains information about the complex calculation.
    """

    def get_sfo_overlap(self, sfo1: str | SFO, sfo2: str | SFO):

        if not isinstance(sfo1, SFO):
            sfo1 = SFO.from_label(sfo1)

        if not isinstance(sfo2, SFO):
            sfo2 = SFO.from_label(sfo2)

        return self.fragments[0].get_overlap(
            kf_file=self.kf_file,
            symmetry=self.calc_info.symmetry,
            irrep1=sfo1.symmetry,
            index1=sfo1.index,
            irrep2=sfo2.symmetry,
            index2=sfo2.index)

    def get_sfo_gross_population(self, fragment: int, sfo: str | SFO):
        if not isinstance(sfo, SFO):
            sfo = SFO.from_label(sfo)

        if not self.calc_info.symmetry:
            return self.fragments[fragment-1].get_gross_population(irrep="A", index=sfo.index)
        return self.fragments[fragment-1].get_gross_population(irrep=sfo.symmetry, index=sfo.index)

    def get_sfo_orbital_energy(self, fragment: int, sfo: str | SFO):
        if not isinstance(sfo, SFO):
            sfo = SFO.from_label(sfo)
        return self.fragments[fragment-1].get_orbital_energy(irrep=sfo.symmetry, index=sfo.index)

    def get_sfo_occupation(self, fragment: int, sfo: str | SFO):
        if not isinstance(sfo, SFO):
            sfo = SFO.from_label(sfo)
        return self.fragments[fragment-1].get_occupation(irrep=sfo.symmetry, index=sfo.index)


class UnrestrictedCalcAnalyser(CalcAnalyzer):
    """
    This class contains information about the complex calculation.
    """

    def get_sfo_overlap(self, sfo1: str | SFO, sfo2: str | SFO):
        raise NotImplementedError

    def get_sfo_gross_population(self, fragment: int, sfo: str | SFO):
        raise NotImplementedError

    def get_sfo_orbital_energy(self, fragment: int, sfo: str | SFO):
        raise NotImplementedError

    def get_sfo_occupation(self, fragment: int, sfo: str | SFO):
        raise NotImplementedError
