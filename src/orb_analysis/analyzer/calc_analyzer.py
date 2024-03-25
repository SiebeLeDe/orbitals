"""
Module containing classes that stores information of the complex calculation in fragment analysis calculations.
"""

from __future__ import annotations

import pathlib as pl
from abc import ABC, abstractmethod
from typing import Sequence

import attrs
import numpy as np
from scm.plams import KFFile

from orb_analysis.analyzer.calc_info import CalcInfo
from orb_analysis.complex.complex import Complex, create_complex
from orb_analysis.custom_types import SpinTypes
from orb_analysis.fragment.fragment import Fragment, RestrictedFragment, UnrestrictedFragment, create_restricted_fragment, create_unrestricted_fragment
from orb_analysis.log_messages import calc_analyzer_call_message
from orb_analysis.orbital.orbital import SFO
from orb_analysis.orbital_manager.orb_manager import MOManager, SFOManager

# --------------------Interface Method(s)-------------------- #


def create_calc_analyser(path_to_rkf_file: str | pl.Path, n_fragments: int = 2, name: str | None = None) -> CalcAnalyzer:
    """
    Main Method that the user should use to create a :FACalcAnalyser: object. The Method will automatically detect whether the calculation is restricted or unrestricted.

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
    name = path_to_rkf_file.parent.name + "/" + path_to_rkf_file.stem if name is None else name
    calc_info = CalcInfo(kf_file=kf_file)
    complex = create_complex(name=name, kf_file=kf_file, restricted_calc=calc_info.restricted)

    if calc_info.restricted:
        fragments = [create_restricted_fragment(kf_file=kf_file, frag_index=i + 1, calc_info=calc_info) for i in range(n_fragments)]
        return RestrictedCalcAnalyser(name=name, kf_file=kf_file, calc_info=calc_info, complex=complex, fragments=fragments)

    fragments = [create_unrestricted_fragment(kf_file=kf_file, frag_index=i + 1, calc_info=calc_info) for i in range(n_fragments)]
    return UnrestrictedCalcAnalyser(name=name, kf_file=kf_file, calc_info=calc_info, complex=complex, fragments=fragments)


# --------------------Classes-------------------- #


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

    def __call__(self, orb_range: tuple[int, int] = (6, 6), irrep: str | None = None, spin: str = SpinTypes.A) -> str:
        sfos = self.get_sfo_orbitals(orb_range, orb_range, irrep, spin)
        mos = self.get_mo_orbitals(orb_range, irrep, spin)
        log_message = calc_analyzer_call_message(restricted=self.calc_info.restricted, calc_name=self.name, orb_range=orb_range, irrep=irrep, spin=spin)
        return log_message + str(sfos) + "\n\n" + str(mos)

    @abstractmethod
    def get_sfo_overlap(self, sfo1: str | SFO, sfo2: str | SFO) -> float:
        """Method that returns the overlap between two SFOs. Format input: "[index]_[irrep]_[spin]" (spin only for unrestricted), or SFO object."""
        pass

    @abstractmethod
    def get_sfo_gross_population(self, fragment: int, sfo: str | SFO) -> float:
        """Method that returns the gross population of a SFO in a fragment. Format input: "[index]_[irrep]_[spin]" (spin only for unrestricted), or SFO object."""
        pass

    @abstractmethod
    def get_sfo_orbital_energy(self, fragment: int, sfo: str | SFO) -> float:
        """Method that returns the orbital energy of a SFO in a fragment. Format input: "[index]_[irrep]_[spin]" (spin only for unrestricted), or SFO object."""
        pass

    @abstractmethod
    def get_sfo_occupation(self, fragment: int, sfo: str | SFO) -> float:
        """Method that returns the occupation of a SFO in a fragment. Format input: "[index]_[irrep]_[spin]" (spin only for unrestricted), or SFO object."""
        pass

    @abstractmethod
    def get_mo_orbitals(self, orb_range: tuple[int, int] = (-10, 10), irrep: str | None = None, spin: str | None = None) -> MOManager:
        """Method that returns (a part of) the molecular orbitals (MOs)."""
        pass

    @abstractmethod
    def get_sfo_orbitals(self, frag1_orb_range: tuple[int, int] = (-10, 10), frag2_orb_range: tuple[int, int] = (-10, 10), irrep: str | None = None, spin: str | None = None) -> SFOManager:
        pass


@attrs.define
class RestrictedCalcAnalyser(CalcAnalyzer):
    """
    This class contains information about the complex calculation.
    """

    fragments: Sequence[RestrictedFragment] = attrs.field(default=list)

    def _get_sfo(self, sfo: str | SFO) -> SFO:
        return SFO.from_label(sfo) if isinstance(sfo, str) else sfo

    def _get_fragment(self, fragment: int):
        return self.fragments[fragment - 1]

    def get_sfo_overlap(self, sfo1: str | SFO, sfo2: str | SFO):
        sfo1, sfo2 = self._get_sfo(sfo1), self._get_sfo(sfo2)
        return self._get_fragment(0).get_overlap(kf_file=self.kf_file, uses_symmetry=self.calc_info.symmetry, irrep1=sfo1.irrep, index1=sfo1.index, irrep2=sfo2.irrep, index2=sfo2.index)

    def get_sfo_gross_population(self, fragment: int, sfo: str | SFO):
        sfo = self._get_sfo(sfo)
        return self._get_fragment(fragment).get_gross_population(irrep=sfo.irrep, index=sfo.index)

    def get_sfo_orbital_energy(self, fragment: int, sfo: str | SFO):
        sfo = self._get_sfo(sfo)
        return self._get_fragment(fragment).get_orbital_energy(irrep=sfo.irrep, index=sfo.index)

    def get_sfo_occupation(self, fragment: int, sfo: str | SFO):
        sfo = self._get_sfo(sfo)
        return self._get_fragment(fragment).get_occupation(irrep=sfo.irrep, index=sfo.index)

    def get_mo_orbitals(self, orb_range: tuple[int, int] = (-10, 10), irrep: str | None = None, spin: str | None = None) -> MOManager:
        mos = self.complex.get_mos(orb_range=orb_range, orb_irrep=irrep, spin=spin)
        return MOManager(complex_mos=mos)

    def get_sfo_orbitals(self, frag1_orb_range: tuple[int, int] = (10, 10), frag2_orb_range: tuple[int, int] = (10, 10), irrep: str | None = None, spin: str | None = None) -> SFOManager:
        frag_sfos = [frag.get_sfos(homo_lumo_range, irrep) for homo_lumo_range, frag in zip([frag1_orb_range, frag2_orb_range], self.fragments)]
        frag_sfos[1] = frag_sfos[1][::-1]  # reverse the orbitals to go from LUMO+x -> HOMO-x to HOMO-x -> LUMO+x

        # First, we want to store frag1 orbitals from LUMO+x to HOMO-x for easier printing later on
        overlap_matrix = np.zeros(shape=(len(frag_sfos[0]), len(frag_sfos[1])))
        try:
            for i, frag1_orb in enumerate(frag_sfos[0]):
                for j, frag2_orb in enumerate(frag_sfos[1]):
                    overlap_matrix[i, j] = self.get_sfo_overlap(frag1_orb, frag2_orb)
        except KeyError:
            print("Detecting irrep error in getting the overlap matrix, skipping it as a result")
        return SFOManager(frag1_sfos=frag_sfos[0], frag2_sfos=frag_sfos[1], overlap_matrix=overlap_matrix)


@attrs.define
class UnrestrictedCalcAnalyser(CalcAnalyzer):
    """
    This class contains information about the complex calculation.
    """

    fragments: Sequence[UnrestrictedFragment] = attrs.field(default=list)

    def _get_sfo(self, sfo: str | SFO) -> SFO:
        return SFO.from_label(sfo) if isinstance(sfo, str) else sfo

    def _get_fragment(self, fragment: int):
        return self.fragments[fragment - 1]

    def get_sfo_overlap(self, sfo1: str | SFO, sfo2: str | SFO):
        sfo1, sfo2 = self._get_sfo(sfo1), self._get_sfo(sfo2)
        if sfo1.spin != sfo2.spin:
            return 0.0

        return self._get_fragment(0).get_overlap(
            kf_file=self.kf_file, uses_symmetry=self.calc_info.symmetry, irrep1=sfo1.irrep, index1=sfo1.index, irrep2=sfo2.irrep, index2=sfo2.index, spin=str(sfo1.spin)
        )

    def get_sfo_gross_population(self, fragment: int, sfo: str | SFO):
        sfo = self._get_sfo(sfo)
        return self._get_fragment(fragment).get_gross_population(irrep=sfo.irrep, index=sfo.index, spin=str(sfo.spin))

    def get_sfo_orbital_energy(self, fragment: int, sfo: str | SFO):
        sfo = self._get_sfo(sfo)
        return self._get_fragment(fragment).get_orbital_energy(irrep=sfo.irrep, index=sfo.index, spin=str(sfo.spin))

    def get_sfo_occupation(self, fragment: int, sfo: str | SFO):
        sfo = self._get_sfo(sfo)
        return self._get_fragment(fragment).get_occupation(irrep=sfo.irrep, index=sfo.index, spin=str(sfo.spin))

    def get_mo_orbitals(self, orb_range: tuple[int, int] = (-10, 10), irrep: str | None = None, spin: str | None = None) -> MOManager:
        mos = self.complex.get_mos(orb_range=orb_range, orb_irrep=irrep, spin=spin)
        return MOManager(complex_mos=mos)

    def get_sfo_orbitals(self, frag1_orb_range: tuple[int, int] = (10, 10), frag2_orb_range: tuple[int, int] = (10, 10), irrep: str | None = None, spin: str | None = None) -> SFOManager:
        frag_sfos = [frag.get_sfos(homo_lumo_range, irrep, str(spin)) for homo_lumo_range, frag in zip([frag1_orb_range, frag2_orb_range], self.fragments)]
        frag_sfos[1] = frag_sfos[1][::-1]  # reverse the orbitals to go from LUMO+x -> HOMO-x to HOMO-x -> LUMO+x

        # First, we want to store frag1 orbitals from LUMO+x to HOMO-x for easier printing later on
        # Second, LUMO - LUMO overlap has no phyiscal meaning so it is turned to 0.0
        overlap_matrix = np.zeros(shape=(len(frag_sfos[0]), len(frag_sfos[1])))
        try:
            for i, frag1_orb in enumerate(frag_sfos[0]):
                for j, frag2_orb in enumerate(frag_sfos[1]):
                    overlap_matrix[i, j] = self.get_sfo_overlap(frag1_orb, frag2_orb) if not (frag1_orb.occupation < 1e-6 and frag2_orb.occupation < 1e-6) else 0.0
        except KeyError:
            print("Detecting irrep error in getting the overlap matrix, skipping it as a result")

        return SFOManager(frag1_sfos=frag_sfos[0], frag2_sfos=frag_sfos[1], overlap_matrix=overlap_matrix)
