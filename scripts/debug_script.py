import pathlib as pl

import numpy as np
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser
from orb_analysis.custom_types import SpinTypes

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


# --------------------Input Arguments-------------------- #
rkf_folder = pl.Path(__file__).parent.parent / "test" / "fixtures" / "rkfs"
rkf_file = rkf_folder / "restricted_largecore_fragsym_c3v_full.adf.rkf"
# rkf_file = rkf_folder / "unrestricted_largecore_fragsym_c3v_full.adf.rkf"

orb_range: tuple[int, int] = (3, 3)  # The range of orbitals to be analyzed (HOMO-X, LUMO+X with X the specified range)
irrep: str | None = None  # irrep such as "A1" (C3v) or "AA" (in Cs)
spin: str = SpinTypes.A  # "A" or "B" | does only matter for unrestricted calculations


# --------------------Main-------------------- #
# Creating the calc_analyzer object that performs the analysis of the (fragment) calculation
calc_analyzer = create_calc_analyser(rkf_file)  # This is the main object that will be used to analyze the calculation

sfo_manager = calc_analyzer.get_sfo_orbitals(orb_range, orb_range)
# print(sfo_manager)
orb_pairs = sfo_manager.get_most_destabilizing_pauli_pairs(9, "HOMO_LUMO")  # type: ignore
[print(pair) for pair in orb_pairs]
