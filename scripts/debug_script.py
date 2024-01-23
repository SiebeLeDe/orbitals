import pathlib as pl

import numpy as np
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser
from orb_analysis.custom_types import SpinTypes

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


# --------------------Input Arguments-------------------- #
rkf_folder = pl.Path(__file__).parent.parent / "test" / "fixtures" / "rkfs"
rkf_file = rkf_folder / "unrestricted_largecore_fragsym_nosym_full.adf.rkf"
# rkf_file = rkf_folder / "unrestricted_largecore_fragsym_c3v_full.adf.rkf"

orb_range: tuple[int, int] = (6, 6)  # The range of orbitals to be analyzed (HOMO-X, LUMO+X with X the specified range)
irrep: str | None = None  # irrep such as "A1" (C3v) or "AA" (in Cs)
spin: str = SpinTypes.A  # "A" or "B" | does only matter for unrestricted calculations


# --------------------Main-------------------- #
# Creating the calc_analyzer object that performs the analysis of the (fragment) calculation
calc_analyzer = create_calc_analyser(rkf_file)  # This is the main object that will be used to analyze the calculation

# Printing the results
fragment1, fragment2 = calc_analyzer.fragments

# Print A and B values next to each other for each irrep
pop_A, pop_B = fragment1.fragment_data.gross_populations.values()
print(pop_A)

frag1_pop_2A = fragment1.get_gross_population(irrep="A1", index=4, spin="A")
frag1_pop_2B = fragment1.get_gross_population(irrep="A1", index=4, spin="B")
# print(frag1_pop_2A)
# print(frag1_pop_2B)

# for (irrep, pops), (irrepB, popsB) in zip(pop_A.items(), pop_B.items()):
#     print(irrep)
#     for pop, popB in zip(pops, popsB):
#         print(f"{pop:6.3f}, {popB:6.3f}")
