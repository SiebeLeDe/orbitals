import os
import pathlib as pl

import numpy as np
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser
from orb_analysis.custom_types import SpinTypes

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


# --------------------Input Arguments-------------------- #
path_to_rkf_file = pl.Path(
    "/Users/siebeld/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/PhD/Scripting/local_packages/orbitals/test/fixtures/rkfs/restricted_largecore_fragsym_c3v_full.adf.rkf"
)  # Path to the rkf file to be analyzed
orb_range: tuple[int, int] = (6, 6)  # The range of orbitals to be analyzed (HOMO-X, LUMO+X with X the specified range)
irrep: str | None = None  # irrep such as "A1" (C3v) or "AA" (in Cs)
spin: str = SpinTypes.A  # "A" or "B" | does only matter for unrestricted calculations


# --------------------Main-------------------- #
# Creating the calc_analyzer object that performs the analysis of the (fragment) calculation
calc_analyzer = create_calc_analyser(path_to_rkf_file)  # This is the main object that will be used to analyze the calculation

# Printing the results
orb_summary = calc_analyzer(orb_range=orb_range, irrep=irrep, spin=spin)
print(orb_summary)

# And/or writing the summary to a file
output_file = pl.Path(os.getcwd()) / "analysis_example_output.txt"
output_file.write_text(orb_summary)
print(f"Output file written to {output_file}")
