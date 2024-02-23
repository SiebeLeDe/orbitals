import pathlib as pl

import numpy as np
import pandas as pd
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser
from orb_analysis.custom_types import SpinTypes
from orb_analysis.orbital.orbital_pair import OrbitalPair
from tabulate import tabulate

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


def format_orbital_pair_for_printing(orb_pairs: list[OrbitalPair], top_header: str = "SFO Interaction Pairs") -> str:
    """
    Returns a string representing each orbital pair in the list in a nicely formatted table
    Example:
        [orb1_label] [orb1_energy] [orb1_grosspop] [orb2_label] [orb2_energy] [orb2_grosspop] [overlap] [stabilization]
    """
    headers = ["SFO1", "energy (eV)", "gross pop (a.u.)", "SFO2", "energy (eV)", "gross pop (a.u.)", "Overlap", "S^2/epsilon * 100"]

    # Create a DataFrame
    df = pd.DataFrame([orb_pair.as_numpy_array for orb_pair in orb_pairs], columns=headers)

    # Use tabulate to create a formatted string
    result = tabulate(df, headers="keys", tablefmt="simple", showindex=False, floatfmt=".3f")  # type: ignore # df is accepted as argument

    return f"\n{top_header}\n{result}"


systems: list[str] = [f"{species}_di_{chalc}_cs_7.adf.rkf" for species in ["urea", "deltamide", "squaramide"] for chalc in ["o", "s", "se"]]

# --------------------Input Arguments-------------------- #
# rkf_folder = pl.Path(__file__).parent.parent / "test" / "fixtures" / "rkfs"
rkf_folder = pl.Path(r"C:\Users\siebb\VU_PhD\PhD\Projects\Squaramides\calcs\dimer_with_urea\fa_orb_results\consistent_geo_fa\rkf_files").resolve()
rkf_file = rkf_folder / systems[0]
# rkf_file = rkf_folder / "unrestricted_largecore_fragsym_c3v_full.adf.rkf"

orb_range: tuple[int, int] = (5, 5)  # The range of orbitals to be analyzed (HOMO-X, LUMO+X with X the specified range)
irrep: str | None = None  # irrep such as "A1" (C3v) or "AA" (in Cs)
spin: str = SpinTypes.A  # "A" or "B" | does only matter for unrestricted calculations


# --------------------Main-------------------- #
# Creating the calc_analyzer object that performs the analysis of the (fragment) calculation
calc_analyzer = create_calc_analyser(rkf_file)  # This is the main object that will be used to analyze the calculation

sfo_manager = calc_analyzer.get_sfo_orbitals(orb_range, orb_range)
# print(sfo_manager)
orb_pairs = sfo_manager.get_most_destabilizing_pauli_pairs(5)
# [print(pair) for pair in orb_pairs]
oi_pairs = sfo_manager.get_most_stabilizing_oi_pairs(5)
# [print(pair.orb1, pair.orb2, pair.overlap) for pair in oi_pairs]
formatted_pairs = format_orbital_pair_for_printing(oi_pairs, "SFO OI Pairs")
print(formatted_pairs)
formatted_pairs = format_orbital_pair_for_printing(orb_pairs, "SFO Pauli Pairs")
print(formatted_pairs)
