import pathlib as pl

import numpy as np
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser
from orb_analysis.orbital.orbital_pair import OrbitalPair

np.set_printoptions(precision=5, suppress=True)

# https://www.scm.com/doc/ADF/Appendices/TAPE21.html

###########################################################################
# Functions ###############################################################
###########################################################################


def get_fa_rkf_file(calc_dir: pl.Path) -> pl.Path:
    """Searches through the calculation directory for a adf.rkf and returns the path"""
    fa_dir = calc_dir
    for file in fa_dir.iterdir():
        if file.name.endswith("adf.rkf"):
            return file

    return calc_dir


def make_interaction_table(type: str, system_names: list[str], interaction_list: list[list[OrbitalPair]]) -> str:
    """Returns a string containing a formatted table of the interactions"""
    ret_str = ""
    for name, interaction_pairs in zip(system_names, interaction_list):
        ret_str += f"\n{name}"
        ret_str += OrbitalPair.format_orbital_pairs_for_printing(interaction_pairs, top_header=type)
        ret_str += "\n"
    return ret_str


base_dir = pl.Path(r"C:\Users\siebb\VU_PhD\PhD\Projects\Squaramides\calcs\dimer_with_urea").resolve()
# base_dir = pl.Path("/Users/siebeld/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/PhD/Projects/Squaramides/calcs/dimer_with_urea").resolve()
calc_dir = base_dir / "fa_consistent"
output_dir = base_dir / "fa_orb_results" / "consistent_geo_fa"
orb_range: tuple[int, int] = (7, 5)  # The range of orbitals to be analyzed (HOMO-X, LUMO+X with X the specified range)
irrep: str | None = "AA"  # irrep such as "A1" (C3v) or "AA" (in Cs)
type_calc: int = 7  # 6 for urea as Hbond acceptor, 7 for urea as Hbond donor
interpol_point: float = 2.8


systems = [
    "urea_di_o_cs_X",
    "urea_di_s_cs_X",
    "urea_di_se_cs_X",
    "deltamide_di_o_cs_X",
    "deltamide_di_s_cs_X",
    "deltamide_di_se_cs_X",
    "squaramide_di_o_cs_X",
    "squaramide_di_s_cs_X",
    "squaramide_di_se_cs_X",
]

# replace "X" by the type_calc
systems = [calc.replace("X", str(type_calc)) for calc in systems]

###########################################################################
# Main ####################################################################
###########################################################################
# We want to perform MO and SFO analyses on the pyfrag calculations that lie closest to the point that we want to study.
# This point is a float number and is along the trajectory such as "bondlength_1".


# 2. Load the specific pyfrag calculation that is the closest to the specified point.
rkf_files = [calc_dir / f"{system}.adf.rkf" for system in systems]
calc_analyzers = [create_calc_analyser(rkf_file, name=calc) for calc, rkf_file in zip(systems, rkf_files)]
sfo_managers = [calc_analyzer.get_sfo_orbitals(orb_range, orb_range, irrep) for calc_analyzer in calc_analyzers]

output_dir.mkdir(exist_ok=True)
for system_name, sfo_manager in zip(systems, sfo_managers):
    output_file = output_dir / f"{system_name}.txt"
    output_file.write_text(f"{sfo_manager}")

    print(f"Output file written to {output_file.parent}/{output_file.name}")


important_pauli_interactions = [sfo_manager.get_most_destabilizing_pauli_pairs(4) for sfo_manager in sfo_managers]
important_oi_orbital_pairs = [sfo_manager.get_most_stabilizing_oi_pairs(4) for sfo_manager in sfo_managers]

pauli_outfile = output_dir / "Pauli.txt"
oi_outfile = output_dir / "OI.txt"
pauli_outfile.write_text(make_interaction_table("Pauli", systems, important_pauli_interactions))
print(f"Pauli file written to {pauli_outfile.parent.name}")
# oi_outfile.write_text(make_interaction_table("OI", systems, important_oi_orbital_pairs))
# print(f"OI file written to {oi_outfile.parent.name}")

# # Copy the rkf_file for transfering to local computer
# copy_folder = output_dir / "rkf_files"
# copy_folder.mkdir(exist_ok=True)
# for system_name, rkf_file in zip(systems, rkf_files):
#     sh.copy(rkf_file, copy_folder / f"{system_name}.adf.rkf")
#     print(f"rkf file of {rkf_file.name} written to {copy_folder}")
