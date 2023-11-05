# from pprint import pprint
import pathlib as pl

import numpy as np

from orb_analysis.complex import create_calc_analyser

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html

# --------------------Input Arguments-------------------- #
current_path = pl.Path(__file__)
path_to_folder_with_rkf_files = (
    current_path.parent.parent.parent / "test" / "fixtures" / "rkfs"
)
rkf_file = "restricted_nocore_fragsym_nosym_full"
path_to_rkf_file = path_to_folder_with_rkf_files / f"{rkf_file}.adf.rkf"

# --------------------Main-------------------- #
calc_analyzer = create_calc_analyser(path_to_rkf_file)

# # NoSym calculation
# frag1_labels = [
#     "4_AA",
#     "5_AA",
#     "1_AAA",
#     "2_AAA",
# ]

# frag2_labels = [
#     "4_AA",
#     "5_AA",
#     "1_AAA",
#     "2_AAA",
# ]

# C3v calculation
# frag1_labels = [
#     "33_AA",
#     "29_AA",
# ]

# frag2_labels = [
#     "33_AA",
#     "30_AA",
# ]

# NoSym calculation
frag1_labels = [
    "1_A",
]

frag2_labels = [
    "1_A",
]

# print("MAIN FUNCTION")
# [pprint(frag.fragment_data.n_frozen_cores_per_irrep) for frag in calc_analyzer.fragments]
# [pprint(frag.fragment_data.gross_populations) for frag in calc_analyzer.fragments]
# [pprint(frag.fragment_data.orb_energies) for frag in calc_analyzer.fragments]

# # Print the label, energy, and population of each SFO in each fragment
# frag_counter = 1
# for frag, frag_label in zip(calc_analyzer.fragments, [frag1_labels, frag2_labels]):
#     name = frag.name
#     for sfo in frag_label:
#         orb_energy = calc_analyzer.get_orbital_energy(fragment=frag_counter, sfo=sfo)
#         gross_pop = calc_analyzer.get_gross_population(fragment=frag_counter, sfo=sfo)
#         print(f"Fragment {frag_counter}, SFO {sfo :6s}: {orb_energy :^+.4f} Ha, {gross_pop :<.5f} electrons")
#     frag_counter += 1

overlap = np.array([
    [calc_analyzer.get_overlap(sfo1=label1, sfo2=label2) for label2 in frag2_labels]
    for label1 in frag1_labels
])
# print(overlap.flatten())
