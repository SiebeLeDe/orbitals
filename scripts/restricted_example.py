import pathlib as pl

import numpy as np
from scm.plams import KFFile
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser
from orb_analysis.orb_functions.sfo_functions import get_gross_populations

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


# ------------------Available test files------------------ #
class Restricted_TestFiles:
    FILE1 = "restricted_largecore_differentfragsym_c4v_full"
    FILE2 = "restricted_largecore_fragsym_c3v_full"
    FILE3 = "restricted_largecore_fragsym_c3v_nonrelativistic_full"
    FILE4 = "restricted_largecore_nofragsym_nosym_full"
    FILE5 = "restricted_nocore_fragsym_c3v_full"
    FILE6 = "restricted_nocore_fragsym_nosym_full"


# --------------------Input Arguments-------------------- #
current_path = pl.Path(__file__).parent
path_to_folder_with_rkf_files = (current_path.parent / "test" / "fixtures" / "rkfs")
# See the test/fixtures/rkfs folder for more examples
rkf_file = Restricted_TestFiles.FILE6
path_to_rkf_file = path_to_folder_with_rkf_files / f"{rkf_file}.adf.rkf"
# path_to_rkf_file = "/Users/siebeld/Desktop/fa.sh_full.adf.rkf"
# --------------------Main-------------------- #
calc_analyzer = create_calc_analyser(path_to_rkf_file)

# # NoSym calculation with fragment symmetry
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

# # Cs calculation with fragment symmetry
# frag1_labels = [
#     "33_AA",
#     "29_AA",
# ]

# frag2_labels = [
#     "33_AA",
#     "30_AA",
# ]

# c3v calculation with fragment symmetry
frag1_labels = [
    "7_A1",  # HOMO-1
    "8_A1",  # HOMO
    "9_A1",  # LUMO
]
frag2_labels = [
    "13_A1",  # HOMO
    "14_A1",  # LUMO
    "15_A1",  # LUMO-1
]

# NoSym calculation without fragment symmetry
# frag1_labels = [
#     "1_A",
# ]

# frag2_labels = [
#     "1_A",
# ]

print("MAIN FUNCTION")
# [pprint(frag.fragment_data.n_frozen_cores_per_irrep) for frag in calc_analyzer.fragments]
# [pprint(frag.fragment_data.gross_populations) for frag in calc_analyzer.fragments]
# [pprint(frag.fragment_data.orb_energies) for frag in calc_analyzer.fragments]

# Print the label, energy, and population of each SFO in each fragment
# frag_counter = 1
# for frag, frag_label in zip(calc_analyzer.fragments, [frag1_labels, frag2_labels]):
#     name = frag.name
#     for sfo in frag_label:
#         orb_energy = calc_analyzer.get_sfo_orbital_energy(fragment=frag_counter, sfo=sfo)
#         gross_pop = calc_analyzer.get_sfo_gross_population(fragment=frag_counter, sfo=sfo)
#         print(f"Fragment {frag_counter}, SFO {sfo :6s}: {orb_energy :^+.4f} Ha, {gross_pop :<.5f} electrons")
#     frag_counter += 1

orbs = calc_analyzer.get_sfo_orbitals(frag1_orb_range=(2, 2), frag2_orb_range=(4, 4))
kf_file = KFFile(path_to_rkf_file)
gross_pop = get_gross_populations(kf_file, frag_index=1)
gross_pop2 = get_gross_populations(kf_file, frag_index=2)
print(gross_pop)

print(calc_analyzer(orb_range=(4, 2)))

# overlap = np.array([
#     [calc_analyzer.get_sfo_overlap(sfo1=label1, sfo2=label2) for label2 in frag2_labels]
#     for label1 in frag1_labels
# ])

# plt.imshow(overlap, cmap="coolwarm", interpolation="nearest", alpha=0.5)
# plt.colorbar()
# plt.xticks(np.arange(len(frag1_labels)), frag1_labels)
# plt.yticks(np.arange(len(frag2_labels)), frag2_labels)
# plt.show()
