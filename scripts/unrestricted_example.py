import pathlib as pl
import matplotlib.pyplot as plt

import numpy as np

from orb_analysis.analyzer.calc_analyzer import create_calc_analyser

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


# ------------------Available test files------------------ #
class Unrestricted_TestFiles:
    FILE1 = "unrestricted_largecore_differentfragsym_c4v_full"
    FILE2 = "unrestricted_largecore_fragsym_c3v_full"
    FILE3 = "unrestricted_largecore_fragsym_c3v_nonrelativistic_full"
    FILE4 = "unrestricted_largecore_nofragsym_nosym_full"
    FILE5 = "unrestricted_nocore_fragsym_c3v_full"
    FILE6 = "unrestricted_nocore_fragsym_nosym_full"


# --------------------Input Arguments-------------------- #
current_path = pl.Path(__file__).parent
path_to_folder_with_rkf_files = (current_path.parent / "test" / "fixtures" / "rkfs")
# See the test/fixtures/rkfs folder for more examples
rkf_file = Unrestricted_TestFiles.FILE2
path_to_rkf_file = path_to_folder_with_rkf_files / f"{rkf_file}.adf.rkf"

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
    "4_A1_A",  # SOMO
    "4_A1_B",  # LUMO
]
frag2_labels = [
    "4_A1_A",  # LUMO
    "4_A1_B",  # SOMO
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
frag_counter = 1
for frag, frag_label in zip(calc_analyzer.fragments, [frag1_labels, frag2_labels]):
    name = frag.name
    for sfo in frag_label:
        orb_energy = calc_analyzer.get_sfo_orbital_energy(fragment=frag_counter, sfo=sfo)
        gross_pop = calc_analyzer.get_sfo_gross_population(fragment=frag_counter, sfo=sfo)
        print(f"Fragment {frag_counter}, SFO {sfo :6s}: {orb_energy :^+.4f} Ha, {gross_pop :<.5f} electrons")
    frag_counter += 1

overlap = np.array([
    [calc_analyzer.get_sfo_overlap(sfo1=label1, sfo2=label2) for label2 in frag2_labels]
    for label1 in frag1_labels
])

plt.imshow(overlap, cmap="coolwarm", interpolation="nearest", alpha=0.5)
plt.colorbar()
plt.xticks(np.arange(len(frag1_labels)), frag1_labels)
plt.yticks(np.arange(len(frag2_labels)), frag2_labels)
plt.show()
