import pathlib as pl
import os
import numpy as np
import orb_analysis
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


# ------------------Available test files------------------ #
class TestFiles:
    # Download the test files from https://github.com/SiebeLeDe/orbitals/tree/main/test/fixtures/rkfs
    # FILE1 does not work and don't know how to fix it. Gross populations fail completely due to unclear documentation of frag frozen cores per irrep in rkf files
    FILE1 = "restricted_largecore_differentfragsym_c4v_full"
    FILE2 = "restricted_largecore_fragsym_c3v_full"
    FILE3 = "restricted_largecore_fragsym_c3v_nonrelativistic_full"
    FILE4 = "restricted_largecore_nofragsym_nosym_full"
    FILE5 = "restricted_nocore_fragsym_c3v_full"
    FILE6 = "restricted_nocore_fragsym_nosym_full"
    FILE7 = "unrestricted_largecore_fragsym_c3v_full"
    FILE8 = "unrestricted_largecore_fragsym_c3v_unrelativistic_full"
    FILE9 = "unrestricted_largecore_fragsym_nosym_full"
    FILE10 = "unrestricted_largecore_nofragsym_nosym_full"
    FILE11 = "unrestricted_nocore_fragsym_c3v_full"
    FILE12 = "unrestricted_nocore_fragsym_nosym_full"
    FILE13 = "unrestricted_nocore_nofragsym_c3v_full"
    FILE14 = "unrestricted_nocore_nofragsym_nosym_full"


# --------------------Input Arguments-------------------- #
module_path = pl.Path(orb_analysis.__file__).parent.parent.parent
path_to_folder_with_rkf_files = module_path / "test" / "fixtures" / "rkfs"

# See the test/fixtures/rkfs folder for more examples (https://github.com/SiebeLeDe/orbitals/tree/main/test/fixtures/rkfs)
rkf_file = TestFiles.FILE2
path_to_rkf_file = path_to_folder_with_rkf_files / f"{rkf_file}.adf.rkf"
path_to_rkf_file = pl.Path("/Users/siebeld/Desktop/fa.sh_full.adf.rkf")

# --------------------Main-------------------- #
calc_analyzer = create_calc_analyser(path_to_rkf_file)  # This is the main object that will be used to analyze the calculation

# Another option is to get the SFOs. It returns an :OrbitalManager: object which contains the SFOs and associated overlap matrix. Printing this will result in a formatted table (string)
orbs = calc_analyzer.get_sfo_orbitals(frag1_orb_range=(10, 10), frag2_orb_range=(10, 10))
print("\nGETTING THE SFOS (symmetrized fragment orbitals) DIRECTLY (get_sfo_orbitals method):")
print(orbs)

# It is also possible to get the SFO overlap matrix. This returns a formatted table (string) which can be printed or written to a file
sfo_overlap = orbs.get_overlap_matrix_table()
print("\nSFO Overlap Matrix (get_overlap_matrix_table method):)")
print(sfo_overlap)

# Another option is to get the MOs. It returns an :OrbitalManager: object (MOManager) which contains the MOs. Printing this will result in a formatted table (string)
orbs = calc_analyzer.get_mo_orbitals(orb_range=(10, 10))
print("\nGETTING THE MOs (molecular orbitals) DIRECTLY (get_mo_orbitals method):")
print(orbs)

# Best way is to call the calc_analyzer object directly. Calling gets the SFOs, SFO overlap matrix, and MOs and returns a formatted table (string)
print("\nCALLING THE CALC_ANALYZER OBJECT DIRECTLY:")
summary = (calc_analyzer(orb_range=(10, 10)))
print(summary)

print("\nWRITING THE OUTPUT TO A FILE IN THE CURRENT DIRECTORY:")
output_file = pl.Path(os.getcwd()) / "analysis_example_output.txt"
output_file.write_text(summary)
print(f"Output file written to {output_file}")
