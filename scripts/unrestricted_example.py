import pathlib as pl
from scm.plams import KFFile
import numpy as np

from orb_analysis.analyzer.calc_analyzer import create_calc_analyser

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


# ------------------Available test files------------------ #

class Unrestricted_TestFiles:
    FILE1 = "unrestricted_largecore_fragsym_c3v_full"
    FILE2 = "unrestricted_largecore_fragsym_c3v_unrelativistic_full"
    FILE3 = "unrestricted_largecore_fragsym_nosym_full"  # Does not work due to spin error
    FILE4 = "unrestricted_largecore_nofragsym_nosym_full"  # Does not work due to spin error
    FILE5 = "unrestricted_nocore_fragsym_c3v_full"
    FILE6 = "unrestricted_nocore_fragsym_nosym_full"  # Does not work due to spin error
    FILE7 = "unrestricted_nocore_nofragsym_c3v_full"
    FILE8 = "unrestricted_nocore_nofragsym_nosym_full"


# --------------------Input Arguments-------------------- #
current_path = pl.Path(__file__).parent
path_to_folder_with_rkf_files = (current_path.parent / "test" / "fixtures" / "rkfs")
# See the test/fixtures/rkfs folder for more examples
rkf_file = Unrestricted_TestFiles.FILE4
path_to_rkf_file = path_to_folder_with_rkf_files / f"{rkf_file}.adf.rkf"

# --------------------Main-------------------- #
calc_analyzer = create_calc_analyser(path_to_rkf_file)

print("MAIN FUNCTION")
kf_file = KFFile(path_to_rkf_file)
summary = calc_analyzer(orb_range=(6, 6), spin="B")
print(summary)
