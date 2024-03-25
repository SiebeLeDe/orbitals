import pathlib as pl

import numpy as np
from orb_analysis import orb_config
from orb_analysis.analyzer.calc_analyzer import create_calc_analyser
from orb_analysis.custom_types import SpinTypes

np.set_printoptions(precision=5, suppress=True)
# https://www.scm.com/doc/ADF/Appendices/TAPE21.html


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
# rkf_folder = pl.Path(__file__).parent.parent / "test" / "fixtures" / "rkfs"
rkf_folder = pl.Path(r"C:\Users\siebb\VU_PhD\PhD\Scripting\local_packages\orbitals\test\fixtures\rkfs").resolve()
rkf_file = rkf_folder / f"{TestFiles.FILE2}.adf.rkf"

orb_range: tuple[int, int] = (5, 5)  # The range of orbitals to be analyzed (HOMO-X, LUMO+X with X the specified range)
irrep: str | None = None  # irrep such as "A1" (C3v) or "AA" (in Cs)
spin: str = SpinTypes.A  # "A" or "B" | does only matter for unrestricted calculations

orb_config.rkf_reading.orbital_energy_key = "escale"

# --------------------Main-------------------- #
# Creating the calc_analyzer object that performs the analysis of the (fragment) calculation
calc_analyzer = create_calc_analyser(rkf_file)  # This is the main object that will be used to analyze the calculation
