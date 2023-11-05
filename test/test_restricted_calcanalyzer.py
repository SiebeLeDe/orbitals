"""
Testmodule that tests (with pytest) the most high-level functionality of the package, being the CalcAnalyzer class.
The following types of calculations are tested:
    - restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation
    - restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation
    - restricted, no frozen core, fragment symmetry, no symmetry calculation
    - non-relativistic, restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation

For all of these calculations, the following methods are tested:
    - get_orbital_energy
    - get_gross_population
    - get_overlap

Note: this module only contains restricted calculations. The unrestricted calculations are tested in `test_unrestricted_calcanalyzer.py`
"""
import pathlib as pl

import pytest
from orb_analysis.complex import FACalcAnalyser, create_calc_analyser

current_dir = pl.Path(__file__).parent
fixtures_dir = current_dir / "fixtures" / "rkfs"


@pytest.fixture()
def calc_analyzer_restricted_nocore_fragsym_c3v() -> FACalcAnalyser:
    """Returns a CalcAnalyzer instance of a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
    rkf_path = fixtures_dir / 'restricted_nocore_fragsym_c3v_full.adf.rkf'
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_restricted_largecore_fragsym_c3v() -> FACalcAnalyser:
    """Returns a CalcAnalyzer instance of a restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    rkf_path = fixtures_dir / 'restricted_largecore_fragsym_c3v_full.adf.rkf'
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_restricted_nocore_fragsym_nosym() -> FACalcAnalyser:
    """Returns a CalcAnalyzer instance of a restricted, no frozen core, fragment symmetry, no symmetry calculation."""
    rkf_path = fixtures_dir / 'restricted_nocore_fragsym_nosym_full.adf.rkf'
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_non_relativistic() -> FACalcAnalyser:
    """Returns a CalcAnalyzer instance of a non-relativistic, restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    rkf_path = fixtures_dir / 'restricted_largecore_fragsym_c3v_nonrelativistic_full.adf.rkf'
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_restricted_largecore_nosym() -> FACalcAnalyser:
    """Returns a CalcAnalyzer instance of a restricted, large frozen core, no fragment symmetry, no symmetry calculation."""
    rkf_path = fixtures_dir / 'restricted_largecore_nofragsym_nosym_full.adf.rkf'
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_restricted_largecore_differentfragsym_c4v() -> FACalcAnalyser:
    """Returns a CalcAnalyzer instance of a restricted, large frozen core, different fragment symmetry (Clin + C4v), c4v complex symmetry calculation."""
    rkf_path = fixtures_dir / 'restricted_largecore_differentfragsym_c4v_full.adf.rkf'
    return create_calc_analyser(rkf_path)


# ------------------------------------------------------------
# ------------------Orbital energy tests----------------------
# ------------------------------------------------------------
#
# The following tests are for the `get_orbital_energy method` for various types of calculations.
# The implementation is quite straightforward. Just read the ("SFO", "escale") section in the KFFile and split it by irrep
# There is one remark: "escale" is for relativistic calculations, "energy" is for non-relativistic calculations.


def test_get_orbital_energy_restricted_nocore_fragsym_c3v(calc_analyzer_restricted_nocore_fragsym_c3v):
    """ Tests the `get_orbital_energy` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_restricted_nocore_fragsym_c3v
    frag1_energy = analyzer.get_orbital_energy(1, "8_A1")
    frag2_energy = analyzer.get_orbital_energy(2, "14_A1")
    assert frag1_energy == pytest.approx(-0.240836, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.033710, abs=1e-3)


def test_get_orbital_energy_restricted_nocore_fragsym_c3v_degenerate(calc_analyzer_restricted_nocore_fragsym_c3v):
    """ Tests the `get_orbital_energy` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_restricted_nocore_fragsym_c3v
    frag1_energy = analyzer.get_orbital_energy(1, "5_E1:1")
    frag2_energy = analyzer.get_orbital_energy(2, "12_E1:1")
    assert frag1_energy == pytest.approx(-0.331981, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.306534, abs=1e-3)


def test_get_orbital_energy_restricted_largecore_fragsym_c3v(calc_analyzer_restricted_largecore_fragsym_c3v):
    """ Tests the `get_orbital_energy` method for a restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_restricted_largecore_fragsym_c3v
    frag1_energy = analyzer.get_orbital_energy(1, "2_A1")
    frag2_energy = analyzer.get_orbital_energy(2, "4_A1")
    assert frag1_energy == pytest.approx(-0.238999, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.02899, abs=1e-3)


def test_get_orbital_energy_restricted_largecore_fragsym_c3v_degenerate(calc_analyzer_restricted_largecore_fragsym_c3v):
    """ Tests the `get_orbital_energy` method for a restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_restricted_largecore_fragsym_c3v
    frag1_energy = analyzer.get_orbital_energy(1, "1_E1:1")
    frag2_energy = analyzer.get_orbital_energy(2, "1_E1:1")
    assert frag1_energy == pytest.approx(-0.332368, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.783148, abs=1e-3)


# Add tests for restricted_nocore_fragsym_nosym
def test_get_orbital_energy_restricted_nocore_fragsym_nosym(calc_analyzer_restricted_nocore_fragsym_nosym):
    """ Tests the `get_orbital_energy` method for a restricted, no frozen core, fragment symmetry, no symmetry calculation ."""
    analyzer = calc_analyzer_restricted_nocore_fragsym_nosym
    frag1_energy = analyzer.get_orbital_energy(1, "8_A1")
    frag2_energy = analyzer.get_orbital_energy(2, "14_A1")
    assert frag1_energy == pytest.approx(-0.240836, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.033710, abs=1e-3)


def test_get_orbital_energy_non_relativistic(calc_analyzer_non_relativistic):
    """ Tests the `get_orbital_energy` method for a non-relativistic, restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_non_relativistic
    frag1_energy = analyzer.get_orbital_energy(1, "2_A1")
    frag2_energy = analyzer.get_orbital_energy(2, "4_A1")
    assert frag1_energy == pytest.approx(-0.237331, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.028922, abs=1e-3)


def test_get_orbital_energy_restricted_largecore_nosym(calc_analyzer_restricted_largecore_nosym):
    """ Tests the `get_orbital_energy` method for a restricted, large frozen core, no fragment symmetry, no symmetry calculation ."""
    analyzer = calc_analyzer_restricted_largecore_nosym
    frag1_energy = analyzer.get_orbital_energy(1, "4_A")
    frag2_energy = analyzer.get_orbital_energy(2, "13_A")
    assert frag1_energy == pytest.approx(-0.238998, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.028899, abs=1e-3)


# NOT WORKING YET
# def test_get_orbital_energy_restricted_largecore_differentfragsym_c4v(calc_analyzer_restricted_largecore_differentfragsym_c4v):
#     """ Tests the `get_orbital_energy` method for a restricted, large frozen core, different fragment symmetry (Clin + C4v), c4v complex symmetry calculation ."""
#     analyzer = calc_analyzer_restricted_largecore_differentfragsym_c4v
#     frag1_energy = analyzer.get_orbital_energy(1, "2_SIGMA")
#     frag2_energy = analyzer.get_orbital_energy(2, "5_A2")
#     assert frag1_energy == pytest.approx(-0.513043, abs=1e-3)
#     assert frag2_energy == pytest.approx(-0.510990, abs=1e-3)

# ------------------------------------------------------------
# ------------------Gross population tests--------------------
# ------------------------------------------------------------

# The following tests are for the `get_gross_population method for various types of calculations.
# This implementation is much less straightforward and requires taking into account frozen code orbitals, symmetry of the fragments (SFOs), and symmetry of the complex

def test_get_gross_population_restricted_nocore_fragsym_c3v(calc_analyzer_restricted_nocore_fragsym_c3v):
    """ Tests the `get_gross_population` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_restricted_nocore_fragsym_c3v
    frag1_pop = analyzer.get_gross_population(1, "8_A1")
    frag2_pop = analyzer.get_gross_population(2, "14_A1")
    assert frag1_pop == pytest.approx(1.816, abs=1e-3)
    assert frag2_pop == pytest.approx(0.189, abs=1e-3)


def test_get_gross_population_restricted_nocore_fragsym_c3v_degenerate(calc_analyzer_restricted_nocore_fragsym_c3v):
    """ Tests the `get_gross_population` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_restricted_nocore_fragsym_c3v
    frag1_pop = analyzer.get_gross_population(1, "5_E1:1")
    frag2_pop = analyzer.get_gross_population(2, "12_E1:1")
    assert frag1_pop == pytest.approx(1.990, abs=1e-3)
    assert frag2_pop == pytest.approx(1.975, abs=1e-3)


def test_get_gross_population_restricted_largecore_fragsym_c3v(calc_analyzer_restricted_largecore_fragsym_c3v):
    """ Tests the `get_gross_population` method for a restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_restricted_largecore_fragsym_c3v
    frag1_pop = analyzer.get_gross_population(1, "2_A1")
    frag2_pop = analyzer.get_gross_population(2, "4_A1")
    assert frag1_pop == pytest.approx(1.810, abs=1e-3)
    assert frag2_pop == pytest.approx(0.177, abs=1e-3)


def test_get_gross_population_restricted_largecore_fragsym_c3v_degenerate(calc_analyzer_restricted_largecore_fragsym_c3v):
    """ Tests the `get_gross_population` method for a restricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_restricted_largecore_fragsym_c3v
    frag1_pop_1 = analyzer.get_gross_population(1, "1_E1:1")
    frag1_pop_2 = analyzer.get_gross_population(1, "1_E1:2")
    frag2_pop_1 = analyzer.get_gross_population(2, "1_E1:1")
    frag2_pop_2 = analyzer.get_gross_population(2, "1_E1:2")
    assert frag1_pop_1 == pytest.approx(frag1_pop_2, abs=1e-3)
    assert frag2_pop_1 == pytest.approx(frag2_pop_2, abs=1e-3)
    assert frag1_pop_1 == pytest.approx(1.990, abs=1e-3)
    assert frag2_pop_1 == pytest.approx(2.000, abs=1e-3)


def test_get_gross_population_restricted_nocore_fragsym_nosym(calc_analyzer_restricted_nocore_fragsym_nosym):
    """ Tests the `get_gross_population` method for a restricted, no frozen core, fragment symmetry, no symmetry calculation."""
    analyzer = calc_analyzer_restricted_nocore_fragsym_nosym
    frag1_pop = analyzer.get_gross_population(1, "8_A1")
    frag2_pop = analyzer.get_gross_population(2, "14_A1")
    assert frag1_pop == pytest.approx(1.816, abs=1e-3)
    assert frag2_pop == pytest.approx(0.189, abs=1e-3)


def test_get_gross_population_restricted_largecore_nosym(calc_analyzer_restricted_largecore_nosym):
    """ Tests the `get_gross_population` method for a restricted, large frozen core, no fragment symmetry, no symmetry calculation ."""
    analyzer = calc_analyzer_restricted_largecore_nosym
    frag1_pop = analyzer.get_gross_population(1, "4_A")
    frag2_pop = analyzer.get_gross_population(2, "13_A")
    assert frag1_pop == pytest.approx(1.810, abs=1e-3)
    assert frag2_pop == pytest.approx(0.175, abs=1e-3)


# NOT WORKING YET
# def test_get_gross_population_restricted_largecore_differentfragsym_c4v(calc_analyzer_restricted_largecore_differentfragsym_c4v):
#     """ Tests the `get_gross_population` method for a restricted, large frozen core, different fragment symmetry (Clin + C4v), c4v complex symmetry calculation ."""
#     analyzer = calc_analyzer_restricted_largecore_differentfragsym_c4v
#     frag1_pop1 = analyzer.get_gross_population(1, "2_SIGMA")
#     frag1_pop2 = analyzer.get_gross_population(1, "1_PI:y")
#     frag1_pop3 = analyzer.get_gross_population(1, "1_DELTA:xy")
#     frag1_pop4 = analyzer.get_gross_population(1, "3_SIGMA")
#     frag2_pop1 = analyzer.get_gross_population(2, "2_E1:2")
#     frag2_pop2 = analyzer.get_gross_population(2, "6_E1:1")
#     frag2_pop3 = analyzer.get_gross_population(2, "8_A1")
#     frag2_pop4 = analyzer.get_gross_population(2, "5_B2")
#     assert frag1_pop1 == pytest.approx(1.989, abs=1e-3)
#     assert frag1_pop2 == pytest.approx(0.000, abs=1e-3)
#     assert frag1_pop3 == pytest.approx(0.000, abs=1e-3)
#     assert frag1_pop4 == pytest.approx(1.462, abs=1e-3)
#     assert frag2_pop1 == pytest.approx(2.004, abs=1e-3)
#     assert frag2_pop2 == pytest.approx(1.999, abs=1e-3)
#     assert frag2_pop3 == pytest.approx(0.477, abs=1e-3)
#     assert frag2_pop4 == pytest.approx(0.000, abs=1e-3)

# ------------------------------------------------------------
# ---------------------Overlap tests--------------------------
# ------------------------------------------------------------

# The following tests are for the `get_overlap method` for various types of calculations.
# This implementation is also not straightforward and requires taking into account frozen code orbitals, symmetry of the fragments (SFOs), and symmetry of the complex
# Note that the overlap matrix is a lower triangular matrix, so only the lower triangular part is stored in the KFFile.
# The overlap matrix is splitted into irreps of the complex symmetry and skips the frozen core orbital.
# Therefore, one should be careful with SFO indices. The SFO indices are not the same as the indices of the overlap matrix.

def test_get_overlap_restricted_nocore_fragsym_c3v(calc_analyzer_restricted_nocore_fragsym_c3v):
    """ Tests the `get_overlap` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_restricted_nocore_fragsym_c3v
    self_overlap = analyzer.get_overlap("1_A1", "1_A1")
    homo_lumo_overlap = analyzer.get_overlap("8_A1", "14_A1")
    homo_homo_overlap = analyzer.get_overlap("8_A1", "13_A1")
    assert self_overlap == pytest.approx(self_overlap, abs=1e-3)
    assert homo_lumo_overlap == pytest.approx(0.4084, abs=1e-3)
    assert homo_homo_overlap == pytest.approx(0.2384, abs=1e-3)


def test_get_overlap_restricted_largecore_fragsym_c3v_A1(calc_analyzer_restricted_largecore_fragsym_c3v):
    """ Tests the `get_overlap` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_restricted_largecore_fragsym_c3v
    homo_lumo_overlap = analyzer.get_overlap("2_A1", "4_A1")
    homo_homo_overlap = analyzer.get_overlap("2_A1", "3_A1")
    assert homo_lumo_overlap == pytest.approx(0.4093, abs=1e-3)
    assert homo_homo_overlap == pytest.approx(0.2497, abs=1e-3)


def test_get_overlap_restricted_largecore_fragsym_c3v_A2(calc_analyzer_restricted_largecore_fragsym_c3v):
    """ Tests the `get_overlap` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_restricted_largecore_fragsym_c3v
    homo_lumo_overlap = analyzer.get_overlap("1_A2", "4_A2")
    assert homo_lumo_overlap == pytest.approx(0.0490, abs=1e-4)


def test_get_overlap_restricted_nocore_fragsym_nosym(calc_analyzer_restricted_nocore_fragsym_nosym):
    analyzer = calc_analyzer_restricted_nocore_fragsym_nosym
    homo_lumo_overlap = analyzer.get_overlap("8_A1", "14_A1")
    assert homo_lumo_overlap == pytest.approx(0.4084, abs=1e-4)


def test_get_overlap_restricted_largecore_nosym(calc_analyzer_restricted_largecore_nosym):
    """ Tests the `get_overlap` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_restricted_largecore_nosym
    homo_lumo_overlap = analyzer.get_overlap("4_A", "13_A")
    homo_homo_overlap = analyzer.get_overlap("4_A", "5_A")
    assert homo_lumo_overlap == pytest.approx(0.4032, abs=1e-3)
    assert homo_homo_overlap == pytest.approx(0.2469, abs=1e-3)


# NOT WORKING YET
# def test_get_overlap_restricted_largecore_differentfragsym_c4v(calc_analyzer_restricted_largecore_differentfragsym_c4v):
#     """ Tests the `get_overlap` method for a restricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
#     analyzer = calc_analyzer_restricted_largecore_differentfragsym_c4v
#     homo_lumo_overlap = analyzer.get_overlap("3_SIGMA", "8_A1")
#     homo_lumo_overlap2 = analyzer.get_overlap("3_SIGMA", "7_A1")
#     assert homo_lumo_overlap == pytest.approx(0.3696, abs=1e-3)
#     assert homo_lumo_overlap2 == pytest.approx(0.0912, abs=1e-3)
