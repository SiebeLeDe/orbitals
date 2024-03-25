"""
Testmodule that tests (with pytest) the most high-level functionality of the package, being the CalcAnalyzer class.
The following types of calculations are tested:
    - unrestricted, no frozen core, fragment symmetry, c3v complex symmetry calculation
    - unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation
    - unrestricted, no frozen core, fragment symmetry, no symmetry calculation
    - non-relativistic, unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation

For all of these calculations, the following methods are tested:
    - get_sfo_orbital_energy
    - get_sfo_gross_population
    - get_sfo_overlap

The tests are based on the calculations with the molecule Me3Ge-H + H-CCl3 (heavier derivative of Me3C-H + H-CCl3) in C3v symmetry.
Note: this module only contains unrestricted calculations.
"""

import pathlib as pl

import pytest
from orb_analysis import orb_config
from orb_analysis.analyzer.calc_analyzer import CalcAnalyzer, create_calc_analyser

current_dir = pl.Path(__file__).parent
fixtures_dir = current_dir / "fixtures" / "rkfs"

# The orbital energy key and unit need to be fixed because the tests have been written for the conditions below
orb_config.rkf_reading.orbital_energy_key = "escale"
orb_config.rkf_reading.orbital_energy_unit = "hartree"


@pytest.fixture()
def calc_analyzer_unrestricted_nocore_fragsym_c3v() -> CalcAnalyzer:
    """Returns a CalcAnalyzer instance of a unrestricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
    rkf_path = fixtures_dir / "unrestricted_nocore_fragsym_c3v_full.adf.rkf"
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_unrestricted_largecore_fragsym_c3v() -> CalcAnalyzer:
    """Returns a CalcAnalyzer instance of a unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    rkf_path = fixtures_dir / "unrestricted_largecore_fragsym_c3v_full.adf.rkf"
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_unrestricted_nocore_fragsym_nosym() -> CalcAnalyzer:
    """Returns a CalcAnalyzer instance of a unrestricted, no frozen core, fragment symmetry, no symmetry calculation."""
    rkf_path = fixtures_dir / "unrestricted_nocore_fragsym_nosym_full.adf.rkf"
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_unrestricted_largecore_fragsym_nosym() -> CalcAnalyzer:
    """Returns a CalcAnalyzer instance of a unrestricted, no frozen core, fragment symmetry, no symmetry calculation."""
    rkf_path = fixtures_dir / "unrestricted_largecore_fragsym_nosym_full.adf.rkf"
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_non_relativistic() -> CalcAnalyzer:
    """Returns a CalcAnalyzer instance of a non-relativistic, unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    rkf_path = fixtures_dir / "unrestricted_largecore_fragsym_c3v_nonrelativistic_full.adf.rkf"
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_unrestricted_largecore_nosym() -> CalcAnalyzer:
    """Returns a CalcAnalyzer instance of a unrestricted, large frozen core, no fragment symmetry, no symmetry calculation."""
    rkf_path = fixtures_dir / "unrestricted_largecore_nofragsym_nosym_full.adf.rkf"
    return create_calc_analyser(rkf_path)


@pytest.fixture()
def calc_analyzer_unrestricted_largecore_differentfragsym_c4v() -> CalcAnalyzer:
    """Returns a CalcAnalyzer instance of a unrestricted, large frozen core, different fragment symmetry (Clin + C4v), c4v complex symmetry calculation."""
    rkf_path = fixtures_dir / "unrestricted_largecore_differentfragsym_c4v_full.adf.rkf"
    return create_calc_analyser(rkf_path)


# ------------------------------------------------------------
# ------------------Orbital energy tests----------------------
# ------------------------------------------------------------
#
# The following tests are for the `get_sfo_orbital_energy method` for various types of calculations.
# The implementation is quite straightforward. Just read the ("SFO", "escale") section in the KFFile for spin A and ("SFO", "escale_B") for spin B and split it by irrep
# There is one remark: "escale" is for relativistic calculations, "energy" is for non-relativistic calculations.
# Energy unit in hatree.


def test_get_sfo_orbital_energy_unrestricted_nocore_fragsym_c3v(calc_analyzer_unrestricted_nocore_fragsym_c3v):
    """Tests the `get_sfo_orbital_energy` method for a unrestricted, no frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_nocore_fragsym_c3v
    frag1_energy = analyzer.get_sfo_orbital_energy(1, "11_A1_A")
    frag2_energy = analyzer.get_sfo_orbital_energy(2, "6_A1_A")
    frag1_energy_2A = analyzer.get_sfo_orbital_energy(1, "2_A2_A")
    frag1_energy_2B = analyzer.get_sfo_orbital_energy(1, "2_A2_B")
    frag2_energy_2A = analyzer.get_sfo_orbital_energy(2, "1_A2_A")
    frag2_energy_2B = analyzer.get_sfo_orbital_energy(2, "1_A2_B")
    assert frag1_energy == pytest.approx(-0.140127, abs=1e-3)
    assert frag2_energy == pytest.approx(-0.185403, abs=1e-3)
    assert frag1_energy_2A == pytest.approx(0.097642, abs=1e-3)
    assert frag1_energy_2B == pytest.approx(0.09747, abs=1e-3)
    assert frag2_energy_2A == pytest.approx(-0.369865, abs=1e-3)
    assert frag2_energy_2B == pytest.approx(-0.372843, abs=1e-3)


def test_get_sfo_orbital_energy_unrestricted_nocore_fragsym_c3v_degenerate(calc_analyzer_unrestricted_nocore_fragsym_c3v):
    """Tests the `get_sfo_orbital_energy` method for a unrestricted, no frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_nocore_fragsym_c3v
    frag1_energy_A = analyzer.get_sfo_orbital_energy(1, "8_E1:1_A")
    frag1_energy_B = analyzer.get_sfo_orbital_energy(1, "8_E1:1_B")
    frag2_energy_A = analyzer.get_sfo_orbital_energy(2, "5_E1:1_A")
    frag2_energy_B = analyzer.get_sfo_orbital_energy(2, "5_E1:1_B")
    assert frag1_energy_A == pytest.approx(-0.347617, abs=1e-3)
    assert frag1_energy_B == pytest.approx(-0.345080, abs=1e-3)
    assert frag2_energy_A == pytest.approx(-0.389153, abs=1e-3)
    assert frag2_energy_B == pytest.approx(-0.394914, abs=1e-3)


def test_get_sfo_orbital_energy_unrestricted_largecore_fragsym_c3v(calc_analyzer_unrestricted_largecore_fragsym_c3v):
    """Tests the `get_sfo_orbital_energy` method for a unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_c3v
    frag1_energy_A = analyzer.get_sfo_orbital_energy(1, "4_A1_A")
    frag2_energy_A = analyzer.get_sfo_orbital_energy(2, "4_A1_A")
    frag1_energy_B = analyzer.get_sfo_orbital_energy(1, "4_A1_B")
    frag2_energy_B = analyzer.get_sfo_orbital_energy(2, "4_A1_B")
    assert frag1_energy_A == pytest.approx(-0.139055, abs=1e-3)
    assert frag2_energy_A == pytest.approx(-0.185624, abs=1e-3)
    assert frag1_energy_B == pytest.approx(-0.100877, abs=1e-3)
    assert frag2_energy_B == pytest.approx(-0.264420, abs=1e-3)


def test_get_sfo_orbital_energy_unrestricted_largecore_fragsym_c3v_degenerate(calc_analyzer_unrestricted_largecore_fragsym_c3v):
    """Tests the `get_sfo_orbital_energy` method for a unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_c3v
    frag1_energy_A1 = analyzer.get_sfo_orbital_energy(1, "5_E1:1_A")
    frag1_energy_A2 = analyzer.get_sfo_orbital_energy(1, "5_E1:2_A")
    frag2_energy_A1 = analyzer.get_sfo_orbital_energy(2, "5_E1:1_A")
    frag2_energy_A2 = analyzer.get_sfo_orbital_energy(2, "5_E1:2_A")
    frag1_energy_B1 = analyzer.get_sfo_orbital_energy(1, "5_E1:1_B")
    frag1_energy_B2 = analyzer.get_sfo_orbital_energy(1, "5_E1:2_B")
    frag2_energy_B1 = analyzer.get_sfo_orbital_energy(2, "5_E1:1_B")
    frag2_energy_B2 = analyzer.get_sfo_orbital_energy(2, "5_E1:2_B")
    assert frag1_energy_A1 == pytest.approx(0.011922, abs=1e-3)
    assert frag1_energy_A2 == pytest.approx(0.011922, abs=1e-3)
    assert frag2_energy_A1 == pytest.approx(0.053380, abs=1e-3)
    assert frag2_energy_A2 == pytest.approx(0.053380, abs=1e-3)
    assert frag1_energy_B1 == pytest.approx(0.013414, abs=1e-3)
    assert frag1_energy_B2 == pytest.approx(0.013414, abs=1e-3)
    assert frag2_energy_B1 == pytest.approx(0.046461, abs=1e-3)
    assert frag2_energy_B2 == pytest.approx(0.046461, abs=1e-3)


def test_get_sfo_orbital_energy_unrestricted_nocore_fragsym_nosym(calc_analyzer_unrestricted_nocore_fragsym_nosym):
    """Tests the `get_sfo_orbital_energy` method for a unrestricted, no frozen core, fragment symmetry, no symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_nocore_fragsym_nosym
    frag1_energy_A = analyzer.get_sfo_orbital_energy(1, "10_E1:1_A")
    frag1_energy_B = analyzer.get_sfo_orbital_energy(1, "10_E1:1_B")
    frag2_energy_1B = analyzer.get_sfo_orbital_energy(2, "7_E1:1_B")
    frag2_energy_2B = analyzer.get_sfo_orbital_energy(2, "7_E1:2_B")
    assert frag1_energy_A == pytest.approx(0.010919, abs=1e-3)
    assert frag1_energy_B == pytest.approx(0.012675, abs=1e-3)
    assert frag2_energy_1B == pytest.approx(0.103560, abs=1e-3)
    assert frag2_energy_2B == pytest.approx(0.103560, abs=1e-3)


def test_get_sfo_orbital_energy_unrestricted_largecore_fragsym_nosym(calc_analyzer_unrestricted_largecore_fragsym_nosym):
    """Tests the `get_sfo_orbital_energy` method for a unrestricted, large frozen core, fragment symmetry, no symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_nosym
    frag1_energy_A = analyzer.get_sfo_orbital_energy(1, "5_E1:1_A")
    frag1_energy_B = analyzer.get_sfo_orbital_energy(1, "5_E1:1_B")
    frag2_energy_1B = analyzer.get_sfo_orbital_energy(2, "7_E1:1_B")
    frag2_energy_2B = analyzer.get_sfo_orbital_energy(2, "7_E1:2_B")
    assert frag1_energy_A == pytest.approx(0.011922, abs=1e-3)
    assert frag2_energy_1B == pytest.approx(0.228625, abs=1e-3)
    assert frag1_energy_B == pytest.approx(0.013414, abs=1e-3)
    assert frag2_energy_2B == pytest.approx(0.228625, abs=1e-3)


def test_get_sfo_orbital_energy_non_relativistic(calc_analyzer_non_relativistic):
    """Tests the `get_sfo_orbital_energy` method for a non-relativistic, unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_non_relativistic
    frag1_energy_A = analyzer.get_sfo_orbital_energy(1, "4_A1_A")
    frag2_energy_A = analyzer.get_sfo_orbital_energy(2, "4_A1_A")
    frag1_energy_B = analyzer.get_sfo_orbital_energy(1, "4_A1_B")
    frag2_energy_B = analyzer.get_sfo_orbital_energy(2, "4_A1_B")
    assert frag1_energy_A == pytest.approx(-0.139521, abs=1e-3)
    assert frag2_energy_A == pytest.approx(-0.185638, abs=1e-3)
    assert frag1_energy_B == pytest.approx(-0.100540, abs=1e-3)
    assert frag2_energy_B == pytest.approx(-0.264582, abs=1e-3)


def test_get_sfo_orbital_energy_unrestricted_largecore_nosym(calc_analyzer_unrestricted_largecore_nosym):
    """Tests the `get_sfo_orbital_energy` method for a unrestricted, large frozen core, no fragment symmetry, no symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_largecore_nosym
    frag1_energy_A = analyzer.get_sfo_orbital_energy(1, "13_A_A")
    frag1_energy_B = analyzer.get_sfo_orbital_energy(1, "13_A_B")
    frag2_energy_A = analyzer.get_sfo_orbital_energy(2, "13_A_A")
    frag2_energy_B = analyzer.get_sfo_orbital_energy(2, "13_A_B")
    assert frag1_energy_A == pytest.approx(-0.139055, abs=1e-3)
    assert frag1_energy_B == pytest.approx(-0.100878, abs=1e-3)
    assert frag2_energy_A == pytest.approx(-0.185623, abs=1e-3)
    assert frag2_energy_B == pytest.approx(-0.264419, abs=1e-3)


# NOT WORKING YET
# def test_get_sfo_orbital_energy_unrestricted_largecore_differentfragsym_c4v(calc_analyzer_unrestricted_largecore_differentfragsym_c4v):
#     """ Tests the `get_sfo_orbital_energy` method for a unrestricted, large frozen core, different fragment symmetry (Clin + C4v), c4v complex symmetry calculation ."""
#     analyzer = calc_analyzer_unrestricted_largecore_differentfragsym_c4v
#     frag1_energy = analyzer.get_sfo_orbital_energy(1, "2_SIGMA")
#     frag2_energy = analyzer.get_sfo_orbital_energy(2, "5_A2")
#     assert frag1_energy == pytest.approx(-0.513043, abs=1e-3)
#     assert frag2_energy == pytest.approx(-0.510990, abs=1e-3)

# ------------------------------------------------------------
# ------------------Gross population tests--------------------
# ------------------------------------------------------------

# The following tests are for the `get_sfo_gross_population method for various types of calculations.
# This implementation is much less straightforward and requires taking into account frozen code orbitals, symmetry of the fragments (SFOs), and symmetry of the complex


def test_get_sfo_gross_population_unrestricted_nocore_fragsym_c3v(calc_analyzer_unrestricted_nocore_fragsym_c3v):
    """Tests the `get_sfo_gross_population` method for a unrestricted, no frozen core, fragment symmetry, c3v complex symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_nocore_fragsym_c3v
    frag1_pop_A = analyzer.get_sfo_gross_population(1, "11_A1_A")
    frag2_pop_A = analyzer.get_sfo_gross_population(2, "6_A1_A")
    frag1_pop_B = analyzer.get_sfo_gross_population(1, "11_A1_B")
    frag2_pop_B = analyzer.get_sfo_gross_population(2, "6_A1_B")
    frag1_pop_1A = analyzer.get_sfo_gross_population(1, "9_E1:2_A")
    frag2_pop_1A = analyzer.get_sfo_gross_population(2, "5_E1:2_A")
    frag1_pop_2B = analyzer.get_sfo_gross_population(1, "9_E1:2_B")
    frag2_pop_2B = analyzer.get_sfo_gross_population(2, "5_E1:2_B")
    assert frag1_pop_A == pytest.approx(0.412, abs=1e-3)
    assert frag2_pop_A == pytest.approx(0.608, abs=1e-3)
    assert frag1_pop_B == pytest.approx(0.382, abs=1e-3)
    assert frag2_pop_B == pytest.approx(0.665, abs=1e-3)
    assert frag1_pop_1A == pytest.approx(0.996, abs=1e-3)
    assert frag2_pop_1A == pytest.approx(0.995, abs=1e-3)
    assert frag1_pop_2B == pytest.approx(0.995, abs=1e-3)
    assert frag2_pop_2B == pytest.approx(0.996, abs=1e-3)


def test_get_sfo_gross_population_unrestricted_largecore_fragsym_c3v(calc_analyzer_unrestricted_largecore_fragsym_c3v):
    """Tests the `get_sfo_gross_population` method for a unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_c3v
    frag1_pop_A = analyzer.get_sfo_gross_population(1, "4_A1_A")
    frag2_pop_A = analyzer.get_sfo_gross_population(2, "4_A1_A")
    frag1_pop_B = analyzer.get_sfo_gross_population(1, "4_A1_B")
    frag2_pop_B = analyzer.get_sfo_gross_population(2, "4_A1_B")
    assert frag1_pop_A == pytest.approx(0.408, abs=1e-3)
    assert frag2_pop_A == pytest.approx(0.607, abs=1e-3)
    assert frag1_pop_B == pytest.approx(0.380, abs=1e-3)
    assert frag2_pop_B == pytest.approx(0.663, abs=1e-3)


def test_get_sfo_gross_population_unrestricted_largecore_fragsym_c3v_degenerate(calc_analyzer_unrestricted_largecore_fragsym_c3v):
    """Tests the `get_sfo_gross_population` method for a unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_c3v
    frag1_pop_1A = analyzer.get_sfo_gross_population(1, "5_E1:1_A")
    frag1_pop_2A = analyzer.get_sfo_gross_population(1, "5_E1:2_A")
    frag2_pop_1A = analyzer.get_sfo_gross_population(2, "5_E1:1_A")
    frag2_pop_2A = analyzer.get_sfo_gross_population(2, "5_E1:2_A")
    frag1_pop_1B = analyzer.get_sfo_gross_population(1, "5_E1:1_B")
    frag1_pop_2B = analyzer.get_sfo_gross_population(1, "5_E1:2_B")
    frag2_pop_1B = analyzer.get_sfo_gross_population(2, "5_E1:1_B")
    frag2_pop_2B = analyzer.get_sfo_gross_population(2, "5_E1:2_B")
    assert frag1_pop_1A == pytest.approx(frag1_pop_2A, abs=1e-3)
    assert frag2_pop_1A == pytest.approx(frag2_pop_2A, abs=1e-3)
    assert frag1_pop_1A == pytest.approx(0.003, abs=1e-3)
    assert frag2_pop_1A == pytest.approx(0.003, abs=1e-3)
    assert frag1_pop_1B == pytest.approx(frag1_pop_2B, abs=1e-3)
    assert frag2_pop_1B == pytest.approx(frag2_pop_2B, abs=1e-3)
    assert frag1_pop_1B == pytest.approx(0.002, abs=1e-3)
    assert frag2_pop_1B == pytest.approx(0.004, abs=1e-3)


def test_get_sfo_gross_population_unrestricted_largecore_fragsym_nosym(calc_analyzer_unrestricted_largecore_fragsym_nosym):
    """Tests the `get_sfo_gross_population` method for a unrestricted, large frozen core, fragment symmetry, no symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_nosym
    frag1_pop_1A = analyzer.get_sfo_gross_population(1, "4_A1_A")
    frag1_pop_1B = analyzer.get_sfo_gross_population(1, "4_A1_B")
    frag1_pop_2A = analyzer.get_sfo_gross_population(1, "2_A2_A")
    frag1_pop_2B = analyzer.get_sfo_gross_population(1, "2_A2_B")
    frag2_pop_1A = analyzer.get_sfo_gross_population(2, "4_A1_A")
    frag2_pop_2B = analyzer.get_sfo_gross_population(2, "4_A1_B")
    assert frag1_pop_1A == pytest.approx(0.408, abs=1e-3)
    assert frag1_pop_1B == pytest.approx(0.380, abs=1e-3)
    assert frag1_pop_2A == pytest.approx(0.000, abs=1e-3)
    assert frag1_pop_2B == pytest.approx(0.000, abs=1e-3)
    assert frag2_pop_1A == pytest.approx(0.607, abs=1e-3)
    assert frag2_pop_2B == pytest.approx(0.663, abs=1e-3)


def test_get_sfo_gross_population_unrestricted_nocore_fragsym_nosym(calc_analyzer_unrestricted_nocore_fragsym_nosym):
    """Tests the `get_sfo_gross_population` method for a unrestricted, no frozen core, fragment symmetry, no symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_nocore_fragsym_nosym
    frag1_pop_1A = analyzer.get_sfo_gross_population(1, "11_A1_A")
    frag1_pop_1B = analyzer.get_sfo_gross_population(1, "11_A1_B")
    frag1_pop_2A = analyzer.get_sfo_gross_population(1, "3_A2_A")
    frag1_pop_2B = analyzer.get_sfo_gross_population(1, "3_A2_B")
    frag2_pop_1B = analyzer.get_sfo_gross_population(2, "6_A1_A")
    frag2_pop_2B = analyzer.get_sfo_gross_population(2, "6_A1_B")
    assert frag1_pop_1A == pytest.approx(0.412, abs=1e-3)
    assert frag1_pop_1B == pytest.approx(0.382, abs=1e-3)
    assert frag1_pop_2A == pytest.approx(0.000, abs=1e-3)
    assert frag1_pop_2B == pytest.approx(0.000, abs=1e-3)
    assert frag2_pop_1B == pytest.approx(0.665, abs=1e-3)
    assert frag2_pop_2B == pytest.approx(0.608, abs=1e-3)


def test_get_sfo_gross_population_unrestricted_largecore_nosym(calc_analyzer_unrestricted_largecore_nosym):
    """Tests the `get_sfo_gross_population` method for a unrestricted, large frozen core, no fragment symmetry, no symmetry calculation ."""
    analyzer = calc_analyzer_unrestricted_largecore_nosym
    frag1_pop_1A = analyzer.get_sfo_gross_population(1, "13_A_A")
    frag1_pop_1B = analyzer.get_sfo_gross_population(1, "13_A_B")
    frag2_pop_1A = analyzer.get_sfo_gross_population(2, "13_A_A")
    frag2_pop_2B = analyzer.get_sfo_gross_population(2, "13_A_B")
    assert frag1_pop_1A == pytest.approx(0.408, abs=1e-3)
    assert frag1_pop_1B == pytest.approx(0.380, abs=1e-3)
    assert frag2_pop_1A == pytest.approx(0.607, abs=1e-3)
    assert frag2_pop_2B == pytest.approx(0.664, abs=1e-3)


# NOT WORKING YET
# def test_get_sfo_gross_population_unrestricted_largecore_differentfragsym_c4v(calc_analyzer_unrestricted_largecore_differentfragsym_c4v):
#     """ Tests the `get_sfo_gross_population` method for a unrestricted, large frozen core, different fragment symmetry (Clin + C4v), c4v complex symmetry calculation ."""
#     analyzer = calc_analyzer_unrestricted_largecore_differentfragsym_c4v
#     frag1_pop1 = analyzer.get_sfo_gross_population(1, "2_SIGMA")
#     frag1_pop2 = analyzer.get_sfo_gross_population(1, "1_PI:y")
#     frag1_pop3 = analyzer.get_sfo_gross_population(1, "1_DELTA:xy")
#     frag1_pop4 = analyzer.get_sfo_gross_population(1, "3_SIGMA")
#     frag2_pop1 = analyzer.get_sfo_gross_population(2, "2_E1:2")
#     frag2_pop2 = analyzer.get_sfo_gross_population(2, "6_E1:1")
#     frag2_pop3 = analyzer.get_sfo_gross_population(2, "8_A1")
#     frag2_pop4 = analyzer.get_sfo_gross_population(2, "5_B2")
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

# The following tests are for the `get_sfo_overlap method` for various types of calculations.
# This implementation is also not straightforward and requires taking into account frozen code orbitals, symmetry of the fragments (SFOs), and symmetry of the complex
# Note that the overlap matrix is a lower triangular matrix, so only the lower triangular part is stored in the KFFile.
# The overlap matrix is splitted into irreps of the complex symmetry and skips the frozen core orbital.
# Therefore, one should be careful with SFO indices. The SFO indices are not the same as the indices of the overlap matrix.


def test_get_sfo_overlap_unrestricted_nocore_fragsym_c3v(calc_analyzer_unrestricted_nocore_fragsym_c3v):
    """Tests the `get_sfo_overlap` method for a unrestricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_nocore_fragsym_c3v
    self_overlap = analyzer.get_sfo_overlap("9_A1_A", "9_A1_A")
    homo_lumo_overlap = analyzer.get_sfo_overlap("11_A1_A", "6_A1_A")
    homo_homo_overlap = analyzer.get_sfo_overlap("10_A1_A", "5_A1_A")
    pauli_overlap_1 = analyzer.get_sfo_overlap("7_E1:1_A", "5_E1:1_A")
    pauli_overlap_2 = analyzer.get_sfo_overlap("7_E1:1_B", "5_E1:1_B")
    assert self_overlap == pytest.approx(self_overlap, abs=1e-3)
    assert homo_lumo_overlap == pytest.approx(0.4347, abs=1e-3)
    assert homo_homo_overlap == pytest.approx(-0.0742, abs=1e-3)
    assert pauli_overlap_1 == pytest.approx(-0.0260, abs=1e-3)
    assert pauli_overlap_2 == pytest.approx(0.0228, abs=1e-3)


def test_get_sfo_overlap_unrestricted_largecore_fragsym_c3v_A1(calc_analyzer_unrestricted_largecore_fragsym_c3v):
    """Tests the `get_sfo_overlap` method for a unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_c3v
    homo_lumo_overlap_A = analyzer.get_sfo_overlap("4_A1_A", "4_A1_A")
    homo_homo_overlap_A = analyzer.get_sfo_overlap("3_E1:1_A", "4_E1:1_A")
    homo_lumo_overlap_B = analyzer.get_sfo_overlap("4_A1_B", "4_A1_B")
    homo_homo_overlap_B = analyzer.get_sfo_overlap("3_E1:1_B", "4_E1:1_B")
    assert homo_lumo_overlap_A == pytest.approx(-0.4384, abs=1e-3)
    assert homo_homo_overlap_A == pytest.approx(0.0207, abs=1e-3)
    assert homo_lumo_overlap_B == pytest.approx(-0.3751, abs=1e-3)
    assert homo_homo_overlap_B == pytest.approx(0.0212, abs=1e-3)


def test_get_sfo_overlap_unrestricted_largecore_fragsym_c3v_A2(calc_analyzer_unrestricted_largecore_fragsym_c3v):
    """Tests the `get_sfo_overlap` method for a unrestricted, large frozen core, fragment symmetry, c3v complex symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_largecore_fragsym_c3v
    homo_lumo_overlap_A = analyzer.get_sfo_overlap("5_A2_A", "2_A2_A")
    homo_lumo_overlap_B = analyzer.get_sfo_overlap("5_A2_B", "2_A2_B")
    assert homo_lumo_overlap_A == pytest.approx(0.0680, abs=1e-3)
    assert homo_lumo_overlap_B == pytest.approx(0.0678, abs=1e-3)


def test_get_sfo_overlap_unrestricted_nocore_fragsym_nosym(calc_analyzer_unrestricted_nocore_fragsym_nosym):
    """Tests the `get_sfo_overlap` method for a unrestricted, no frozen core, fragment symmetry, no complex symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_nocore_fragsym_nosym
    homo_lumo_overlap_A = analyzer.get_sfo_overlap("11_A1_A", "6_A1_A")
    homo_lumo_overlap_B = analyzer.get_sfo_overlap("11_A1_B", "6_A1_B")
    homo_homo_overlap_A = analyzer.get_sfo_overlap("11_A1_A", "5_A1_A")
    homo_homo_overlap_B = analyzer.get_sfo_overlap("9_A1_B", "6_A1_B")
    extra_overlap_A = analyzer.get_sfo_overlap("8_E1:2_A", "5_E1:2_A")  # 40A_A MO
    extra_overlap_B = analyzer.get_sfo_overlap("8_E1:2_B", "5_E1:2_B")  # 40A_B MO
    assert homo_lumo_overlap_A == pytest.approx(0.4347, abs=1e-3)
    assert homo_lumo_overlap_B == pytest.approx(0.3711, abs=1e-3)
    assert homo_homo_overlap_A == pytest.approx(0.2691, abs=1e-3)
    assert homo_homo_overlap_B == pytest.approx(0.1907, abs=1e-3)
    assert extra_overlap_A == pytest.approx(0.0223, abs=1e-3)
    assert extra_overlap_B == pytest.approx(0.0224, abs=1e-3)


def test_get_sfo_overlap_unrestricted_largecore_nosym(calc_analyzer_unrestricted_largecore_nosym):
    """Tests the `get_sfo_overlap` method for a unrestricted, large frozen core, no fragment symmetry, no complex symmetry calculation."""
    analyzer = calc_analyzer_unrestricted_largecore_nosym
    homo_lumo_overlap_A = analyzer.get_sfo_overlap("13_A_A", "13_A_A")
    homo_lumo_overlap_B = analyzer.get_sfo_overlap("13_A_B", "13_A_B")
    homo_homo_overlap_A = analyzer.get_sfo_overlap("13_A_A", "7_A_A")
    homo_homo_overlap_B = analyzer.get_sfo_overlap("4_A_B", "13_A_B")
    assert homo_lumo_overlap_A == pytest.approx(-0.4384, abs=1e-3)
    assert homo_lumo_overlap_B == pytest.approx(-0.3751, abs=1e-3)
    assert homo_homo_overlap_A == pytest.approx(-0.2709, abs=1e-3)
    assert homo_homo_overlap_B == pytest.approx(-0.1908, abs=1e-3)


# NOT WORKING YET
# def test_get_sfo_overlap_unrestricted_largecore_differentfragsym_c4v(calc_analyzer_unrestricted_largecore_differentfragsym_c4v):
#     """ Tests the `get_sfo_overlap` method for a unrestricted, no frozen core, fragment symmetry, c3v complex symmetry calculation."""
#     analyzer = calc_analyzer_unrestricted_largecore_differentfragsym_c4v
#     homo_lumo_overlap = analyzer.get_sfo_overlap("3_SIGMA", "8_A1")
#     homo_lumo_overlap2 = analyzer.get_sfo_overlap("3_SIGMA", "7_A1")
#     assert homo_lumo_overlap == pytest.approx(0.3696, abs=1e-3)
#     assert homo_lumo_overlap2 == pytest.approx(0.0912, abs=1e-3)
