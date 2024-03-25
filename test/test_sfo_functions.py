"""
Testmodule that tests low level the sfo orbital functions.
"""

import pathlib as pl

from orb_analysis import orb_config
from orb_analysis.orb_functions.sfo_functions import (
    get_frag_name,
    get_irrep_each_sfo_one_frag,
    get_number_sfos_per_irrep_per_frag,
    get_ordered_irreps_of_one_frag,
    get_sfo_indices_of_one_frag,
    uses_symmetry,
)
from scm.plams import KFFile

current_dir = pl.Path(__file__).parent
fixtures_dir = current_dir / "fixtures" / "rkfs"

restricted_largecore_fragsym_c3v = fixtures_dir / "restricted_largecore_fragsym_c3v_full.adf.rkf"
restricted_largecore_differentfragsym_c4v = fixtures_dir / "restricted_largecore_differentfragsym_c4v_full.adf.rkf"
restricted_largecore_nofragsym_nosym = fixtures_dir / "restricted_largecore_nofragsym_nosym_full.adf.rkf"

# The orbital energy key and unit need to be fixed because the tests have been written for the conditions below
orb_config.rkf_reading.orbital_energy_key = "escale"
orb_config.rkf_reading.orbital_energy_unit = "hartree"

# ------------------------------------------------------------
# ---------------------- Unit tests --------------------------
# ------------------------------------------------------------


def test_get_frag_name():
    kf_file = KFFile(restricted_largecore_fragsym_c3v)
    frag1_name = get_frag_name(kf_file, 1)
    frag2_name = get_frag_name(kf_file, 2)

    assert isinstance(frag1_name, str)
    assert frag1_name in "f1"
    assert frag2_name in "f2"


def test_uses_symmetry():
    kf_file_sym = KFFile(restricted_largecore_fragsym_c3v)
    kf_file_nosym = KFFile(restricted_largecore_nofragsym_nosym)
    symmetry = uses_symmetry(kf_file_sym)

    assert isinstance(symmetry, bool)
    assert symmetry is True
    assert uses_symmetry(kf_file_nosym) is False


def test_get_sfo_indices_of_one_frag():
    expected_indices_frag1 = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        42,
        43,
        44,
        45,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
    ]
    expected_indices_frag2 = [
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
    ]

    kf_file = KFFile(restricted_largecore_fragsym_c3v)
    sfo_indices_frag1 = get_sfo_indices_of_one_frag(kf_file, 1)
    sfo_indices_frag2 = get_sfo_indices_of_one_frag(kf_file, 2)

    assert isinstance(sfo_indices_frag1, list)
    assert len(sfo_indices_frag1) == 57
    assert len(sfo_indices_frag2) == 96
    assert sfo_indices_frag1 == expected_indices_frag1
    assert sfo_indices_frag2 == expected_indices_frag2


def test_get_ordered_irreps_of_one_frag():
    kf_file_different_fragsym = KFFile(restricted_largecore_differentfragsym_c4v)
    kf_file_nosym = KFFile(restricted_largecore_nofragsym_nosym)
    irreps_different_fragsym1 = get_ordered_irreps_of_one_frag(kf_file_different_fragsym, 1)
    irreps_different_fragsym2 = get_ordered_irreps_of_one_frag(kf_file_different_fragsym, 2)
    irreps_nosym = get_ordered_irreps_of_one_frag(kf_file_nosym, 1)

    assert irreps_different_fragsym1 == ["SIGMA", "DELTA:x2-y2", "DELTA:xy", "PI:x", "PHI:x3-3xy2", "PI:y", "PHI:3x2y-y3"]
    assert irreps_different_fragsym2 == ["A1", "A2", "B1", "B2", "E1:1", "E1:2"]
    assert irreps_nosym == ["A"]


def test_get_irrep_each_sfo_one_frag():
    kf_file = KFFile(restricted_largecore_fragsym_c3v)
    irrep_each_sfo = get_irrep_each_sfo_one_frag(kf_file, 1)
    assert isinstance(irrep_each_sfo, list)


def test_get_number_sfos_per_irrep_per_frag():
    kf_file_c3v = KFFile(restricted_largecore_fragsym_c3v)
    kf_file_different_fragsym = KFFile(restricted_largecore_differentfragsym_c4v)
    kf_file_different_nosym = KFFile(restricted_largecore_nofragsym_nosym)
    n_sfos_per_irrep_different1 = get_number_sfos_per_irrep_per_frag(kf_file_different_fragsym, 1)
    n_sfos_per_irrep_different2 = get_number_sfos_per_irrep_per_frag(kf_file_different_fragsym, 2)
    n_sfos_per_irrep_c3v_1 = get_number_sfos_per_irrep_per_frag(kf_file_c3v, 1)
    n_sfos_per_irrep_c3v_2 = get_number_sfos_per_irrep_per_frag(kf_file_c3v, 2)
    n_sfos_per_irrep_nosym = get_number_sfos_per_irrep_per_frag(kf_file_different_nosym, 1)

    assert isinstance(n_sfos_per_irrep_different1, dict)
    assert n_sfos_per_irrep_different1 == {"PI:x": 10, "DELTA:x2-y2": 4, "DELTA:xy": 4, "SIGMA": 16, "PHI:x3-3xy2": 2, "PI:y": 10, "PHI:3x2y-y3": 2}
    assert n_sfos_per_irrep_different2 == {"A1": 56, "B1": 24, "E1:1": 66, "A2": 16, "E1:2": 66, "B2": 40}
    assert n_sfos_per_irrep_c3v_1 == {"A1": 17, "A2": 4, "E1:1": 18, "E1:2": 18}
    assert n_sfos_per_irrep_c3v_2 == {"A1": 25, "A2": 9, "E1:1": 31, "E1:2": 31}
    assert n_sfos_per_irrep_nosym == {"A": 57}
    assert len(n_sfos_per_irrep_nosym.keys()) == 1


# ------------------------------------------------------------
