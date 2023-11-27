"""
Module containing functions for extracting Molecular Orbital (MO) data from the rkf files of fragment analysis calculations.
The following terms can be extracted form the rkf files:
- Overlap (in a.u.)
- Orbital Energy (in eV)
- Occupation (in a.u.)

BEFORE READING FURTHER: please read the following section about the format of the rkf files:
https://www.scm.com/doc/ADF/Appendices/TAPE21.html

Important sections together with associated variables are (format: ("section", "variable")):
- "Symmetry", "symlab"  = Symmetry labels of the irreps
- "Symmetry", "ncbs"    = Number of frozen cores per irrep
- "[IRREP]", "froc_[SPIN]" = Orbital occupations for SPIN ("A"/"B") scaled by relativistic effects (ZORA).
- "[IRREP]", "escale_[SPIN]" = Orbital energies for SPIN ("A"/"B") scaled by relativistic effects (ZORA). If no relativistic effects are present, "eps_[SPIN]" is the appropriate key.

that can be viewed in the KF Browser of AMS (open a "adf.rkf" file and press "ctrl + E" on Windows or "cmd + E" on Mac).
* Active SFOs are SFOs that are not frozen cores.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Sequence

import numpy as np
from scm.plams import KFFile
from orb_analysis.custom_types import UnrestrictedPropertyDict
from orb_analysis.custom_types import Array1D, SpinTypes


# -------------------Low-level KF reading -------------------- #

def uses_symmetry(kf_file: KFFile) -> bool:
    """ Returns True if the complex calculation uses symmetry for its MOs and other parts such as gross populations and overlap. """
    grouplabel = kf_file.read("Symmetry", "grouplabel").split()  # type: ignore

    if grouplabel[0].lower() == "nosym":
        return False
    return True


def get_irreps(kf_file: KFFile) -> Sequence[str]:
    """ Returns the ordered symlabels of *active* MOs (frozen core MOs excluded) with the symmetry for MO labeling. """
    irreps = kf_file.read("Symmetry", "symlab", return_as_list=True).split()  # type: ignore
    return irreps


def get_number_MOs_per_irrep_per_frag(kf_file: KFFile, spin: str = "A") -> OrderedDict[str, int]:
    """ Returns the number of *active* SFOs of each irrep (frozen core SFOs excluded) belonging to one fragment. """
    irreps = get_irreps(kf_file)
    sfo_sym_label_sum = OrderedDict({irrep: 0 for irrep in set(irreps)})
    for irrep in irreps:
        sfo_sym_label_sum[irrep] += int(kf_file.read(irrep, f"nmo_{spin}"))  # type: ignore
    return sfo_sym_label_sum


# --------------------Frozen Core Handling-------------------- #

def get_frozen_cores_per_irrep(kf_file: KFFile) -> dict[str, int]:
    """
    Reads the number of frozen cores per irrep from the KFFile. This function is much simpler than the SFO variant because there is just one symmetry instead of two

    The number of frozen cores per irrep is important for getting gross populations and overlap analysis.
    Basically, the SFO index shown in AMSLevels is different than the index shown in the overlap and population analysis because they can be shifted by frozen cores.
    """
    ordered_frag_sym_labels = get_irreps(kf_file)
    n_core_orbs_per_irrep: list[int] = kf_file.read("Symmetry", "ncbs", return_as_list=True)  # type: ignore since n_core_orbs is a list of ints
    frozen_core_per_irrep = {irrep: n_frozen_cores for irrep, n_frozen_cores in zip(ordered_frag_sym_labels, n_core_orbs_per_irrep)}  # type: ignore

    return frozen_core_per_irrep

# -------------------Property Function(s)-------------------- #


def read_MO_energies(kf_file: KFFile, irrep: str, spin: str) -> Array1D[np.float64]:
    """ Reads the molecular orbital energies from the KFFile. """
    # escale refers energies scaled by relativistic effects (ZORA). If no relativistic effects are present, "energy" is the appropriate key.
    variable = f"escale_{spin}"
    if (irrep, f"escale_{spin}") not in kf_file:
        variable = f"eps_{spin}"

    # Reads the orbital energies for both fragments and selects the data for the current fragment
    orb_energies = np.array(kf_file.read(irrep, variable))  # type: ignore

    return orb_energies


def read_MO_occupations(kf_file: KFFile, irrep: str, spin: str) -> Array1D[np.float64]:
    """ Reads the molecular orbital occupations from the KFFile. """
    occupations = np.array(kf_file.read(irrep, f"froc_{spin}"))  # type: ignore
    return occupations


# --------------------Property to Function Mapping-------------------- #


# Format: {property: (callable function for reading property, section in KFFile, variable in KFFile)}
KEY_FUNC_MAPPING: dict[str, Callable] = {
    "orb_energies": read_MO_energies,
    "occupations": read_MO_occupations,
}

# --------------------Interface Function(s)-------------------- #


def get_complex_properties(kf_file: KFFile, restricted: bool = True) -> UnrestrictedPropertyDict:
    """
    Returns a dictionary of dictionaries with the properties of the fragments.

    The properties are:
        - Orbital Energies
        - Occupations

    Format of the data:
    {
        "orb_energies":
            {
                "A": {"IRREP1": [...], "IRREP2": [...]},
                "B": {"IRREP1": [...], "IRREP2": [...]},
            },
        "occupations":
            {
                "A": {"IRREP1": [...], "IRREP2": [...]},
                "B": {"IRREP1": [...], "IRREP2": [...]},
            },
    }
    """
    irreps = get_irreps(kf_file)
    spin_states = SpinTypes if not restricted else SpinTypes.A

    data_dic_to_be_unpacked: dict[str, dict[str, dict[str, Array1D[np.float64]]]] = {}

    for property, func in KEY_FUNC_MAPPING.items():
        data_dic_to_be_unpacked[property] = {}
        for spin in spin_states:
            irrep_property_dic = {irrep: func(kf_file, irrep, spin) for irrep in irreps}
            data_dic_to_be_unpacked[property][spin] = irrep_property_dic

    return data_dic_to_be_unpacked


def main():
    import pathlib as pl

    current_dir = pl.Path(__file__).parent
    rkf_dir = current_dir.parent.parent.parent / "test" / "fixtures" / "rkfs"
    rkf_file = "restricted_largecore_differentfragsym_c4v_full.adf.rkf"
    kf_file = KFFile(str(rkf_dir / rkf_file))

    print(get_complex_properties(kf_file))


if __name__ == "__main__":
    main()
