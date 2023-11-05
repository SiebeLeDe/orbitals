"""
Module containing functions for extracting SFO data from the rkf files of fragment analysis calculations.
The following terms can be extracted form the rkf files:
- Overlap (in a.u.)
- Orbital Energy (in eV)
- Occupation (in a.u.)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Sequence

import numpy as np
from scm.plams import KFFile

from orb_analysis.custom_types import Array1D

# --------------------Helper Function(s)-------------------- #


def split_1d_array_into_dict_sorted_by_symlabels(data_array: Array1D, symlabels: Sequence[str], ) -> dict[str, Array1D]:
    """
    Splits a 1D array into a dictionary of arrays based on the symlabels. The symlabels are the keys of the dictionary.
    data_array and symlabels must have the same length.
    """
    n_entries_per_symlabel = [symlabels.count(symlabel) for symlabel in set(symlabels)]
    data_ordered_by_symlabel = {symlabel: np.zeros(n_entries) for symlabel, n_entries in zip(set(symlabels), n_entries_per_symlabel)}

    # Now we loop over the symlabels and add the data to the right array
    sym_label_counter = {symlabel: 0 for symlabel in set(symlabels)}
    for data_entry, symlabel in zip(data_array, symlabels):
        data_ordered_by_symlabel[symlabel][sym_label_counter[symlabel]] = data_entry
        sym_label_counter[symlabel] += 1
    return data_ordered_by_symlabel


# -------------------Low-level KF reading -------------------- #
def uses_symmetry(kf_file: KFFile) -> bool:
    """ Returns True if the complex calculation uses symmetry for its MOs and other parts such as gross populations and overlap. """
    grouplabel = kf_file.read("Symmetry", "grouplabel").split()  # type: ignore
    
    if grouplabel[0].lower() == "nosym":
        return False
    return True

def get_sfo_indices_of_one_frag(kf_file: KFFile, frag_index: int) -> Sequence[int]:
    """ Returns the indices of *active* SFOs belonging to one fragment. """
    sfo_frag_indices: list[int] = kf_file.read("SFOs", "fragment", return_as_list=True)  # type: ignore
    sfo_frag_indices = [int(sfo_index) for sfo_index in sfo_frag_indices if sfo_index == frag_index]
    
    return sfo_frag_indices

def get_ordered_symlabels_of_one_frag(kf_file: KFFile, frag_index: int) -> Sequence[str]:
    """ Returns the ordered symlabels of *active* SFOs (frozen core SFOs excluded) belonging to one fragment. """    
    sfo_frag_index: list[int] = kf_file.read("SFOs", "fragment", return_as_list=True)  # type: ignore
    sfo_sym_labels: list[str] = kf_file.read("SFOs", "subspecies", return_as_list=True).split()  # type: ignore
    
    sfo_sym_labels_of_one_frag = []
    for sfo_index, sfo_sym_label in zip(sfo_frag_index, sfo_sym_labels):
        if sfo_index == frag_index and sfo_sym_label not in sfo_sym_labels_of_one_frag:
            sfo_sym_labels_of_one_frag.append(sfo_sym_label)
 
    return sfo_sym_labels_of_one_frag
        

def get_number_sfos_per_irrep_per_frag(kf_file: KFFile, frag_index: int) -> OrderedDict[str, int]:
    """ Returns the number of *active* SFOs of one irrep (frozen core SFOs excluded) belonging to one fragment. """
    sfo_frag_index: list[int] = kf_file.read("SFOs", "fragment", return_as_list=True)  # type: ignore
    sfo_sym_labels: list[str] = kf_file.read("SFOs", "subspecies", return_as_list=True).split()  # type: ignore
    
    sfo_sym_label_sum = OrderedDict({sym_label: 0 for sym_label in set(sfo_sym_labels)})
    for sfo_index, sfo_sym_label in zip(sfo_frag_index, sfo_sym_labels):
        if sfo_index == frag_index:
            sfo_sym_label_sum[sfo_sym_label] += 1

    return sfo_sym_label_sum

# --------------------Frozen Core Handling-------------------- #

def get_frozen_cores_per_irrep(kf_file: KFFile, frag_index: int) -> dict[str, int]:
    """
    Reads the number of frozen cores per irrep from the KFFile.

    The number of frozen cores per irrep is important for getting gross populations and overlap analysis.
    Basically, the SFO index shown in AMSLevels is different than the index shown in the overlap and population analysis because they can be shifted by frozen cores.

    Moreover, if the complex calculation uses symmetry, but the fragments themselves do not, then the "A" irrep is added, being the sum of all frozen cores.

    In case there is no frozen core and no symmetry, but the fragments use symmetry, then the frozen core is 0 for all irreps that are present in the fragments.
    """
    ordered_frag_sym_labels = get_ordered_symlabels_of_one_frag(kf_file, frag_index=frag_index)
    n_core_orbs_per_irrep: list[int] = kf_file.read("Symmetry", "ncbs", return_as_list=True)  # type: ignore since n_core_orbs is a list of ints  
    
    frozen_core_per_irrep = {irrep: n_frozen_cores for irrep, n_frozen_cores in zip(ordered_frag_sym_labels, n_core_orbs_per_irrep)}  # type: ignore
    
    # Add the "A" irrep to the dictionary for the case when symmetry is not used (e.g. NoSym), but the fragments themselves use symmetry.
    # This is only used for the overlap analysis.    
    if not uses_symmetry(kf_file):
        frozen_core_per_irrep["A"] = sum(n_core_orbs_per_irrep)
    return frozen_core_per_irrep

    # # This is the case when there are no frozen cores and without symmetry, but the fragments themselves use symmetry...
    # if len(n_core_orbs) == 1 and n_core_orbs[0] == 0:
    #     return n_frozen_cores_per_irrep_summed | {irrep: 0 for irrep in sym_labels_right_order}

# --------------------Restricted Property Function(s)-------------------- #


def read_restricted_gross_populations(kf_file: KFFile, section: str, variable: str) -> Array1D[np.float64]:
    """ Reads the gross populations from the KFFile. """
    gross_populations = np.array(kf_file.read(section, variable))  # type: ignore
    return gross_populations


def read_restricted_orbital_energies(kf_file: KFFile, section: str, variable: str) -> Array1D[np.float64]:
    """ Reads the orbital energies from the KFFile. """
    # escale refers energies scaled by relativistic effects (ZORA). If no relativistic effects are present, "energy" is the appropriate key.
    if ("SFOs", "escale") not in kf_file:
        variable = "energy"

    # Reads the orbital energies for both fragments and selects the data for the current fragment
    orb_energies = np.array(kf_file.read(section, variable))  # type: ignore

    return orb_energies


def read_restricted_occupations(kf_file: KFFile, section: str, variable: str) -> Array1D[np.float64]:
    """ Reads the occupations from the KFFile. """
    occupations = np.array(kf_file.read(section, variable))  # type: ignore
    return occupations


# --------------------Unrestricted Property Function(s)-------------------- #


# --------------------Property to Function Mapping-------------------- #


# Format: {property: (callable function for reading property, section in KFFile, variable in KFFile)}
RESTRICTED_KEY_FUNC_MAPPING: dict[str, tuple[Callable, str, str]] = {
    "orb_energies": (read_restricted_orbital_energies, "SFOs", "escale"),
    "occupations": (read_restricted_occupations, "SFOs", "occupation"),
}

# --------------------Interface Function(s)-------------------- #


def get_restricted_fragment_properties(kf_file: KFFile, sfo_indices_of_one_frag: Sequence[int], frag_symlabels_each_sfo: Sequence[str]) -> dict[str, dict[str, Array1D]]:
    """
    Returns a dictionary of dictionaries with the properties of the fragments.

    The properties are:
        - Orbital Energies
        - Occupations
    """
    data_dic_to_be_unpacked: dict[str, dict[str, Array1D[np.float64]]] = {}
    for property, (func, section, variable) in RESTRICTED_KEY_FUNC_MAPPING.items():

        data = func(kf_file, section, variable)

        data = np.array([data[i] for i in sfo_indices_of_one_frag])

        # Now we turn one long array into a dictionary of arrays sorted by symlabels (e.g. [.....] -> {"A1": [.....], "A2": [.....]})
        data_dic_to_be_unpacked[property] = split_1d_array_into_dict_sorted_by_symlabels(data_array=data, symlabels=frag_symlabels_each_sfo)

    return data_dic_to_be_unpacked


def get_gross_populations(kf_file: KFFile, frag_index: int=1) -> dict[str, Array1D[np.float64]]:
    """
    Reads the gross populations from the KFFile by taking into account the frozen cores.
    Annoyingly, the "SFOs" sections contains the SFOs of both fragments that ALREADY HAVE BEEN FILTERED for the frozen cores.
    For example, the SFOs number may be 114, but the gross population array may have 148 entries. This is because the first 34 entries are the frozen cores.

    Structure of the ("SFOs popul","sfo_grosspop") section for a restricted calculation with c3v symmetry:
    [n Frozen Cores A1, Active SFOs Frag1 A1, Active SFOs Frag2 A1, n Frozen Cores A2, Active SFOs Frag1 A2, Active SFOs Frag2 A2, ...] 

    Therefore, the sum of `sfo_indices_of_one_frag` and `n_frozen_cores_per_irrep` is used to get the correct indices for SFOs on both fragments and all irreps.
    """
    symmetry_used = uses_symmetry(kf_file)
    frags_sfo_irrep_sums = [get_number_sfos_per_irrep_per_frag(kf_file, frag_index=frag_index) for frag_index in [1, 2]]
    
    ordered_sym_labels = get_ordered_symlabels_of_one_frag(kf_file, frag_index=frag_index)
    frozen_core_per_irrep = get_frozen_cores_per_irrep(kf_file, frag_index=frag_index)
    
    raw_gross_pop_all_sfos = read_restricted_gross_populations(kf_file, "SFO popul", "sfo_grosspop")

    if not symmetry_used:
        start_index = sum(frozen_core_per_irrep[irrep] for irrep in frozen_core_per_irrep)
        total_sfo_sum_frag1 = sum(frags_sfo_irrep_sums[0][irrep] for irrep in frags_sfo_irrep_sums[0]) 
        total_sfo_sum_frag2 = sum(frags_sfo_irrep_sums[1][irrep] for irrep in frags_sfo_irrep_sums[1]) 
        
        if frag_index == 1:
            return {"A": raw_gross_pop_all_sfos[start_index:total_sfo_sum_frag1]}

        return {"A": raw_gross_pop_all_sfos[start_index + total_sfo_sum_frag1: start_index + total_sfo_sum_frag1 + total_sfo_sum_frag2]}

    gross_pop_active_sfos = {irrep: np.zeros_like(frags_sfo_irrep_sums[frag_index-1][irrep], dtype=np.float64) for irrep in frags_sfo_irrep_sums[frag_index - 1]}

    raw_gross_pop_index = 0
    for irrep in ordered_sym_labels:
        n_frozen_cores = frozen_core_per_irrep[irrep] if irrep in frozen_core_per_irrep else 0
        n_sfos_frag1 = frags_sfo_irrep_sums[0][irrep]
        n_sfos_frag2 = frags_sfo_irrep_sums[1][irrep]
        
        start_irrep_index = raw_gross_pop_index + n_frozen_cores
        
        if frag_index == 1:
            end_irrep_index = start_irrep_index + n_sfos_frag1
        else:
            start_irrep_index += n_sfos_frag1
            end_irrep_index = start_irrep_index + n_sfos_frag2
            
        gross_pop_active_sfos[irrep] = raw_gross_pop_all_sfos[start_irrep_index: end_irrep_index]
        
        raw_gross_pop_index += sum(frags_sfo_irrep_sums[frag_i][irrep] for frag_i in [0, 1]) + n_frozen_cores
             
    return gross_pop_active_sfos

def main():
    import pathlib as pl
    
    current_dir = pl.Path(__file__).parent
    rkf_dir = current_dir.parent.parent / "test" / "fixtures" / "rkfs"
    rkf_file = "restricted_largecore_differentfragsym_c4v_full.adf.rkf"
    kf_file = KFFile(str(rkf_dir / rkf_file))
    
    print(get_number_sfos_per_irrep_per_frag(kf_file, frag_index=2))
    print(uses_symmetry(kf_file))
    grospop = get_gross_populations(kf_file, frag_index=2)
    print(grospop)

if __name__ == "__main__":
    main()