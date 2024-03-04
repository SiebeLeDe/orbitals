"""
Module containing functions for extracting SFO data from the rkf files of fragment analysis calculations.
The following terms can be extracted form the rkf files:
- Overlap (in a.u.)
- Orbital Energy (in eV)
- Occupation (in a.u.)
- Gross Populations (in a.u.)

BEFORE READING FURTHER: please read the following section about the format of the rkf files:
https://www.scm.com/doc/ADF/Appendices/TAPE21.html

Important sections together with associated variables are (format: ("section", "variable")):
- "Symmetry", "ncbs"   = Number of frozen cores per irrep of the complex calculation (could not find a better alternative for Fragment calculations)
- "SFOs", "fragment"   = Fragment index of the ACTIVE* SFOs
- "SFOs", "subspecies" = Symmetry labels of each ACTIVE SFO (e.g. 1: "A1", 2: "A1", 3: "A2", 4: "A2", ...)
- "SFOs", "occupation" = Occupations of the ACTIVE SFOs
- "SFOs", "escale"     = Orbital energies of ACTIVE SFOs in Ha (relativistic effects are taken into account)
- "SFOs", "energy"     = Orbital energies of ACTIVE SFOs in Ha (relativistic effects are NOT taken into account)
- "SFO popul", "sfo_grosspop" = Gross populations of the SFOs (FROZEN SFOS INCLUDED!)

that can be viewed in the KF Browser of AMS (open a "adf.rkf" file and press "ctrl + E" on Windows or "cmd + E" on Mac).
* Active SFOs are SFOs that are not frozen cores.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scm.plams import KFFile

from orb_analysis.custom_types import Array1D, SpinTypes, UnrestrictedPropertyDict

# --------------------Helper Function(s)-------------------- #


def split_1d_array_into_dict_sorted_by_irreps(
    data_array: Array1D,
    irreps: Sequence[str],
) -> dict[str, Array1D]:
    """
    Splits a 1D array into a dictionary of arrays based on the irreps. The irreps are the keys of the dictionary.
    data_array and irreps must have the same length.
    """
    n_entries_per_irrep = [irreps.count(irrep) for irrep in set(irreps)]
    data_ordered_by_irrep = {irrep: np.zeros(n_entries) for irrep, n_entries in zip(set(irreps), n_entries_per_irrep)}

    # Now we loop over the irreps and add the data to the right array
    irrep_counter = {irrep: 0 for irrep in set(irreps)}
    for data_entry, irrep in zip(data_array, irreps):
        data_ordered_by_irrep[irrep][irrep_counter[irrep]] = data_entry
        irrep_counter[irrep] += 1
    return data_ordered_by_irrep


# -------------------Low-level KF reading -------------------- #


def get_frag_name(kf_file: KFFile, frag_index: int) -> str:
    """Returns the name of the fragment."""
    frag_name_per_sfo = kf_file.read("SFOs", "fragtype").split()  # type: ignore
    frag_names = list(dict.fromkeys(frag_name_per_sfo))
    return frag_names[frag_index - 1]


def uses_symmetry(kf_file: KFFile) -> bool:
    """Returns True if the complex calculation uses symmetry for its MOs and other parts such as gross populations and overlap."""
    grouplabel: str = kf_file.read("Symmetry", "grouplabel").split()[0]  # type: ignore
    return grouplabel.lower() != "nosym"


def get_total_number_sfos(kf_file: KFFile) -> int:
    """Returns the total number of *active* SFOs (frozen core SFOs excluded), which is the sum of the SFOs of both fragments."""
    n_active_sfos = int(kf_file.read("SFOs", "number"))  # type: ignore
    return n_active_sfos


def get_sfo_indices_of_one_frag(kf_file: KFFile, frag_index: int) -> Sequence[int]:
    """Returns the indices of *active* SFOs belonging to one fragment."""
    sfo_frag_indices = list(kf_file.read("SFOs", "fragment", return_as_list=True))  # type: ignore
    sfo_frag_indices = [i for i, sfo_frag_index in enumerate(sfo_frag_indices) if sfo_frag_index == frag_index]
    return sfo_frag_indices


def get_irrep_each_sfo_one_frag(kf_file: KFFile, frag_index: int) -> Sequence[str]:
    sfo_indices_one_frag = get_sfo_indices_of_one_frag(kf_file, frag_index=frag_index)
    frag_symlabels_each_sfo = list(kf_file.read("SFOs", "subspecies").split())  # type: ignore
    frag_symlabels_each_sfo = [frag_symlabels_each_sfo[i] for i in sfo_indices_one_frag]
    return frag_symlabels_each_sfo


def get_ordered_irreps_of_one_frag(kf_file: KFFile, frag_index: int) -> list[str]:
    """Returns the ordered irreps of *active* SFOs (frozen core SFOs excluded) belonging to one fragment."""
    sfo_frag_indices = get_sfo_indices_of_one_frag(kf_file, frag_index=frag_index)
    all_sfo_irreps: list[str] = kf_file.read("SFOs", "subspecies", return_as_list=True).split()  # type: ignore
    sfo_irreps_of_one_frag = list(dict.fromkeys([all_sfo_irreps[i] for i in sfo_frag_indices]))
    return sfo_irreps_of_one_frag


def get_number_sfos_per_irrep_per_frag(kf_file: KFFile, frag_index: int) -> dict[str, int]:
    """Returns the number of *active* SFOs of each irrep (frozen core SFOs excluded) belonging to one fragment."""
    sfo_irreps = get_irrep_each_sfo_one_frag(kf_file, frag_index=frag_index)
    sfo_irrep_sum = {irrep: sfo_irreps.count(irrep) for irrep in set(sfo_irreps)}
    return sfo_irrep_sum


# --------------------Frozen Core Handling-------------------- #


def get_frozen_cores_per_irrep(kf_file: KFFile, frag_index: int) -> dict[str, int]:
    """
    Reads the number of frozen cores per irrep from the KFFile.

    The number of frozen cores per irrep is important for getting gross populations and overlap analysis.
    Basically, the SFO index shown in AMSLevels is different than the index shown in the overlap and population analysis because they can be shifted by frozen cores.

    Moreover, if the complex calculation uses symmetry, but the fragments themselves do not, then the "A" irrep is added, being the sum of all frozen cores.

    In case there is no frozen core and no symmetry, but the fragments use symmetry, then the frozen core is 0 for all irreps that are present in the fragments.
    """
    ordered_frag_irreps = get_ordered_irreps_of_one_frag(kf_file, frag_index=frag_index)
    n_core_orbs_per_irrep: list[int] = kf_file.read("Symmetry", "ncbs", return_as_list=True)  # type: ignore since n_core_orbs is a list of ints

    frozen_core_per_irrep = {irrep: 0 for irrep in ordered_frag_irreps}
    for irrep, n_core_orbs in zip(ordered_frag_irreps, n_core_orbs_per_irrep):
        frozen_core_per_irrep[irrep] = n_core_orbs

    # Add the "A" irrep to the dictionary for the case when symmetry is not used (e.g. NoSym), but the fragments themselves use symmetry.
    # This is only used for the overlap analysis.
    if not uses_symmetry(kf_file):
        frozen_core_per_irrep["A"] = sum(n_core_orbs_per_irrep)
    return frozen_core_per_irrep


# --------------------Restricted Property Function(s)-------------------- #


def _get_orbital_energy_variable(kf_file: KFFile) -> str:
    """
    Determines which key to use for reading the orbital energies from the KFFile.
    The preference is in the order:
        - "site-energies" (needs to be specified in the ADF input file)
        - "escale" (when relativistic effects are present)
        - "energy" (otherwise)
    """
    if ("SFOs", "site_energy") in kf_file:
        return "site_energy"
    if ("SFOs", "escale") in kf_file:
        return "escale"
    return "energy"


def get_orbital_energies(kf_file: KFFile, spin: str = SpinTypes.A) -> Array1D[np.float64]:
    """Reads the orbital energies from the KFFile."""
    # escale refers energies scaled by relativistic effects (ZORA). If no relativistic effects are present, "energy" is the appropriate key.
    variable = _get_orbital_energy_variable(kf_file)
    # print(f"Using {variable} for orbital energies")

    # It is either "escale" or "escale_B", apparently there is no "escale_A" key (same for "energy")
    if spin == SpinTypes.B and ("SFOs", f"{variable}_{SpinTypes.B}") in kf_file:
        variable = f"{variable}_{SpinTypes.B}"

    # Reads the orbital energies for both fragments and selects the data for the current fragment
    orb_energies = np.array(kf_file.read("SFOs", variable))  # type: ignore

    return orb_energies


def get_occupations(kf_file: KFFile, spin: str = SpinTypes.A) -> Array1D[np.float64]:
    """Reads the occupations from the KFFile."""
    # It is either "occupation" or "occupation_B", apparently there is no "occupation_A" key
    occupation_key = f"occupation_{SpinTypes.B}" if spin == SpinTypes.B and ("SFOs", f"occupation_{SpinTypes.B}") in kf_file else "occupation"
    occupations = np.array(kf_file.read("SFOs", occupation_key))  # type: ignore

    return occupations


# --------------------Unrestricted Property Function(s)-------------------- #


# --------------------Property to Function Mapping-------------------- #


# Format: {property: (callable function for reading property, section in KFFile, variable in KFFile)}
RESTRICTED_KEY_FUNC_MAPPING: dict[str, Callable] = {
    "orb_energies": get_orbital_energies,
    "occupations": get_occupations,
}

# --------------------Interface Function(s)-------------------- #


def get_fragment_properties(kf_file: KFFile, frag_index: int) -> UnrestrictedPropertyDict:
    """
    Returns a dictionary of dictionaries with the properties of the fragments.

    The properties are:
        - Orbital Energies
        - Occupations

    Output format:
    {
        spin ("A"/"B"): {
            property ("orb_energies" / occupations): {
                irrep (e.g., "A1", "B2", "E1:1"): [data]
    }

    Note: this will produce double the amount of data when restricted fragments are used because the spin key is not needed for restricted fragments.
    Currently, the "B" spin is discarded for restricted calcs in the `create_fragment_data` function.
    """
    sfo_indices_of_one_frag = get_sfo_indices_of_one_frag(kf_file, frag_index)
    frag_irreps_each_sfo = get_irrep_each_sfo_one_frag(kf_file, frag_index)

    data_dic_to_be_unpacked = {property: {str(spin): {} for spin in SpinTypes} for property in RESTRICTED_KEY_FUNC_MAPPING}

    for property, func in RESTRICTED_KEY_FUNC_MAPPING.items():
        for spin in SpinTypes:
            data = func(kf_file, spin=spin)

            data = np.array([data[i] for i in sfo_indices_of_one_frag])

            # Now we turn one long array into a dictionary of arrays sorted by irreps (e.g. [.....] -> {"A1": [.....], "A2": [.....]})
            data_dic_to_be_unpacked[property][spin] = split_1d_array_into_dict_sorted_by_irreps(data_array=data, irreps=frag_irreps_each_sfo)

    return data_dic_to_be_unpacked


def get_gross_populations(kf_file: KFFile, frag_index: int = 1) -> dict[str, dict[str, Array1D[np.float64]]]:
    """
    Reads the gross populations from the KFFile by taking into account the frozen cores.
    Annoyingly, the "SFOs" sections contains the SFOs of both fragments that ALREADY HAVE BEEN FILTERED for the frozen cores.
    For example, the SFOs number may be 114, but the gross population array may have 148 entries. This is because the first 34 entries are the frozen cores.

    Structure of the ("SFOs popul","sfo_grosspop") section for a restricted calculation with c3v symmetry:
    [n Frozen Cores A1, Active SFOs Frag1 A1, Active SFOs Frag2 A1, n Frozen Cores A2, Active SFOs Frag1 A2, Active SFOs Frag2 A2, ...]

    Therefore, the sum of `sfo_indices_of_one_frag` and `n_frozen_cores_per_irrep` is used to get the correct indices for SFOs on both fragments and all irreps.

    Output format:
    {
        spin ("A"/"B"): {
            irrep (e.g., "A1", "B2", "E1:1"): [data]
    }
    """
    frags_sfo_irrep_sums = [get_number_sfos_per_irrep_per_frag(kf_file, frag_index=frag_index) for frag_index in [1, 2]]
    ordered_irreps = get_ordered_irreps_of_one_frag(kf_file, frag_index=frag_index)
    frozen_core_per_irrep = get_frozen_cores_per_irrep(kf_file, frag_index=frag_index)
    complex_has_symmetry = uses_symmetry(kf_file)  # refers to the complex calculation
    frag_has_symmetry = len(ordered_irreps) > 1  # refers to the fragments

    raw_gross_pop_all_sfos = np.array(kf_file.read("SFO popul", "sfo_grosspop"))

    # Table writing (comment out if not needed)

    # header = [
    #     "Frag1 Irrep",
    #     "Frag1 # SFOs",
    #     "Frag2 Irrep",
    #     "Frag2 # SFOs",
    # ]

    # frag1_sfos = [[irrep, frags_sfo_irrep_sums[0][irrep]] for irrep in get_ordered_irreps_of_one_frag(kf_file, frag_index=1)]
    # frag2_sfos = [[irrep, frags_sfo_irrep_sums[1][irrep]] for irrep in get_ordered_irreps_of_one_frag(kf_file, frag_index=2)]

    # table_data = []

    # for (irrep1, n_sfos1), (irrep2, n_sfos2) in zip_longest(frag1_sfos, frag2_sfos, fillvalue=("", "")):
    #     table_data.append([irrep1, n_sfos1, irrep2, n_sfos2])

    # print(tabulate(table_data, headers=header, tablefmt="pipe"))

    # header = [
    #     "Irrep",
    #     "# Frozen cores",
    # ]
    # frozen_cores = [[irrep, n_frozen_cores] for irrep, n_frozen_cores in frozen_core_per_irrep.items()]

    # print(tabulate(frozen_cores, headers=header, tablefmt="pipe"))
    # print(get_ordered_irreps_of_one_frag(kf_file, frag_index=1))
    # print(get_ordered_irreps_of_one_frag(kf_file, frag_index=2))

    # Table writing (comment out if not needed)

    if not complex_has_symmetry and not frag_has_symmetry:  # no symmetry for both the complex nor the fragments
        start_index = sum(frozen_core_per_irrep.values())
        total_sfo_sum_frag1 = sum(frags_sfo_irrep_sums[0][irrep] for irrep in frags_sfo_irrep_sums[0])
        total_sfo_sum_frag2 = sum(frags_sfo_irrep_sums[1][irrep] for irrep in frags_sfo_irrep_sums[1])
        total_sfo_for_one_spin = total_sfo_sum_frag1 + total_sfo_sum_frag2 + start_index

        if frag_index == 1:
            return {
                SpinTypes.A: {"A": raw_gross_pop_all_sfos[start_index : start_index + total_sfo_sum_frag1]},  # NOQA: E203
                SpinTypes.B: {"A": raw_gross_pop_all_sfos[total_sfo_for_one_spin : total_sfo_for_one_spin + total_sfo_sum_frag1]},  # NOQA: E203
            }

        return {
            SpinTypes.A: {"A": raw_gross_pop_all_sfos[start_index + total_sfo_sum_frag1 : start_index + total_sfo_sum_frag1 + total_sfo_sum_frag2]},  # NOQA: E203
            SpinTypes.B: {"A": raw_gross_pop_all_sfos[total_sfo_for_one_spin + total_sfo_sum_frag1 : total_sfo_for_one_spin + total_sfo_sum_frag1 + total_sfo_sum_frag2]},  # NOQA: E203
        }

    gross_pop_active_sfos = {str(spin): {irrep: np.zeros_like(frags_sfo_irrep_sums[frag_index - 1][irrep], dtype=np.float64) for irrep in frags_sfo_irrep_sums[frag_index - 1]} for spin in SpinTypes}

    # only works if frag1 and frag2 have the same irreps and thus belong to the same point group
    for spin in SpinTypes:
        raw_gross_pop_index = 0 if spin == SpinTypes.A else get_total_number_sfos(kf_file) + sum(frozen_core_per_irrep.values())
        for irrep in ordered_irreps:
            n_frozen_cores = frozen_core_per_irrep.get(irrep, 0)
            n_sfos_frag1 = frags_sfo_irrep_sums[0][irrep]
            n_sfos_frag2 = frags_sfo_irrep_sums[1][irrep]
            start_irrep_index = raw_gross_pop_index + n_frozen_cores

            if frag_index == 1:
                end_irrep_index = start_irrep_index + n_sfos_frag1
            else:
                start_irrep_index += n_sfos_frag1
                end_irrep_index = start_irrep_index + n_sfos_frag2

            gross_pop_active_sfos[spin][irrep] = raw_gross_pop_all_sfos[start_irrep_index:end_irrep_index]

            raw_gross_pop_index += sum(frags_sfo_irrep_sums[frag_i][irrep] for frag_i in [0, 1]) + n_frozen_cores

    return gross_pop_active_sfos


def main():
    import pathlib as pl

    current_dir = pl.Path(__file__).parent
    rkf_dir = current_dir.parent.parent.parent / "test" / "fixtures" / "rkfs"
    # rkf_file = 'restricted_largecore_differentfragsym_c4v_full.adf.rkf'
    # rkf_file = "restricted_largecore_differentfragsym_c4v_full.adf.rkf"
    rkf_file = "unrestricted_largecore_fragsym_c3v_full.adf.rkf"
    kf_file = KFFile(str(rkf_dir / rkf_file))

    # print(get_orbital_energies(kf_file))

    # print(get_occupations(kf_file))
    # print(get_number_sfos_per_irrep_per_frag(kf_file, frag_index=1))
    # print(get_frag_name(kf_file, frag_index=2))
    # data = get_fragment_properties(kf_file, frag_index=1)
    # pprint(data)
    grospop = get_gross_populations(kf_file, frag_index=1)
    print(grospop)


if __name__ == "__main__":
    main()
