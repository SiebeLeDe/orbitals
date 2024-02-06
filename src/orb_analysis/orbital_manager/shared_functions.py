import numpy as np
from orb_analysis.custom_types import SFOInteractionTypes
from orb_analysis.orbital.orbital import SFO


def filter_sfos_by_interaction_type(frag1_sfos: list[SFO], frag2_sfos: list[SFO], interaction_type: SFOInteractionTypes) -> tuple[list[int], list[int]]:
    """Returns the indices of relevant frag1 and frag 2 SFOs depending on the interaction type"""

    if interaction_type == SFOInteractionTypes.HOMO_HOMO:
        frag1_filtered_indices = [index for index, sfo in enumerate(frag1_sfos) if sfo.is_occupied]
        frag2_filtered_indices = [index for index, sfo in enumerate(frag2_sfos) if sfo.is_occupied]

    elif interaction_type == SFOInteractionTypes.HOMO_LUMO:
        frag1_filtered_indices = [index for index, sfo in enumerate(frag1_sfos) if sfo.is_occupied]
        frag2_filtered_indices = [index for index, sfo in enumerate(frag2_sfos) if not sfo.is_occupied]

    elif interaction_type == SFOInteractionTypes.LUMO_HOMO:
        frag1_filtered_indices = [index for index, sfo in enumerate(frag1_sfos) if not sfo.is_occupied]
        frag2_filtered_indices = [index for index, sfo in enumerate(frag2_sfos) if sfo.is_occupied]

    else:
        frag1_filtered_indices = [index for index in range(len(frag1_sfos))]
        frag2_filtered_indices = [index for index in range(len(frag2_sfos))]

    return frag1_filtered_indices, frag2_filtered_indices


def calculate_matrix_element(sfo1: SFO, sfo2: SFO, overlap: float) -> float:
    """Checks if the interaction is HOMO-HOMO / HOMO-LUMO / LUMO-LUMO and returns the correct value (see parent function docstring"""
    # LUMO-LUMO: non-physical
    if not sfo1.is_occupied and not sfo2.is_occupied:
        return 0.0

    # HOMO-HOMO: Pauli repulsion
    if sfo1.is_occupied and sfo2.is_occupied:
        return overlap**2 * 100

    # HOMO-LUMO / LUMO-HOMO: favorable orbital interactions (SCF process)
    energy_gap: float = abs(sfo1.energy - sfo2.energy) if sfo1.energy > sfo2.energy else abs(sfo2.energy - sfo1.energy)
    # print(f"{sfo1.energy :.2f} {sfo2.energy :.2f} {energy_gap:.2f} {energy_gap2:.2f}")
    if np.isclose(energy_gap, 0):
        return -overlap * 100
    else:
        return (overlap**2 / energy_gap) * 100
