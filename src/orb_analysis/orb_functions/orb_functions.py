

from orb_analysis.orbital.orbital import Orbital
from typing import TypeVar

T = TypeVar('T', bound=Orbital)


def filter_orbitals(orbitals: list[T], max_occupied_orbitals: int, max_unoccupied_orbitals: int, irreps: list[str]) -> list[T]:
    """ Filters the orbitals based on the number of occupied and unoccupied orbitals and the specified irreps."""
    orbitals.sort(key=lambda orbital: orbital.energy)
    homo_index = next((index for index, orbital in enumerate(orbitals) if orbital.occupation == 0.0), -100)
    filtered_orbitals: list[T] = []

    occupied_orbitals = orbitals[:homo_index][::-1]  # reverse because we go down from HOMO -> HOMO-1 -> ...
    unoccupied_orbitals = orbitals[homo_index:]  # Here we go up from LUMO -> LUMO+1 -> LUMO+2 -> ...

    # Note that this "for loop" makes sure we don't get "out of array bouds" errors
    handle_occupied_orbitals = True
    for partial_orbitals in [occupied_orbitals, unoccupied_orbitals]:
        counter = 0
        for index, orbital in enumerate(partial_orbitals):

            if orbital.irrep not in irreps:
                continue

            # Stop with the occupied loop when the number of orbitals is equal to the desired number of orbitals
            # This resets the counter and enables the unoccupied loop
            if (counter >= max_occupied_orbitals and handle_occupied_orbitals):
                handle_occupied_orbitals = False
                counter = 0
                break

            if (counter >= max_unoccupied_orbitals and not handle_occupied_orbitals):
                break

            orbital.homo_lumo_index = index

            filtered_orbitals.append(orbital)
            counter += 1

    return sorted(filtered_orbitals, key=lambda orb: orb.energy, reverse=True)
