﻿import attrs

from orb_analysis.orbital.orbital import SFO
from orb_analysis.orbital_manager.shared_functions import calculate_matrix_element


@attrs.define
class OrbitalPair:
    orb1: SFO
    orb2: SFO
    overlap: float

    def __str__(self):
        if self.is_pauli_pair:
            return f"frag1 SFO: {self.orb1.homo_lumo_label:15s} frag2 SFO: {self.orb2.homo_lumo_label:15s} Overlap: {self.overlap:.3f}"

        return f"frag1 SFO: {self.orb1.homo_lumo_label:15s} frag2 SFO: {self.orb2.homo_lumo_label:15s} Overlap: {self.overlap:.3f} Stabilization: {self.stabilization}"

    @property
    def is_pauli_pair(self) -> bool:
        return self.orb1.is_occupied and self.orb2.is_occupied

    @property
    def energy_gap(self) -> float:
        return abs(self.orb1.energy - self.orb2.energy)

    @property
    def stabilization(self) -> float:
        if self.is_pauli_pair:
            return -1.0
        return calculate_matrix_element(self.orb1, self.orb2, self.overlap)
