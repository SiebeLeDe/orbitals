import attrs

from orb_analysis.orbital.orbital import SFO
from orb_analysis.orbital_manager.shared_functions import calculate_matrix_element


@attrs.define
class OrbitalPair:
    orb1: SFO
    orb2: SFO
    overlap: float

    def __str__(self):
        return f"frag1 SFO: {self.orb1.homo_lumo_label:15s} frag2 SFO: {self.orb2.homo_lumo_label:15s} Overlap: {self.overlap:.3f}"

    @property
    def energy_gap(self) -> float:
        return abs(self.orb1.energy - self.orb2.energy)

    @property
    def stabilization(self) -> float:
        return calculate_matrix_element(self.orb1, self.orb2, self.overlap)
