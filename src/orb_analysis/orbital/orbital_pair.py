import attrs

from orb_analysis.orbital.orbital import Orbital


@attrs.define
class OrbitalPair:
    orb1: Orbital
    orb2: Orbital
