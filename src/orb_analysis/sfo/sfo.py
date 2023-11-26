"""
Module containing the :SFO: class that stores information about symmetrized fragment orbitals (SFOs).
"""
from __future__ import annotations

from typing import Optional

import attrs


@attrs.define
class SFO:
    """
    This class contains information about a symmetrized fragment orbital (SFO). Initalizing the class requires
    the index, symmetry and spin of the SFO. The index is the order in which the SFOs are stored in the rkf file.

    Also possible is to initialize the class with a label. The correct format of the label is:
    <index>_<irrep or <index>_<irrep>_<spin> if the SFO is from an unrestricted calculation.
    """

    index: int
    symmetry: str
    spin: Optional[str] = None

    @classmethod
    def from_label(cls, label: str):
        """
        Extracts the index, symmetry and spin from the label of the SFO. The correct format of the label is:

        <index>_<irrep>_<spin> or <index>_<irrep> if the SFO is from an unrestricted calculation.
        """
        index, symmetry, *spin = label.split("_")
        spin = spin[0] if spin else None
        return cls(index=int(index), symmetry=symmetry, spin=spin)

    def __eq__(self, __value: str | SFO) -> bool:
        if isinstance(__value, str):
            return self == SFO.from_label(__value)
        else:
            return (
                self.index == __value.index
                and self.symmetry == __value.symmetry
                and self.spin == __value.spin
            )


def main():
    sfo = SFO.from_label("1_A_A")
    label = "1_A_A"
    print(sfo == label)


if __name__ == "__main__":
    main()
