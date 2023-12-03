"""
Module containing the :Orbital: class that acts as a basis for symmetrized fragment orbitals (SFOs) and molecular orbitals (MOs).
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import attrs

from orb_analysis.custom_types import SpinTypes

OCCUPATION_TO_LABEL: dict[float, str] = {
    0.0: "LUMO",
    1.0: "SOMO",
    2.0: "HOMO"
}


@attrs.define
class Orbital(ABC):
    """
    Abstract class that contains information about an orbital. This class should not be instantiated directly.
    """
    index: int
    irrep: str
    spin: str
    energy: float = 1000.0
    occupation: float = -1.0
    homo_lumo_index: int = 1000  # Displays either HOMO-[x] or LUMO+[x]

#  ------------------------------------------------------------------
# ---------------------- Shared Classmethods ------------------------
#  ------------------------------------------------------------------

    @classmethod
    def from_label(cls, label: str):
        """
        Extracts the index, irrep and spin from the label of the SFO. The correct format of the label is:

        <index>_<irrep>_<spin> or <index>_<irrep> if the SFO is from an unrestricted calculation.
        """
        index, irrep, *spin = label.split("_")
        spin = spin[0] if spin else SpinTypes.A
        return cls(index=int(index), irrep=irrep, spin=spin)

#  ------------------------------------------------------------------
# ------------------ Shared Property Methods ------------------------
#  ------------------------------------------------------------------

    @property
    def is_occupied(self) -> bool:
        return self.occupation >= 1e-6

    @property
    def homo_lumo_label(self) -> str:
        """ Returns the label in the format HOMO(-x), SOMO(-x) / SOMO(+x), or LUMO(+x) """
        ret_str = OCCUPATION_TO_LABEL[round(self.occupation, 1)]

        if self.is_occupied:
            ret_str = f"{ret_str}-{self.homo_lumo_index}" if self.homo_lumo_index != 0 else ret_str
        else:
            ret_str = f"{ret_str}+{self.homo_lumo_index}" if self.homo_lumo_index != 0 else ret_str
        return ret_str

    @property
    @abstractmethod
    def amsview_label(self) -> str:
        pass
#  ------------------------------------------------------------------
# ------------------------ Magic Methods ----------------------------
#  ------------------------------------------------------------------

    @abstractmethod
    def __eq__(self, __value: str | SFO) -> bool:
        pass


@attrs.define
class SFO(Orbital):
    """
    This class contains information about a symmetrized fragment orbital (SFO). Initalizing the class requires
    the index, irrep and spin of the SFO. The index is the order in which the SFOs are stored in the rkf file.

    Also possible is to initialize the class with a label. The correct format of the label is:
    <index>_<irrep or <index>_<irrep>_<spin> if the SFO is from an unrestricted calculation.
    """
    gross_pop: float = 1000.0

    def __eq__(self, __value: str | SFO) -> bool:
        if isinstance(__value, str):
            return self == SFO.from_label(__value)
        else:
            return (
                self.index == __value.index
                and self.irrep == __value.irrep
                and self.spin == __value.spin
            )

    @property
    def amsview_label(self) -> str:
        """ Returns the orbital label that can be used for AMSView plotting """
        return f"SFO_{self.irrep}_{self.index}"


class MO(Orbital):
    """
    This class contains information about a molecular orbital (MO). Initalizing the class requires
    the index, irrep and spin of the MO. The index is the order in which the MOs are stored in the rkf file.

    Also possible is to initialize the class with a label. The correct format of the label is:
    <index>_<irrep or <index>_<irrep>_<spin> if the MO is from an unrestricted calculation.
    """

    def __eq__(self, __value: str | SFO) -> bool:
        if isinstance(__value, str):
            return self == SFO.from_label(__value)
        else:
            return (
                self.index == __value.index
                and self.irrep == __value.irrep
                and self.spin == __value.spin
            )

    @property
    def amsview_label(self) -> str:
        """ Returns the orbital label that can be used for AMSView plotting """
        return f"SCF_{self.irrep}_{self.index}"


def main():
    sfo = SFO.from_label("1_A_A")
    # mo = MO.from_label("1_A_A")
    label = "1_A_A"
    print(sfo == label)


if __name__ == "__main__":
    main()
