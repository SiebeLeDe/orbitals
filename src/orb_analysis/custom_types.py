from enum import StrEnum, auto
from typing import Annotated, Literal, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1D = Annotated[npt.NDArray[DType], Literal[1]]
Array2D = Annotated[npt.NDArray[DType], Literal[2]]
Array3D = Annotated[npt.NDArray[DType], Literal[3]]

# Format: {irrep: [data]} with spin being either "A" or "B" (see `SpinTypes`)
RestrictedProperty = TypeAlias = dict[str, Array1D[np.float64]]
# Format: {property: {irrep: [data]}} with property being either "orb_energies" or "occupations"
RestrictedPropertyDict = TypeAlias = dict[str, dict[str, Array1D[np.float64]]]

# Format: {spin: {irrep: [data]}} with spin being either "A" or "B" (see `SpinTypes`)
UnrestrictedProperty = TypeAlias = dict[str, dict[str, Array1D[np.float64]]]
# Format: {property: {spin: {irrep: [data]}}} with property being either "orb_energies" or "occupations" and spin being either "A" or "B" (see `SpinTypes`)
UnrestrictedPropertyDict = TypeAlias = dict[str, dict[str, dict[str, Array1D[np.float64]]]]


class SpinTypes(StrEnum):
    A = "A"
    B = "B"


class SFOInteractionTypes(StrEnum):
    """Enum class for the different types of SFO interactions"""

    HOMO_HOMO = auto()
    HOMO_LUMO = auto()
    LUMO_HOMO = auto()
