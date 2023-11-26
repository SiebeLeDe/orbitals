from enum import StrEnum
from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1D = Annotated[npt.NDArray[DType], Literal[1]]
Array2D = Annotated[npt.NDArray[DType], Literal[2]]
Array3D = Annotated[npt.NDArray[DType], Literal[3]]

# Format: {property: {irrep: [data]}} with property being either "orb_energies" or "occupations"
RestrictedProperty = TypeAlias = dict[str, Array1D[np.float64]]
RestrictedPropertyDict = TypeAlias = dict[str, dict[str, Array1D[np.float64]]]

# Format: {spin: {property: {irrep: [data]}}} with property being either "orb_energies" or "occupations" and spin being either "A" or "B" (see `SpinTypes`)
UnrestrictedProperty = TypeAlias = dict[str, dict[str, Array1D[np.float64]]]
UnrestrictedPropertyDict = TypeAlias = dict[str, dict[str, dict[str, Array1D[np.float64]]]]


class SpinTypes(StrEnum):
    A = "A"
    B = "B"
