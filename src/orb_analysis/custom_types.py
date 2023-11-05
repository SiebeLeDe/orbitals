from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)
Array1D = Annotated[npt.NDArray[DType], Literal[1]]
Array2D = Annotated[npt.NDArray[DType], Literal[2]]
Array3D = Annotated[npt.NDArray[DType], Literal[3]]
