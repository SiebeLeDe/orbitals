import attrs
from scm.plams import KFFile


@attrs.define
class CalcInfo:
    """
    This class contains information about the orbitals present in the complex calculation
    """

    kf_file: KFFile
    restricted: bool = True
    relativistic: bool = False
    symmetry: bool = False

    def __attrs_post_init__(self):
        # First, get relevant terms such as symmetry group label, unrestricted, relativistic, etc.
        self.symmetry = str(self.kf_file.read("Symmetry", "grouplabel")).split()[0].lower() not in ["nosym"]
        self.restricted = int(self.kf_file.read("General", "nspin")) == 1  # type: ignore returns an integer
        self.relativistic = int(self.kf_file.read("General", "ioprel")) != 0  # type: ignore returns an integer
