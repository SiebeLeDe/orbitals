"""
Module containing presets for the plams.DensfJob job as part of the SCM / AMS software package
The presets are defined as functions that return a :Settings: dictclass.
This instance should be the argument of a :DensfJob:, e.g., DensfJob(name="[example_name]", settings=[Settings_instance], inputjob = ["path_to_rkf_file" or :SCMJob: or :SCMResults:])

The current presets are IMPLEMENTED:
    - Output: Used for specifying the output file. Can be (a path to) either a cube file or a rkf file (.t41)
    - Orbitals: Used for plotting molecular orbitals (MOs) and fragment orbitals (SFOs).
    - Grid: Specifies grid parameters.
    - Units: Settings the unit for length.
    - Density: Generates the charge density in the grid.
    - KinDens: Generates the Kinetic energy density and electron localization function on the grid.
    - Laplacian: Generates the Laplacian on a grid.
    - DenGrad: Calculates the gradient of the density.
    - DenHess: Calculates the hessian of the density.
    - Potential: Calculates the coulomb and/or exchange-correlation potential in the grid.

The following presets are NOT (YET) IMPLEMENTED:
    - NOCV: Calculates ϵ*ϕ2 values of Natural Orbitals for Chemical Valence (NOCVs).
    - NCI: Calculates the non-covalent interaction (NCI) index.
    - SEDD: Calculats the single exponential decay detector (SEDD), which extracts information about bonding and localization in atoms, molecules, or molecular assemblies.
    - DualDescriptor: Calculates the dual descriptor (DD) in the frontier molecular orbitals approximations.
    - POLTDDFT: Calculate the complex dynamical polarizability with PolTDDFT, which can be used to calculate the photoabsorption and CD spectra.

More info on the presets can be found in the SCM manual: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf
More info about the settings format for DensfJobs can be found here:

Notes on the presets:
    - The presets are functions that return a :Settings: dictclass.
    - Sometimes, a settings instance is also in the argument to change the input settings (e.g. removing "outputfile" if "cubeoutput" is present).

Each returned Settings has the structure:
    - settings.input.[BLOCK]
"""
from typing import Optional, Union
from scm.plams import Settings
from pathlib import Path


def output(
    outputfile: Optional[str] = None,
    cubeoutput: Optional[str] = None
) -> Settings:
    """
    Specifies the output files. These are mutually exclusive and will be considered in the order: outputfile, cubeoutput

    Args:
        outputfile (str): The name of the output file
        cubeoutput (str): The name of the cube file
    """
    set = Settings()

    # Extentention of the outputfile should be .t41
    if outputfile is not None:
        outputfile = str(Path(outputfile).with_suffix(".t41"))

    if cubeoutput is None:
        set.input.OUTPUTFILE = outputfile

    if cubeoutput is not None and outputfile is not None:
        set.input.OUTPUTFILE = outputfile

    if cubeoutput is not None and outputfile is None:
        set.input.CUBOUTPUT = cubeoutput

    return set


def grid(
    save_grid: bool = False,
    grid_type: str = "medium",
) -> Settings:
    """
    Specifies grid parameters using simple parameters

    Args:
        save_grid (bool): Whether to save the grid or not
        grid_type (str): The type of grid to use. Can be "coarse", "medium", or "fine"

    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#grid
    """
    if grid_type not in ["coarse", "medium", "fine"]:
        raise ValueError(f"grid_type should be 'coarse', 'medium', or 'fine'. Got '{grid_type}'")

    set = Settings()
    if save_grid:
        set.input.GRID._h = f"{save_grid} {grid_type}"
    else:
        set.input.GRID._h = grid_type

    return set


def specific_grid(
    save_grid: bool = False,
    origin: list[float] = [0, 0, 0],
    n_points: list[int] = [100, 100, 100],
    direction_vectors: list[list[float]] = [[1, 0, 0, 4.0], [0, 1, 0, 4.0], [0, 0, 1, 4.0]],
) -> Settings:
    """
    Specifies the grid using grid-specific parameters

    Args:
        save_grid (bool): Whether to save the grid or not
        origin (list[float]): The origin of the grid, must be a list of 3 floats (x, y, z)
        n_points (list[int]): The number of points in the grid, can be a list of 1 up to 3 ints (x, y, z)
        direction_vectors (list[list[float]]): The direction vectors of the grid, can be a list of 1 up to 3 lists of 4 floats (x, y, z, length)

    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#grid

    """
    # if len(n_points) != len(direction_vectors):
    #     raise ValueError(
    #         f"n_points and direction_vectors should specify the same dimensions. Got {len(n_points)} and {len(direction_vectors)}"
    #     )

    set = Settings()
    if save_grid:
        set.input.GRID._h = str(save_grid)

    grid_list_for_settings = []
    grid_list_for_settings.append(" ".join([str(entry) for entry in origin]))
    grid_list_for_settings.append(" ".join([str(entry) for entry in n_points]))
    for direction_vector in direction_vectors:
        grid_list_for_settings.append(" ".join([str(entry) for entry in direction_vector]))

    set.input.GRID._1 = grid_list_for_settings
    return set


def units(unit: str = "bohr") -> Settings:
    """
    Specifies the unit for length

    Args:
        unit (str): The unit for length. Can be "bohr" or "angstrom"

    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#units

    """
    if unit not in ["bohr", "angstrom"]:
        raise ValueError(f"unit should be 'bohr' or 'angstrom'. Got '{unit}'")
    set = Settings()
    set.input.UNITS._h = unit
    return set


def density(
    fit: bool = False,
    frag: bool = False,
    ortho: bool = False,
    scf: bool = True,
    trans: bool = False,
    core: bool = False
) -> Settings:
    """
    Generates the charge density in the grid

    Args:
        fit (bool): Whether to use the fit functions (approximation of the exact density)
        frag (bool): Whether to use the sum-of-fragments (i.e. the initial) density
        ortho (bool): Whether to use the orthogonalized fragments (orthogonalization to account for the Pauli repulsion)
        scf (bool): Whether to use the final result of the adf calculation
        trans (bool): Whether to use the transition density (product of initial and final states of an excitation)
        core (bool): Whether to calculate the frozen core density. Note: this is only possible if the frozen core approximation was used in the adf calculation

    Defaults to scf=True, which is the final result of the adf calculation.

    The `core` option takes precedence over the other options. If `core` is True, the other options are ignored as they are mutually exclusive.
    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#density
    """
    set = Settings()

    if core:
        set.input.DENSITY = "core"
        return set

    keys = ["fit", "frag", "ortho", "scf", "trans"]
    header_str = ""

    for key in keys:
        if eval(key):
            header_str += f" {key}"

    set.input.DENSITY = header_str

    return set


def kinetic_energy_density(
    frag: bool = False,
    orth: bool = False,
    scf: bool = True,
) -> Settings:
    """
    Generates the Kinetic energy density and electron localization function on the grid.

    Args:
        frag (bool): Whether to use the sum-of-fragments (i.e. the initial) density
        orth (bool): Whether to use the orthogonalized fragments (orthogonalization to account for the Pauli repulsion)
        scf (bool): Whether to use the final result of the adf calculation

    Defaults to scf=True, which is the final result of the adf calculation.

    More info:https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#kinetic-energy-density-and-electron-localization-function-elf
    """
    set = Settings()

    keys = ["frag", "orth", "scf"]
    header_str = ""

    for key in keys:
        if eval(key):
            header_str += f" {key}"

    set.input.KinDens = header_str

    return set


def laplacian_density(
    fit: bool = False
) -> Settings:
    """
    Generates the Laplacian on a grid.

    Args:
        fit (bool): Whether to use the fit functions (approximation of the exact density)

    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#laplacian-of-the-density
    """

    set = Settings()

    set.input.Laplacian = "fit" if fit else ""

    return set


def gradient_density(
    fit: bool = False,
    core: bool = False
) -> Settings:
    """
    Generates the Laplacian on a grid.

    Args:
        fit (bool): Whether to use the fit functions (approximation of the exact density)
        core (bool): Whether to calculate the frozen core density. Note: this is only possible if the frozen core approximation was used in the adf calculation

    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#gradient-of-the-density
    """

    set = Settings()

    if core:
        set.input.DenGrad = "core"
    elif fit:
        set.input.DenGrad = "fit"
    else:
        set.input.DenGrad = ""

    return set


def hessian_density(
    fit: bool = False,
    core: bool = False
) -> Settings:
    """
    Generates the Hessian on a grid.

    Args:
        fit (bool): Whether to use the fit functions (approximation of the exact density)
        core (bool): Whether to calculate the frozen core density. Note: this is only possible if the frozen core approximation was used in the adf calculation

    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#hessian-of-the-density
    """

    set = Settings()

    if core:
        set.input.DenHess = "core"
    elif fit:
        set.input.DenHess = "fit"
    else:
        set.input.DenHess = ""

    return set


def potential(
    potential_type: str = "xc",
    frag: bool = False,
    ortho: bool = False,
    scf: bool = True,
) -> Settings:
    """
    Calculates the coulomb and/or exchange-correlation potential in the grid.

    Args:
        potential_type (str): The type of potential to calculate. Can be "XC" or "Coulomb". Defaults to "XC"
        frag (bool): Whether to use the sum-of-fragments (i.e. the initial) density
        ortho (bool): Whether to use the orthogonalized fragments (orthogonalization to account for the Pauli repulsion)
        scf (bool): Whether to use the final result of the adf calculation

    Defaults to scf=True and potential_type="xc".

    More info: https://www.scm.com/doc/ADF/Input/Densf.html?highlight=Densf#potential
    """
    if potential_type.lower() not in ["xc", "coul"]:
        raise ValueError(f"potential_type should be 'xc' or 'coul'. Got '{potential_type}'")

    set = Settings()

    header_str = potential_type.lower()

    keys = ["frag", "ortho", "scf"]
    for key in keys:
        if eval(key):
            header_str += f" {key}"

    set.input.potential = header_str

    return set


def _base_orbital(
    grad: bool = False,
    type: str = "SCF",
) -> Settings:
    """
    Common options for orbital and localized_orbital

    Args:
        grad (bool): Calculates the MO gradient. Defaults to False
        type (str): Use MOs (SCF) or fragment orbitals (SFOs), or localied orbital (LOC)
    """
    set = Settings()

    allowed_types = ["SCF", "SFO", "LOC"]
    if type.upper() not in allowed_types:
        raise ValueError(f"Orbital type not {', '.join(allowed_types)}. Got '{type}'")

    header_str = ""

    if grad:
        header_str += "GRAD"
    header_str += type
    set.input.ORBITALS._h = header_str
    return set


def orbital(
    grad: bool = False,
    type: str = "SCF",
    spin: Optional[str] = None,
    irrep_number_label: Optional[tuple[str, list[int]]] = None,
    irrep_homo_lumo: Optional[tuple[str, int, int]] = None,
    irrep_occ_virt: Optional[tuple[str, str]] = None,
    all_option: Optional[Union[str, tuple[int, int]]] = None
) -> Settings:
    """
    Calculates Molecular Orbitals (SCF) or Symmetry Fragment Orbital (SFO)
    Specify either irrep_number_label, or irrep_homo_lumo, or irrep_occ_virt

    Args:
        grad (bool): Calculates the MO gradient. Defaults to False
        type (str): Use MOs (SCF) or fragment orbitals (SFOs)
        spin (bool): Whether to use alpha or beta spin orbitals. Defaults to False (alpha / restricted calculations)
        irrep_number (str): Optionally specify orbitals via [irrep][index], such as A1, or E1:2
        homo_limo_index (str, int, int): Optionally specify the range from HOMO-[x] to LUMO+[x], e.g., (A1, 2, 2) -> HOMO-2 up to LUMO+2
        irrep_occ_virt (str, str): Optionally specify the irrep and if these are occupied ("occ") or virtual ("virt") orbitals, e.g. (A1, virt)

    Defaults to 'All HOMO-1 LUMO+1'
    """
    set = Settings()

    # Check that only one of irrep_number_label, irrep_homo_lumo, or irrep_occ_virt, or spin is specified
    count = sum(x is not None for x in [irrep_number_label, irrep_homo_lumo, irrep_occ_virt, spin, all_option])
    if count != 1:
        print("Not detecting any specific input. Defaulting to 'All HOMO-1 LUMO+1'")

    if all_option is not None:
        if isinstance(all_option, str) and str(all_option).lower() not in ["virt", "occ"]:
            raise ValueError(f"Please use 'virt' or 'occ' as label for all_option, not {all_option[1]}")

    set.update(_base_orbital(grad, type))

    # Deal with alpha / beta if the calculation is unrestricted
    if spin is not None:
        set.input.ORBITALS[spin.lower()] = ""

    elif irrep_number_label is not None:
        irrep, orb_indices = irrep_number_label
        orb_indices = [str(index) for index in orb_indices]
        set.input.ORBITALS[irrep] = f"{', '.join(orb_indices)}"

    elif irrep_homo_lumo is not None:
        irrep, homo_i, lumo_i = irrep_homo_lumo
        set.input.ORBITALS[irrep] = f"HOMO-{homo_i}, LUMO+{lumo_i}"

    elif irrep_occ_virt is not None:
        irrep, occ_virt = irrep_occ_virt
        set.input.ORBITALS._h = f"{irrep}={occ_virt.upper()}"

    elif all_option is not None:
        if isinstance(all_option, str):
            set.input.ORBITALS.all = all_option  # virt or occ
        else:
            homo_i, lumo_i = all_option[0], all_option[1]
            set.input.ORBITALS.all = f"HOMO-{homo_i}, LUMO+{lumo_i}"

    # Default option
    else:
        set.input.ORBITALS.all = "HOMO-1 LUMO+1"

    return set


def localized_orbital(
    grad: bool = False,
    spin: bool = False,
    all_label: bool = False,
    indices: Optional[str] = None
) -> Settings:
    """
    Returns a Settings object with settings for plotting localized orbitals.

    Args:
        grad (bool): Whether to plot the gradient of the orbitals.
        spin (bool): Whether to plot the spin of the orbitals.
        all_label (bool): Whether to label all orbitals.
        indices (Optional[str]): A string of indices to plot.

    Defaults to no additional settings returned
    """
    set = Settings()

    # Check that only one of irrep_number_label, irrep_homo_lumo, or irrep_occ_virt is specified
    count = sum(x is not None for x in [spin, all_label, indices])
    if count != 1:
        raise ValueError("Specify exactly one of irrep_number_label, irrep_homo_lumo, or irrep_occ_virt")

    set.update(_base_orbital(grad, "LOC"))

    if all_label is not None:
        set.input.ORBITALS.all = ""
    elif indices is not None:
        set.input.ORBITALS._1 = ", ".join([str(index) for index in indices])

    return set
