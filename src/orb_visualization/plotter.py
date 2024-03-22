import pathlib as pl
import shutil
import subprocess
from typing import Sequence, TypeVar

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from attrs import asdict, define
from orb_analysis.orbital.orbital import SFO, Orbital


@define
class PlotSettings:
    """General plot settings used for amsreport"""

    bgcolor: str = "#FFFFFF"
    scmgeometry: str = "1920x1080"
    zoom: float = 2.0
    antialias: bool = True
    viewplane: str = "{0 0 1}"
    print_command: bool = True


@define
class AMSViewPlotSettings(PlotSettings):
    """Plot settings for amsview. Inherits from PlotSettings"""

    wireframe: bool = False
    transparent: bool = True
    viewplane: str = "0 0 1"  # Viewplane normal to the specified x,y,z direction
    grid: str = "Medium"
    hide_view: bool = True
    colorfield: str = "0 65"  # No idea what this value should be
    printrange: bool = True
    camera: int = -1  # Camera load-outs from AMS
    val: float = 0.03  # isovalue
    # ciso: bool = False


@define
class AMSReportPlotSettings(PlotSettings):
    """Plot settings for amsreport. Inherits from PlotSettings"""

    grid: str = "Medium"  # Grid size (Coarse, Medium, Fine)


PlotSettingsType = TypeVar("PlotSettingsType", bound=PlotSettings)


def plot_orbital_with_amsview(
    input_file: str | pl.Path,
    sfo_specifier: str | None = None,
    plot_settings: AMSViewPlotSettings | None = None,
    save_file: str | pl.Path | None = None,
    calculated_field_specified: str | None = None,
) -> None:
    """
    Runs the amsview command on the rkf files. Can be used to plot orbitals and geometry.

    Args:
        input_file: Path to the input file that contains volume data such as .t21, .t41, .rkf, .vtk and .runkf files
        sfo_specifier: The orbital specifier with the format [type]_[irrep]_[index] such as SCF_A_6 or SFO_E1:1_1
        plot_settings: Instance of PlotSettings with the following attributes:
            - bgcolor: The background color in hexadecimals (start with # and then 6 digits)
            - scmgeometry: The size of the image (WxH in pixels, e.g. "1920x1080")
            - zoom: The zoom level (float)
            - antialias: Whether to use antialiasing (bool)
            - viewplane: The viewplane normal to the specified x,y,z direction (three numbers for x,y,z e.g. "1 0 1")
            - grid: The grid size (Coarse, Medium, Fine)
            - wireframe: Whether to use wireframe (bool)
            - transparent: Whether to use transparency (bool)
            - colorfield: The colorfield (three numbers for r,g,b e.g. "100 299 321")
            - printrange: Whether to print the colour range (bool)
            - camera: The camera load-outs from AMS (int)
            - hide_view: Whether to hide the amsview application (bool)
            - print_command: Whether to print the command (bool)

    Check for all options by running amsview -h

    Example command for one MO: amsview result.t41 -var SCF_A_8 -save "my_pic.png" -bgcolor "#FFFFFF" -transparent -antialias -scmgeometry "2160x1440" -wireframe
    Example command for one SFO: amsview result.t41 -var SFO_8 -save "my_pic.png" -bgcolor "#FFFFFF" -transparent -antialias -scmgeometry "2160x1440" -wireframe
    Example command for overlap field: amsview result.t41 -calculated "SFO_7 * SFO_7" -save "my_pic.png" -bgcolor "#FFFFFF" -transparent -antialias -scmgeometry "2160x1440" -wireframe
    """
    plot_settings = plot_settings or AMSViewPlotSettings()

    command = ["amsview", str(input_file)]

    # Either calculate a product of MOs with
    if calculated_field_specified is not None:
        command.append("-calculated")
        command.append(f"'{str(calculated_field_specified)}'")
    elif sfo_specifier is not None:
        command.append("-var")
        command.append(str(sfo_specifier))

    if plot_settings.hide_view:
        command.append("-batch")

    if save_file is not None:
        save_file = pl.Path(save_file).with_suffix(".png")
        command.append("-save")
        command.append(str(save_file))

    dict_settings = asdict(plot_settings)

    if plot_settings.camera > 0:
        dict_settings.pop("viewplane")
    elif plot_settings.camera < 0:
        dict_settings.pop("camera")

    for key, value in dict_settings.items():
        if key.lower() in ["wireframe", "transparent", "antialias", "printrange", "ciso"]:
            if value:
                command.append(f"-{key}")
            continue

        command.append(f"-{key}")
        command.append(str(value))

    if plot_settings.print_command:
        print(" ".join(command))
    subprocess.run(command)


def plot_orbital_with_amsreport(
    input_file: str | pl.Path,
    out_dir: str | pl.Path,
    sfo_specifier: str,
    plot_settings: PlotSettings | None = None,
) -> None:
    """
    Runs the amsreport command on the rkf files
    Look at https://www.scm.com/doc/Scripting/Commandline_Tools/AMSreport.html for options and more informations

    Args:
        input_file: Path to the input file e.g., .t21, .t41, .rkf,
        sfo_specifier: The orbital specifier, e.g., "HOMO[-x]" and "LUMO[+x]" (see link for more options)
        plot_settings: Instance of PlotSettings with the following attributes:
            - bgcolor: The background color in hexadecimals (start with # and then 6 digits)
            - scmgeometry: The size of the image (WxH in pixels, e.g. "1920x1080")
            - zoom: The zoom level (float)
            - antialias: Whether to use antialiasing (bool)
            - viewplane: The viewplane normal to the specified x,y,z direction (three numbers for x,y,z e.g. "{1 0 1}")
            - grid: The grid size (Coarse, Medium, Fine)
            - hide_view: Whether to hide the amsview application (bool)
            - print_command: Whether to print the command (bool)
    """
    plot_settings = plot_settings or AMSViewPlotSettings()
    out_dir = pl.Path(out_dir)
    tmp_dir = out_dir / sfo_specifier

    command = ["amsreport", "-i", str(input_file), "-o", str(tmp_dir), sfo_specifier]

    dict_settings = asdict(plot_settings)
    for key, value in dict_settings.items():
        if key == "antialias":
            if value:
                command.append("-v")
                command.append("-antialias")
            continue

        command.append("-v")
        command.append(f"-{key} {value}")

    if plot_settings.print_command:
        print(" ".join(command))

    subprocess.run(command)

    # Copy all the files from the tmp directory to the out_dir
    subprocess.run(["cp", f"{str(tmp_dir)}.jpgs/0.jpg", str(out_dir / f"{sfo_specifier}.jpg")])
    shutil.rmtree(f"{tmp_dir}.jpgs")
    (out_dir / sfo_specifier).unlink()  # remove a file that is created by amsreport which is not used


def combine_orb_images_with_matplotlib(
    system_name: str,
    sfos: Sequence[Orbital],
    sfo_image_paths: Sequence[str | pl.Path],
    out_path: str | pl.Path,
) -> None:
    """Combines multiple orbital images into one image using matplotlib. The images are plotted in an array of subplots."""
    n_sfos = len(sfos)
    n_cols = 4
    n_rows = n_sfos // n_cols + 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Flatten the axes
    ax = ax.flatten()

    sfo_images = [mpimg.imread(sfo_image_path) for sfo_image_path in sfo_image_paths]

    for i in range(n_rows * n_cols):
        if i < n_sfos:
            sfo = sfos[i]
            img = sfo_images[i]
            ax[i].imshow(img)
            ax[i].set_title(f"{sfo.amsview_label}\n{sfo.homo_lumo_label}\nEnergy (eV): {sfo.energy :.3f}", fontsize=12)  # NOQA E203
            ax[i].axis("off")  # Turn off the axis

            # Zoom in on the image
            ax[i].set_xlim(img.shape[1] * 0.28, img.shape[1] * 0.72)
            ax[i].set_ylim(img.shape[0] * 0.72, img.shape[0] * 0.28)
        else:
            # Remove the extra subplot
            fig.delaxes(ax[i])

    fig.tight_layout()

    plt.suptitle(system_name)
    plt.savefig(out_path)
    plt.close()


def combine_sfo_images_with_matplotlib(
    sfo1: SFO,
    sfo1_image_path: str | pl.Path,
    sfo2: SFO,
    sfo2_image_path: str | pl.Path,
    out_path: str | pl.Path,
    overlap: float | None = None,
    energy_gap: float | None = None,
    stabilization: float | None = None,
) -> None:
    """
    Combines two SFO images into one image using matplotlib. The images are plotted on top of each other.

    Args:
        sfo1: The first SFO
        sfo1_image_path: The path to the first SFO image
        sfo2: The second SFO
        sfo2_image_path: The path to the second SFO image
        out_path: The path to the output image
    """

    fig, ax = plt.subplots(1, 2, figsize=(5, 5))
    img1 = mpimg.imread(sfo1_image_path)
    img2 = mpimg.imread(sfo2_image_path)

    for i, (sfo, img) in enumerate(zip([sfo1, sfo2], [img1, img2])):
        ax[i].imshow(img)
        ax[i].set_title(f"{sfo.amsview_label}\nGross Pop: {sfo.gross_pop :.3f}\nEnergy (eV): {sfo.energy :.2f}", fontsize=12)  # NOQA E203
        ax[i].axis("off")  # Turn off the axis

        # Zoom in on the image
        ax[i].set_xlim(img.shape[1] * 0.28, img.shape[1] * 0.72)
        ax[i].set_ylim(img.shape[0] * 0.72, img.shape[0] * 0.28)

    overlap_str = f"Overlap: {overlap:.3f}" or ""
    energy_gap_str = f"Energy gap (eV): {energy_gap:.3f}" or ""
    stabilization_str = f"Stabilization: {stabilization:.3f}" if stabilization is not None else ""

    fig.tight_layout()

    plt.suptitle(f"{overlap_str}\n{energy_gap_str}\n{stabilization_str}")
    plt.savefig(out_path)
    plt.close()
