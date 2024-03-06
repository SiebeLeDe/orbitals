import pathlib as pl
import shutil
import subprocess
from typing import Sequence, TypeVar

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from attrs import asdict, define
from orb_analysis.orbital.orbital import Orbital


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
    orb_specifier: str | None = None,
    plot_settings: AMSViewPlotSettings | None = AMSViewPlotSettings(),
    save_file: str | pl.Path | None = None,
) -> None:
    """
    Runs the amsview command on the rkf files. Can be used to plot orbitals and geometry.

    Args:
        input_file: Path to the input file that contains volume data such as .t21, .t41, .rkf, .vtk and .runkf files
        orb_specifier: The orbital specifier with the format [type]_[irrep]_[index] such as SCF_A_6 or SFO_E1:1_1
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

    Example command: amsview result.t41 -var SCF_A_8 -save "my_pic.png" -bgcolor "#FFFFFF" -transparent -antialias -scmgeometry "2160x1440" -wireframe
    """
    command = ["amsview", str(input_file)]

    if orb_specifier is not None:
        command.append("-var")
        command.append(str(orb_specifier))

    if plot_settings.hide_view:
        command.append("-batch")

    if save_file is not None:
        save_file = pl.Path(save_file).with_suffix(".png")
        command.append("-save")
        command.append(str(save_file))

    dict_settings = asdict(plot_settings)
    for key, value in dict_settings.items():
        if key.lower() in ["wireframe", "transparent", "antialias", "printrange", "ciso"]:
            if value:
                command.append(f"-{key}")
            continue

        if key.lower() == "camera" and not value > 0:
            continue

        command.append(f"-{key}")
        command.append(str(value))

    if plot_settings.print_command:
        print(" ".join(command))
    subprocess.run(command)


def plot_orbital_with_amsreport(
    input_file: str | pl.Path,
    out_dir: str | pl.Path,
    orb_specifier: str,
    plot_settings: PlotSettingsType | None = AMSViewPlotSettings(),
) -> None:
    """
    Runs the amsreport command on the rkf files
    Look at https://www.scm.com/doc/Scripting/Commandline_Tools/AMSreport.html for options and more informations

    Args:
        input_file: Path to the input file e.g., .t21, .t41, .rkf,
        orb_specifier: The orbital specifier, e.g., "HOMO[-x]" and "LUMO[+x]" (see link for more options)
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
    tmp_dir = out_dir / orb_specifier

    command = ["amsreport", "-i", str(input_file), "-o", str(tmp_dir), orb_specifier]

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
    subprocess.run(["cp", f"{str(tmp_dir)}.jpgs/0.jpg", str(out_dir / f"{orb_specifier}.jpg")])
    shutil.rmtree(f"{tmp_dir}.jpgs")
    (out_dir / orb_specifier).unlink()  # remove a file that is created by amsreport which is not used


T = TypeVar("T", bound=Orbital)


def combine_orb_images_with_matplotlib(
    system_name: str,
    orbs: Sequence[T],
    orb_image_paths: Sequence[str | pl.Path],
    out_path: str | pl.Path,
) -> None:
    """Combines multiple orbital images into one image using matplotlib. The images are plotted in an array of subplots."""
    n_orbs = len(orbs)
    n_cols = 4
    n_rows = n_orbs // n_cols + 1

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    # Flatten the axes
    ax = ax.flatten()

    orb_images = [mpimg.imread(orb_image_path) for orb_image_path in orb_image_paths]

    for i in range(n_rows * n_cols):
        if i < n_orbs:
            orb = orbs[i]
            img = orb_images[i]
            ax[i].imshow(img)
            ax[i].set_title(f"{orb.amsview_label}\n{orb.homo_lumo_label}\nEnergy (eV): {orb.energy :.3f}", fontsize=12)  # NOQA E203
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
    orb1: Orbital,
    orb1_image_path: str | pl.Path,
    orb2: Orbital,
    orb2_image_path: str | pl.Path,
    out_path: str | pl.Path,
    overlap: float | None = None,
    energy_gap: float | None = None,
    stabilization: float | None = None,
) -> None:
    """
    Combines two orbital images into one image using matplotlib. The images are plotted on top of each other.

    Args:
        orb1: The first orbital
        orb1_image_path: The path to the first orbital image
        orb2: The second orbital
        orb2_image_path: The path to the second orbital image
        out_path: The path to the output image
    """

    fig, ax = plt.subplots(1, 2, figsize=(5, 5))
    img1 = mpimg.imread(orb1_image_path)
    img2 = mpimg.imread(orb2_image_path)

    for i, (orb, img) in enumerate(zip([orb1, orb2], [img1, img2])):
        ax[i].imshow(img)
        ax[i].set_title(f"{orb.amsview_label}\nGross Pop: {orb.gross_pop :.3f}\nEnergy (eV): {orb.energy :.2f}", fontsize=12)  # NOQA E203
        ax[i].axis("off")  # Turn off the axis

        # Zoom in on the image
        ax[i].set_xlim(img.shape[1] * 0.28, img.shape[1] * 0.72)
        ax[i].set_ylim(img.shape[0] * 0.72, img.shape[0] * 0.28)

    overlap_str = f"Overlap: {overlap:.3f}" if overlap is not None else ""
    energy_gap_str = f"Energy gap (eV): {energy_gap:.3f}" if energy_gap is not None else ""
    stabilization_str = f"Stabilization: {stabilization:.3f}" if stabilization is not None else ""

    fig.tight_layout()

    plt.suptitle(f"{overlap_str}\n{energy_gap_str}\n{stabilization_str}")
    plt.savefig(out_path)
    plt.close()
