import pathlib as pl
import shutil
import subprocess
from typing import TypeVar

from attrs import asdict, define


@define
class PlotSettings:
    """General plot settings used for amsreport"""

    bgcolor: str = "#FFFFFF"
    scmgeometry: str = "1920x1080"
    zoom: float = 1.5
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
    # colorfield: str = "100 299 321"  # No idea what this value should be
    # printrange: bool = True
    # camera: int = -1  # Camera load-outs from AMS
    # val: float = 0.03  # isovalue
    # ciso: bool = False


@define
class AMSReportPlotSettings(PlotSettings):
    """Plot settings for amsreport. Inherits from PlotSettings"""

    grid: str = "Medium"  # Grid size (Coarse, Medium, Fine)


PlotSettingsType = TypeVar("PlotSettingsType", bound=PlotSettings)


def plot_orbital_with_amsview(
    input_file: str | pl.Path,
    orb_specifier: str,
    plot_settings: AMSViewPlotSettings | None = AMSViewPlotSettings(),
    save_file: str | pl.Path | None = None,
) -> None:
    """
    Runs the amsview command on the rkf files

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
    command = ["amsview", str(input_file), "-var", orb_specifier]

    if plot_settings.hide_view:
        command.append("-batch")

    if save_file is not None:
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
