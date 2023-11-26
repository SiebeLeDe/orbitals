import pathlib as pl
import subprocess
from typing import Optional

from attrs import define, asdict


@define
class PlotSettings:
    """ General plot settings used for amsreport """
    bgcolor: str = "#ffffff"
    scmgeometry: str = "600x600"
    zoom: float = 1.5
    antialias: bool = True
    viewplane: str = "{0 0 1}"


@define
class AMSViewPlotSettings(PlotSettings):
    """ Plot settings for amsview. Inherits from PlotSettings """
    wireframe: bool = False
    transparent: bool = False
    viewplane: str = "0 0 1"


def plot_orbital_with_amsview(
    input_file: str | pl.Path,
    orb_specifier: str,
    plot_settings: Optional[AMSViewPlotSettings] = AMSViewPlotSettings(),
    save_file: Optional[str | pl.Path] = None,
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

    Example command: amsview result.t41 -var SCF_A_8 -save "my_pic.png" -bgcolor "#FFFFFF" -transparent -antialias -scmgeometry "2160x1440" -wireframe
    """
    command = ["amsview", input_file, "-var", orb_specifier]

    if save_file is not None:
        command.append("-save")
        command.append(str(save_file))

    dict_settings = asdict(plot_settings)
    for key, value in dict_settings.items():
        if key.lower() in ["wireframe", "transparent", "antialias"]:
            if value:
                command.append(f"-{key}")
            continue

        command.append(f"-{key}")
        command.append(str(value))

    subprocess.run(command)


def plot_orbital_with_amsreport(
    out_dir: str,
    rkf_file_path: str,
    orb_specifier: str,
    plot_settings: Optional[PlotSettings] = PlotSettings(),
) -> None:
    """
    Runs the amsreport command on the rkf files

    Args:
        out_dir: The path to the output directory
        rkf_file_path: The path to the rkf file
        orb_specifier: The orbital specifier
        plot_settings: Instance of PlotSettings with the following attributes:
            - bgcolor: The background color in hexadecimals (start with # and then 6 digits)
            - scmgeometry: The size of the image (WxH in pixels, e.g. "1920x1080")
            - zoom: The zoom level (float)
            - antialias: Whether to use antialiasing (bool)
            - viewplane: The viewplane normal to the specified x,y,z direction (three numbers for x,y,z e.g. "{1 0 1}")
    """

    command = ["amsreport", "-i", rkf_file_path, "-o", out_dir, orb_specifier]

    dict_settings = asdict(plot_settings)
    for key, value in dict_settings.items():
        if key == "antialias":
            if value:
                command.append("-v")
                command.append("-antialias")
            continue

        command.append("-v")
        command.append(f"-{key} {value}")

    subprocess.run(command)
