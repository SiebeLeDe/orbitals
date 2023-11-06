from scm.plams import Settings, DensfJob, init, finish, config, KFFile
import pathlib as pl
from orb_visualization.densf_presets import grid, orbital, output


# ------------------Available test files------------------
class Restricted_TestFiles:
    FILE1 = "restricted_largecore_differentfragsym_c4v_full"
    FILE2 = "restricted_largecore_differentfragsym_c4v_full"
    FILE3 = "restricted_largecore_fragsym_c3v_nonrelativistic_full"
    FILE4 = "restricted_largecore_fragsym_c3v_nonrelativistic_full"
    FILE5 = "restricted_nocore_fragsym_c3v_full"
    FILE6 = "restricted_nocore_fragsym_nosym_full"


# --------------------Input Arguments-------------------- #
current_path = pl.Path(__file__).parent
path_to_folder_with_rkf_files = (current_path.parent / "test" / "fixtures" / "rkfs")
rkf_file = "restricted_nocore_fragsym_nosym_full"
output_dirname = "densf_output"
plams_foldername = "densf_calc"

# Set up directories
output_dir = current_path / output_dirname
rkf_path = path_to_folder_with_rkf_files / f"{rkf_file}.adf.rkf"

# DensF settings
set = Settings()
set.update(grid(grid_type="fine"))
set.update(output(outputfile=str(output_dir / "result")))
set.update(orbital(all_option=(1, 2)))

# Make sure the output dir exists
output_dir.mkdir(exist_ok=True)

# Run DensF
init(path=current_path, folder=plams_foldername)
job = DensfJob(inputjob=rkf_path, settings=set)
job.run()
# print(job.get_input())
if job.ok():
    config.erase_workdir = True
finish()

# plot orbital
outfile = output_dir / "result.t41"
print(KFFile(outfile).sections())
# print(KFFile(outfile).read_section("SCF_A"))
orb_specifier = "SCF_A_11"
# plot_settings = AMSViewPlotSettings(scmgeometry="1920x1080", zoom=1.5)

# plot_orbital_with_amsview(outfile, orb_specifier, save_file=output_dir / f"{orb_specifier}.png")
