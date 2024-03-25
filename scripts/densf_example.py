import pathlib as pl

import matplotlib.pyplot as plt
from orb_visualization.densf_presets import grid, orbital, output
from orb_visualization.plotter import AMSViewPlotSettings, plot_orbital_with_amsview
from scm.plams import DensfJob, KFFile, Settings, finish, init


# ------------------Available test files------------------
class RestrictedTestFiles:
    FILE1 = "restricted_largecore_differentfragsym_c4v_full"
    FILE2 = "restricted_largecore_differentfragsym_c4v_full"
    FILE3 = "restricted_largecore_fragsym_c3v_nonrelativistic_full"
    FILE4 = "restricted_largecore_fragsym_c3v_nonrelativistic_full"
    FILE5 = "restricted_nocore_fragsym_c3v_full"
    FILE6 = "restricted_nocore_fragsym_nosym_full"


# --------------------Input Arguments-------------------- #
current_path = pl.Path(__file__).parent
path_to_folder_with_rkf_files = current_path.parent / "test" / "fixtures" / "rkfs"
rkf_file = "restricted_nocore_fragsym_nosym_full"
output_dirname = "densf_unrestricted_output"
plams_foldername = "densf_calc"

# Set up directories
output_dir = current_path / output_dirname
rkf_path = pl.Path("/Users/siebeld/ADF_Calcs/Me2.results/adf.rkf")

# DensF settings
set = Settings()
set.update(grid(grid_type="coarse"))
set.update(output(outputfile=str(output_dir / "result")))
set.update(orbital(type="SFO", irrep_number_label=("A", [11, 16])))

# Make sure the output dir exists
output_dir.mkdir(exist_ok=True)

# Run DensF
init(path=current_path, folder=plams_foldername)
job = DensfJob(inputjob=rkf_path, settings=set)
job.run()
print(job.get_input())
# if job.ok():
# config.erase_workdir = True
finish()

# plot orbital
outfile = output_dir / "result.t41"
kf_file = KFFile(outfile)

orbitals = []
possible_orbital_sections = ["SCF_A_A", "SCF_A_B", "SFO_A_A", "SFO_A_B"]
for section in possible_orbital_sections:
    if section in kf_file:
        variables = KFFile(outfile).read_section(section).keys()
        for var in variables:
            # Next, check if the variable is an integer. If so, append it to the list of orbitals with the variable name + the orbital number
            try:
                int(var)
                orbitals.append(f"{section}_{var}")
            except ValueError:
                pass

# print(KFFile(outfile).read_section("SCF_A"))
plot_settings = AMSViewPlotSettings(scmgeometry="1600x920", zoom=1.0, viewplane="1 1 0", transparent=False)
image_paths = []

for orb in orbitals:
    image_path = output_dir / f"{orb}.png"
    plot_orbital_with_amsview(outfile, orb, plot_settings=plot_settings, save_file=output_dir / f"{orb}.png")
    image_paths.append(image_path)

# Using the image paths, we can create a plot with all the orbitals using matplotlib and subplots

# Create subplots
num_plots = len(image_paths)

# Divide the figure into a grid of rows with three columns
num_rows = (num_plots + 2) // 3 if num_plots > 1 else 1
num_cols = min(num_plots, 3) if num_plots > 1 else 1

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
if axes.ndim == 1:
    axes = axes.reshape(1, -1)

# Plot each orbital image
for i, image_path in enumerate(image_paths):
    row = i // num_cols if num_plots > 1 else 0
    col = i % num_cols if num_plots > 1 else 0

    img = plt.imread(str(image_path), format="png")
    axes[row, col].imshow(img)
    axes[row, col].axis("off")
    axes[row, col].set_title(image_path.stem, fontsize=12)

# Adjust spacing between subplots
plt.tight_layout()

# Save the plot
plt.savefig(output_dir / "orbitals_plot.png")

# Show the plot
plt.show()
