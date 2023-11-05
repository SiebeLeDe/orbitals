from scm.plams import Settings, DensfJob, init, finish
import pathlib as pl
from orb_visualization.densf_presets import grid, orbital, output, potential

# amsview result.t41 -var SCF_A_8 -save "my_pic.png" -bgcolor "#FFFFFF" -transparent -antialias -scmgeometry "2160x1440" -wireframe

# Set up paths
current_dir = pl.Path(__file__).parent
plams_name = "test_nh3_bh3"
calculation = ("complex", "1")  # (name, index), e.g., (frag1, 3), (frag2, 1), (complex, 2)
output_dirname = "output_files"

# Set up directories
output_dir = current_dir / output_dirname
pyfrag_dir = current_dir / "NH3_BH3" / "nh3_bh3"
rkf_path = pyfrag_dir / f"{calculation[0]}.{calculation[1].zfill(5)}" / "adf.rkf"

# DensF settings
set = Settings()
set.update(grid(grid_type="fine"))
set.update(output(outputfile=str(output_dir / "result")))
set.update(orbital(all_option=(1, 2)))
set.update(potential("xc", True))

# Make sure the output dir exists
output_dir.mkdir(exist_ok=True)

# Run DensF
init(path=current_dir, folder=plams_name)
job = DensfJob(inputjob=rkf_path, settings=set)
job.run()
print(job.get_input())
finish()
