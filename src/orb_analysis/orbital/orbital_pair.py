from __future__ import annotations

import pathlib as pl

import attrs
import numpy as np
import pandas as pd
from orb_visualization.plotter import AMSViewPlotSettings, combine_sfo_images_with_matplotlib, plot_orbital_with_amsview
from tabulate import tabulate

from orb_analysis.custom_types import Array1D
from orb_analysis.orbital.orbital import SFO
from orb_analysis.orbital_manager.shared_functions import calculate_matrix_element


@attrs.define
class OrbitalPair:
    sfo1: SFO
    sfo2: SFO
    overlap: float

    def __str__(self):
        if self.is_pauli_pair:
            return f"frag1 SFO: {self.sfo1.homo_lumo_label:15s} frag2 SFO: {self.sfo2.homo_lumo_label:15s} Overlap: {self.overlap:.3f}"

        return f"frag1 SFO: {self.sfo1.homo_lumo_label:15s} frag2 SFO: {self.sfo2.homo_lumo_label:15s} Overlap: {self.overlap:.3f} Stabilization: {self.stabilization:.3f}"

    @property
    def is_pauli_pair(self) -> bool:
        return self.sfo1.is_fully_occupied and self.sfo2.is_fully_occupied

    @property
    def energy_gap(self) -> float:
        return abs(self.sfo1.energy - self.sfo2.energy)

    @property
    def stabilization(self) -> float | None:
        if self.is_pauli_pair:
            return None
        return calculate_matrix_element(self.sfo1, self.sfo2, self.overlap)

    @property
    def as_numpy_array(self) -> Array1D:
        array = np.array(
            [
                f"{self.sfo1.amsview_label} {self.sfo1.homo_lumo_label}",
                self.sfo1.energy,
                self.sfo1.gross_pop,
                f"{self.sfo2.amsview_label} {self.sfo2.homo_lumo_label}",
                self.sfo2.energy,
                self.sfo2.gross_pop,
                self.overlap,
                self.energy_gap,
                self.stabilization,
            ]
        )
        return array

    def plot(self, rkf_file: pl.Path | str, output_dir: str | pl.Path, plot_settings: AMSViewPlotSettings | None = None):
        """Plots the orbitals associated with this pair"""
        plot_settings = AMSViewPlotSettings() if plot_settings is None else plot_settings

        image_paths = []
        for orb in [self.sfo1, self.sfo2]:
            save_file = pl.Path(output_dir / f"{orb.irrep}_{orb.index}.png")
            if not save_file.exists():
                plot_orbital_with_amsview(str(rkf_file), orb.plot_label, plot_settings, save_file=save_file)
            image_paths.append(save_file)

        combine_sfo_images_with_matplotlib(
            sfo1=self.sfo1,
            sfo2=self.sfo2,
            sfo1_image_path=image_paths[0],
            sfo2_image_path=image_paths[1],
            out_path=output_dir / f"{self.sfo1.amsview_label}-{self.sfo2.amsview_label}",
            overlap=self.overlap,
            energy_gap=self.energy_gap,
            stabilization=self.stabilization,
        )

    @staticmethod
    def format_orbital_pairs_for_printing(orb_pairs: list[OrbitalPair], top_header: str = "SFO Interaction Pairs") -> str:
        """
        Returns a string representing each orbital pair in the list in a nicely formatted table
        Example:
            [sfo1_label] [sfo1_energy] [sfo1_grosspop] [sfo2_label] [sfo2_energy] [sfo2_grosspop] [overlap] [stabilization]
        """
        headers = ["SFO1", "energy (eV)", "gross pop (a.u.)", "SFO2", "energy (eV)", "gross pop (a.u.)", "Overlap S", "epsilon (eV)", "S^2/epsilon * 100"]

        # Create a DataFrame
        df = pd.DataFrame([orb_pair.as_numpy_array for orb_pair in orb_pairs], columns=headers)

        # Use tabulate to create a formatted string
        result = tabulate(df, headers="keys", tablefmt="simple", showindex=False, floatfmt="+.3f")  # type: ignore # df is accepted as argument

        return f"\n{top_header}\n{result}"
