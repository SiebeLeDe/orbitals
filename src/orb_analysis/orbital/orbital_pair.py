from __future__ import annotations

import pathlib as pl

import attrs
import numpy as np
import pandas as pd
from orb_visualization.plotter import AMSViewPlotSettings, plot_orbital_with_amsview
from tabulate import tabulate

from orb_analysis.custom_types import Array1D
from orb_analysis.orbital.orbital import SFO
from orb_analysis.orbital_manager.shared_functions import calculate_matrix_element


@attrs.define
class OrbitalPair:
    orb1: SFO
    orb2: SFO
    overlap: float

    def __str__(self):
        if self.is_pauli_pair:
            return f"frag1 SFO: {self.orb1.homo_lumo_label:15s} frag2 SFO: {self.orb2.homo_lumo_label:15s} Overlap: {self.overlap:.3f}"

        return f"frag1 SFO: {self.orb1.homo_lumo_label:15s} frag2 SFO: {self.orb2.homo_lumo_label:15s} Overlap: {self.overlap:.3f} Stabilization: {self.stabilization:.3f}"

    @property
    def is_pauli_pair(self) -> bool:
        return self.orb1.is_fully_occupied and self.orb2.is_fully_occupied

    @property
    def energy_gap(self) -> float:
        return abs(self.orb1.energy - self.orb2.energy)

    @property
    def stabilization(self) -> float | str:
        if self.is_pauli_pair:
            return "---"
        return calculate_matrix_element(self.orb1, self.orb2, self.overlap)

    @property
    def as_numpy_array(self) -> Array1D:
        array = np.array(
            [
                f"{self.orb1.amsview_label} {self.orb1.homo_lumo_label}",
                self.orb1.energy,
                self.orb1.gross_pop,
                f"{self.orb2.amsview_label} {self.orb2.homo_lumo_label}",
                self.orb2.energy,
                self.orb2.gross_pop,
                self.overlap,
                self.energy_gap,
                self.stabilization,
            ]
        )
        return array

    def plot(self, rkf_file: pl.Path | str, output_dir: str | pl.Path, plot_settings: AMSViewPlotSettings | None = None):
        """Plots the orbitals associated with this pair"""
        plot_settings = AMSViewPlotSettings() if plot_settings is None else plot_settings

        for orb in [self.orb1, self.orb2]:
            plot_orbital_with_amsview(str(rkf_file), self.orb1.amsview_label, plot_settings, save_file=output_dir / f"{self.orb1.irrep}_{self.orb1.index}")

    @staticmethod
    def format_orbital_pairs_for_printing(orb_pairs: list[OrbitalPair], top_header: str = "SFO Interaction Pairs") -> str:
        """
        Returns a string representing each orbital pair in the list in a nicely formatted table
        Example:
            [orb1_label] [orb1_energy] [orb1_grosspop] [orb2_label] [orb2_energy] [orb2_grosspop] [overlap] [stabilization]
        """
        headers = ["SFO1", "energy (eV)", "gross pop (a.u.)", "SFO2", "energy (eV)", "gross pop (a.u.)", "Overlap S", "epsilon (eV)", "S^2/epsilon * 100"]

        # Create a DataFrame
        df = pd.DataFrame([orb_pair.as_numpy_array for orb_pair in orb_pairs], columns=headers)

        # Use tabulate to create a formatted string
        result = tabulate(df, headers="keys", tablefmt="simple", showindex=False, floatfmt="+.3f")  # type: ignore # df is accepted as argument

        return f"\n{top_header}\n{result}"
