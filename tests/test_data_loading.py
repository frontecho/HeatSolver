from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_loading import build_regular_index, fill_regular_grid, load_heat_flow_observations


class DataLoadingTest(unittest.TestCase):
    def testbuild_regular_index_andfill_regular_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "X", "Y"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"id": "1", "X": 0.0, "Y": 0.0},
                        {"id": "2", "X": 1.0, "Y": 0.0},
                        {"id": "3", "X": 0.0, "Y": 1.0},
                        {"id": "4", "X": 1.0, "Y": 1.0},
                    ]
                )
            _, x_unique, y_unique, ri, ci = build_regular_index(path)
            arr = fill_regular_grid(np.array([10.0, 20.0, 30.0, 40.0]), len(y_unique), len(x_unique), ri, ci)

        np.testing.assert_array_equal(x_unique, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(y_unique, np.array([0.0, 1.0]))
        np.testing.assert_allclose(arr, np.array([[10.0, 20.0], [30.0, 40.0]]))

    def test_load_heat_flow_observations_maps_scatter_points_to_grid_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grid_csv = root / "grid.csv"
            heatflow_csv = root / "heatflow.csv"
            with grid_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "X", "Y"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"id": "1", "X": 0.0, "Y": 0.0},
                        {"id": "2", "X": 1_000.0, "Y": 0.0},
                        {"id": "3", "X": 0.0, "Y": 1_000.0},
                        {"id": "4", "X": 1_000.0, "Y": 1_000.0},
                    ]
                )
            with heatflow_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "lon", "lat", "X", "Y", "Heat_Flow"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"id": "a", "lon": 0.0, "lat": 0.0, "X": 10.0, "Y": 10.0, "Heat_Flow": 50.0},
                        {"id": "b", "lon": 0.0, "lat": 0.0, "X": 1_020.0, "Y": 20.0, "Heat_Flow": 70.0},
                        {"id": "c", "lon": 0.0, "lat": 0.0, "X": 0.0, "Y": 990.0, "Heat_Flow": 90.0},
                    ]
                )

            full = load_heat_flow_observations(grid_csv, heatflow_csv, stride=1)
            coarse = load_heat_flow_observations(grid_csv, heatflow_csv, stride=2)

        np.testing.assert_allclose(full, np.array([[50.0, 70.0], [90.0, np.nan]]), equal_nan=True)
        np.testing.assert_allclose(coarse, np.array([[70.0]]))


if __name__ == "__main__":
    unittest.main()
