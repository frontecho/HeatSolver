from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from data_loading import THERMAL_PROPERTIES_FILENAME
from lithosphere_model import (
    SEDIMENT_STRATUM_NAMES,
    THERMAL_LAYER_NAMES,
    LayerGrid,
    build_reference_fields,
    load_layer_parameters,
)

THERMAL_PROPERTIES_PATH = Path("Data") / THERMAL_PROPERTIES_FILENAME


def write_layergrid_csvs(
    root: Path,
    *,
    nx: int = 3,
    ny: int = 2,
    include_thermal: bool = True,
    include_stratum: bool = False,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    grid_csv = root / "grid.csv"
    layer_csv = root / "layer.csv"
    thermal_csv = root / "thermal.csv"
    heatflow_csv = root / "heatflow.csv"
    stratum_csv = root / "stratum.csv"

    grid_rows = []
    layer_rows = []
    thermal_rows = []
    heatflow_rows = []
    stratum_rows = []
    idx = 1
    for iy in range(ny):
        for ix in range(nx):
            cell_id = str(idx)
            x = float(ix * 1_000.0)
            y = float(iy * 1_000.0)
            grid_rows.append({"id": cell_id, "lon": 100.0 + ix, "lat": 35.0 + iy, "X": x, "Y": y})
            layer_rows.append(
                {
                    "id": cell_id,
                    "Total_Sediment": 1.0 + 0.1 * ix,
                    "Moho": 30.0 + iy,
                    "thk_uc": 10.0,
                    "thk_mc": 12.0,
                }
            )
            thermal_rows.append({"id": cell_id, "Surface_Temperature": 8.0 + iy + ix})
            heatflow_rows.append({"id": cell_id, "lon": 100.0 + ix, "lat": 35.0 + iy, "X": x, "Y": y, "Heat_Flow": 60.0 + idx})
            stratum_rows.append(
                {
                    "id": cell_id,
                    **{name: 100.0 * (i + 1) for i, name in enumerate(SEDIMENT_STRATUM_NAMES)},
                    "zhongyuangujie": 900.0,
                }
            )
            idx += 1

    with grid_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "lon", "lat", "X", "Y"])
        writer.writeheader()
        writer.writerows(grid_rows)
    with layer_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "Total_Sediment", "Moho", "thk_uc", "thk_mc"])
        writer.writeheader()
        writer.writerows(layer_rows)
    if include_thermal:
        with thermal_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "Surface_Temperature"])
            writer.writeheader()
            writer.writerows(thermal_rows)
    with heatflow_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "lon", "lat", "X", "Y", "Heat_Flow"])
        writer.writeheader()
        writer.writerows(heatflow_rows)
    if include_stratum:
        with stratum_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", *SEDIMENT_STRATUM_NAMES, "zhongyuangujie"])
            writer.writeheader()
            writer.writerows(stratum_rows)
    return {"grid": grid_csv, "layer": layer_csv, "thermal": thermal_csv, "heatflow": heatflow_csv, "stratum": stratum_csv}


class LoadLayerParametersTest(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_layer_parameters(Path(tmp) / "missing.json")

    def test_explicit_layers_json_is_parsed(self) -> None:
        payload = {
            "layers": {
                "sediment": {"A": 1.2, "K0": 2.0, "c": 0.0},
                "upper_crust": {"A0": 2.1, "D": 15.0, "K0": 2.9, "c": 1e-5},
                "middle_crust": {"A": 0.7, "K0": 2.7, "c": 2e-4},
                "lower_crust": {"A": 0.2, "K0": 2.8, "c": 3e-4},
                "mantle": {"A": 0.03, "K0": 3.4, "c": 0.0},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thermal.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            a0, decay, layers = load_layer_parameters(path)

        self.assertEqual(a0, 2.1)
        self.assertEqual(decay, 15.0)
        self.assertEqual(layers["sediment"], (1.2, 2.0, 0.0))
        self.assertEqual(layers["upper_crust"], (None, 2.9, 1e-5))
        self.assertEqual(layers["middle_crust"], (0.7, 2.7, 2e-4))
        self.assertEqual(layers["lower_crust"], (0.2, 2.8, 3e-4))
        self.assertEqual(layers["mantle"], (0.03, 3.4, 0.0))

    def test_incomplete_layers_json_raises(self) -> None:
        payload = {"layers": {"sediment": {"A": 1.2, "K0": 2.0, "c": 0.0}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thermal.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(KeyError):
                load_layer_parameters(path)


class LayerGridTest(unittest.TestCase):
    def test_layergrid_loads_tiny_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=3, ny=2)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)

        self.assertEqual(grid.shape, (2, 3))
        self.assertFalse(hasattr(grid, "surface_temp"))
        self.assertFalse(hasattr(grid, "heat_flow"))
        self.assertEqual(grid.boundaries(0, 0)["sed_base"], 1.0)
        self.assertEqual(grid.boundaries(0, 0)["uc_base"], 11.0)
        self.assertEqual(grid.boundaries(0, 0)["mc_base"], 23.0)
        self.assertEqual(grid.boundaries(0, 0)["moho"], 30.0)

    def test_layergrid_stride_reduces_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=4, ny=3)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=2)

        self.assertEqual(grid.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(grid.x)))
        self.assertTrue(np.all(np.isfinite(grid.y)))

    def test_layergrid_does_not_own_surface_temperature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=2, ny=2, include_thermal=False)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)

        self.assertFalse(hasattr(grid, "surface_temp"))
        self.assertFalse(hasattr(grid, "heat_flow"))

    def test_layergrid_uses_stratum_csv_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=2, ny=2, include_stratum=True)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1, stratum_csv=paths["stratum"])

        self.assertIsNotNone(grid.stratum_thick)
        self.assertAlmostEqual(float(grid.sed_thick[0, 0]), 3.6)
        self.assertNotIn("zhongyuangujie", grid.stratum_thick)
        self.assertAlmostEqual(grid.boundaries(0, 0)["sed_base"], 3.6)
        self.assertAlmostEqual(grid.boundaries(0, 0)["uc_base"], 13.6)


class ReferenceFieldsTest(unittest.TestCase):
    def test_build_reference_fields_shapes_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=2, ny=2)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)
            z_nodes = np.array([0.0, 1.0, 5.0, 15.0, 25.0, 35.0])
            refs = build_reference_fields(grid, z_nodes, thermal_properties_path=THERMAL_PROPERTIES_PATH)

        self.assertEqual(refs.K0_ref.shape, (2, 2, 6))
        self.assertEqual(refs.c_ref.shape, refs.K0_ref.shape)
        self.assertEqual(refs.A_ref.shape, refs.K0_ref.shape)
        self.assertTrue(np.all(np.isfinite(refs.K0_ref)))
        self.assertTrue(np.all(np.isfinite(refs.A_ref)))
        stacked = np.stack(list(refs.layer_masks.values()), axis=0)
        self.assertTrue(np.all(np.sum(stacked, axis=0) == 1))
        self.assertFalse(refs.layer_masks["sediment"][0, 0, 1])
        self.assertTrue(refs.layer_masks["upper_crust"][0, 0, 1])

    def test_build_reference_fields_horizontal_transition_smooths_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=4, ny=3)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)
            z_nodes = np.array([0.0, 0.5, 1.0, 1.5, 5.0, 15.0, 25.0, 35.0])
            hard = build_reference_fields(grid, z_nodes, thermal_properties_path=THERMAL_PROPERTIES_PATH, layer_transition_xy_km=0.0)
            refs = build_reference_fields(grid, z_nodes, thermal_properties_path=THERMAL_PROPERTIES_PATH, layer_transition_xy_km=1.0)

        expected_K0 = gaussian_filter(hard.K0_ref, sigma=(1.0, 1.0, 0.0), mode="nearest")
        expected_A = gaussian_filter(hard.A_ref / 1e-6, sigma=(1.0, 1.0, 0.0), mode="nearest") * 1e-6
        np.testing.assert_allclose(refs.K0_ref, expected_K0)
        np.testing.assert_allclose(refs.A_ref, expected_A)
        stacked = np.stack(list(refs.layer_masks.values()), axis=0)
        np.testing.assert_allclose(np.sum(stacked, axis=0), np.ones(refs.K0_ref.shape))
        self.assertTrue(np.any((refs.layer_masks["sediment"] > 0.0) & (refs.layer_masks["sediment"] < 1.0)))
        self.assertTrue(np.all(np.isfinite(refs.K0_ref)))
        self.assertTrue(np.all(np.isfinite(refs.A_ref)))

    def test_build_reference_fields_vertical_transition_smooths_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=4, ny=3)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)
            z_nodes = np.arange(0.0, 36.0, 1.0)
            hard = build_reference_fields(grid, z_nodes, thermal_properties_path=THERMAL_PROPERTIES_PATH, layer_transition_z_km=0.0)
            refs = build_reference_fields(grid, z_nodes, thermal_properties_path=THERMAL_PROPERTIES_PATH, layer_transition_z_km=1.0)

        expected_K0 = gaussian_filter(hard.K0_ref, sigma=(0.0, 0.0, 1.0), mode="nearest")
        expected_A = gaussian_filter(hard.A_ref / 1e-6, sigma=(0.0, 0.0, 1.0), mode="nearest") * 1e-6
        np.testing.assert_allclose(refs.K0_ref, expected_K0)
        np.testing.assert_allclose(refs.A_ref, expected_A)
        stacked = np.stack(list(refs.layer_masks.values()), axis=0)
        np.testing.assert_allclose(np.sum(stacked, axis=0), np.ones(refs.K0_ref.shape))
        self.assertTrue(np.any((refs.layer_masks["sediment"] > 0.0) & (refs.layer_masks["sediment"] < 1.0)))
        self.assertTrue(np.all(np.isfinite(refs.K0_ref)))
        self.assertTrue(np.all(np.isfinite(refs.A_ref)))

    def test_build_reference_fields_rejects_invalid_transition_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=2, ny=2)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)
            with self.assertRaises(ValueError):
                build_reference_fields(grid, np.array([0.0, 1.0, 2.0]), thermal_properties_path=THERMAL_PROPERTIES_PATH, layer_transition_xy_km=-1.0)
            with self.assertRaises(ValueError):
                build_reference_fields(grid, np.array([0.0, 1.0, 2.0]), thermal_properties_path=THERMAL_PROPERTIES_PATH, layer_transition_z_km=-1.0)

    def test_build_reference_fields_uses_stratum_default_properties(self) -> None:
        payload = {
            "layers": {
                "sediment": {"A": 1.0, "K0": 2.0, "c": 0.0},
                "upper_crust": {"A0": 2.0, "D": 10.0, "K0": 9.0, "c": 0.001},
                "middle_crust": {"A": 0.7, "K0": 2.7, "c": 1e-4},
                "lower_crust": {"A": 0.2, "K0": 2.9, "c": 1e-4},
                "mantle": {"A": 0.03, "K0": 3.5, "c": 0.0},
            },
            "sedments": {
                name: {"default": {"A": 10.0 + i, "K0": 20.0 + i, "c": 0.01 * i}}
                for i, name in enumerate(SEDIMENT_STRATUM_NAMES)
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_layergrid_csvs(root, nx=2, ny=2, include_stratum=True)
            thermal_path = root / "thermal_properties.json"
            thermal_path.write_text(json.dumps(payload), encoding="utf-8")
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1, stratum_csv=paths["stratum"])
            refs = build_reference_fields(grid, np.array([0.05, 0.15, 0.35, 3.5, 3.7]), thermal_properties_path=thermal_path)

        self.assertEqual(set(refs.layer_masks), set(THERMAL_LAYER_NAMES))
        stacked = np.stack(list(refs.layer_masks.values()), axis=0)
        self.assertTrue(np.all(np.sum(stacked, axis=0) == 1))
        self.assertAlmostEqual(float(refs.K0_ref[0, 0, 0]), 20.0)
        self.assertAlmostEqual(float(refs.A_ref[0, 0, 0]), 10.0e-6)
        self.assertAlmostEqual(float(refs.K0_ref[0, 0, 2]), 22.0)
        self.assertAlmostEqual(float(refs.c_ref[0, 0, 2]), 0.02)
        self.assertAlmostEqual(float(refs.K0_ref[0, 0, 3]), 27.0)
        self.assertAlmostEqual(float(refs.K0_ref[0, 0, 4]), 9.0)
        self.assertFalse(bool(refs.layer_masks["sediment"][0, 0, 4]))
        self.assertTrue(bool(refs.layer_masks["upper_crust"][0, 0, 4]))

    def test_build_reference_fields_horizontal_masks_with_stratum_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=3, ny=3, include_stratum=True)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1, stratum_csv=paths["stratum"])
            refs = build_reference_fields(grid, np.array([0.0, 0.5, 1.5, 3.5, 4.0]), thermal_properties_path=THERMAL_PROPERTIES_PATH, layer_transition_xy_km=1.0)

        self.assertEqual(set(refs.layer_masks), set(THERMAL_LAYER_NAMES))
        for name in SEDIMENT_STRATUM_NAMES:
            self.assertNotIn(name, refs.layer_masks)
        stacked = np.stack(list(refs.layer_masks.values()), axis=0)
        np.testing.assert_allclose(np.sum(stacked, axis=0), np.ones(refs.K0_ref.shape))
        self.assertTrue(np.all(stacked >= 0.0))
        self.assertTrue(np.all(stacked <= 1.0))
        self.assertTrue(np.all(np.isfinite(refs.K0_ref)))
        self.assertTrue(np.all(np.isfinite(refs.c_ref)))
        self.assertTrue(np.all(np.isfinite(refs.A_ref)))

    def test_horizontal_transition_does_not_blend_internal_stratum_boundaries_vertically(self) -> None:
        payload = {
            "layers": {
                "sediment": {"A": 1.0, "K0": 2.0, "c": 0.0},
                "upper_crust": {"A0": 2.0, "D": 10.0, "K0": 9.0, "c": 0.001},
                "middle_crust": {"A": 0.7, "K0": 2.7, "c": 1e-4},
                "lower_crust": {"A": 0.2, "K0": 2.9, "c": 1e-4},
                "mantle": {"A": 0.03, "K0": 3.5, "c": 0.0},
            },
            "sedments": {
                name: {"default": {"A": 10.0 + i, "K0": 20.0 + i, "c": 0.01 * i}}
                for i, name in enumerate(SEDIMENT_STRATUM_NAMES)
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_layergrid_csvs(root, nx=3, ny=3, include_stratum=True)
            thermal_path = root / "thermal_properties.json"
            thermal_path.write_text(json.dumps(payload), encoding="utf-8")
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1, stratum_csv=paths["stratum"])
            refs = build_reference_fields(grid, np.array([0.05, 0.1, 0.15]), thermal_properties_path=thermal_path, layer_transition_xy_km=1.0)

        self.assertAlmostEqual(float(refs.K0_ref[1, 1, 1]), 21.0)
        self.assertGreater(float(refs.layer_masks["sediment"][1, 1, 1]), 0.99)

    def test_build_reference_fields_accepts_custom_thermal_properties(self) -> None:
        payload = {
            "layers": {
                "sediment": {"A": 1.0, "K0": 2.1, "c": 0.0},
                "upper_crust": {"A0": 2.0, "D": 12.0, "K0": 2.2, "c": 0.0},
                "middle_crust": {"A": 0.7, "K0": 2.7, "c": 1e-4},
                "lower_crust": {"A": 0.2, "K0": 2.9, "c": 1e-4},
                "mantle": {"A": 0.03, "K0": 3.5, "c": 0.0},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_layergrid_csvs(root, nx=2, ny=2)
            thermal_path = root / "thermal_properties.json"
            thermal_path.write_text(json.dumps(payload), encoding="utf-8")
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)
            refs = build_reference_fields(grid, np.array([0.0, 0.5, 2.0, 15.0, 35.0]), thermal_properties_path=thermal_path)

        self.assertAlmostEqual(float(refs.K0_ref[0, 0, 0]), 2.1)
        self.assertAlmostEqual(float(refs.K0_ref[0, 0, 2]), 2.2)
        self.assertAlmostEqual(float(refs.A_ref[0, 0, 0]), 1.0e-6)

    def test_build_reference_fields_rejects_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_layergrid_csvs(Path(tmp), nx=2, ny=2)
            grid = LayerGrid(paths["grid"], paths["layer"], stride=1)
            with self.assertRaises(ValueError):
                build_reference_fields(grid, np.array([0.0, 1.0]), thermal_properties_path=THERMAL_PROPERTIES_PATH)
            with self.assertRaises(ValueError):
                build_reference_fields(grid, np.array([0.0, 2.0, 1.0]), thermal_properties_path=THERMAL_PROPERTIES_PATH)
            grid.moho[0, 0] = np.nan
            with self.assertRaises(ValueError):
                build_reference_fields(grid, np.array([0.0, 1.0, 2.0]), thermal_properties_path=THERMAL_PROPERTIES_PATH)



if __name__ == "__main__":
    unittest.main()
