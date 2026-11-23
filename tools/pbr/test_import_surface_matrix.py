import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import import_surface_matrix as matrix


def manifest(label="baseline-a"):
    return {
        "label": label,
        "width": 360,
        "height": 270,
        "frames": 32,
        "fixed_seed": 5,
        "adaptive": False,
        "render_mode": "preseed-single-render",
        "fixtures_sha256": {f"fixture-{i}": f"{i:064x}" for i in range(20)},
    }


class ImportSurfaceHarnessTests(unittest.TestCase):
    def test_explicit_label_selection_stays_narrow(self):
        self.assertEqual(matrix.selected_labels(["baseline-a"]), ["baseline-a"])
        self.assertEqual(matrix.selected_labels(["candidate", "baseline-b"]), ["candidate", "baseline-b"])
        self.assertEqual(matrix.selected_labels(None), ["baseline-a"])

    def test_cross_label_contract_accepts_identical_fixtures_and_settings(self):
        result = matrix.validate_comparison_settings(manifest(), manifest("candidate"))
        self.assertTrue(result["fixture_hashes_identical"])
        self.assertTrue(result["single_preseeded_render"])
        self.assertEqual(result["frames"], 32)

    def test_cross_label_contract_rejects_fixture_seed_adaptive_and_render_mode_changes(self):
        for key in ("fixed_seed", "adaptive", "render_mode", "fixtures_sha256"):
            with self.subTest(key=key):
                other = manifest("candidate")
                other[key] = True if key == "adaptive" else {}
                with self.assertRaises(ValueError):
                    matrix.validate_comparison_settings(manifest(), other)

    def test_renderer_report_proves_preseeded_single_render(self):
        matrix.validate_renderer_report({"frames": 32, "preseed": 5, "renderPasses": 1}, 5, 32)
        for report in (
            {"frames": 32, "preseed": 5, "renderPasses": 2},
            {"frames": 32, "preseed": 5, "renderPasses": 0},
            {"frames": 32, "renderPasses": 1},
            {"frames": 64, "preseed": 5, "renderPasses": 1},
        ):
            with self.subTest(report=report), self.assertRaises(ValueError):
                matrix.validate_renderer_report(report, 5, 32)

    def test_analyse_requires_a_complete_selected_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                matrix.load_manifests(Path(temporary))

    def test_factor_only_fixture_has_no_image_or_texture_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            phase = Path(temporary)
            path = phase / "factor-only.gltf"
            matrix.write_gltf(
                path,
                [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                [(0, 0), (1, 0), (0, 1)],
                [(1, 0, 0, 1)] * 3,
                {"name": "constants", "pbrMetallicRoughness": {
                    "baseColorFactor": [0.16, 0.42, 0.78, 1],
                    "roughnessFactor": 0.29, "metallicFactor": 0.37,
                }},
            )
            document = json.loads(path.read_text())
            self.assertNotIn("images", document)
            self.assertNotIn("textures", document)
            self.assertIn("factor-only-faceted", matrix.SCENE_NAMES)
            args = matrix.scene_command_args(phase, "factor-only-faceted")
            self.assertIn(f"model={phase / 'fixtures' / 'factor-only.gltf'}", args)

    def test_zero_exit_cannot_accept_stale_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            phase = Path(temporary)
            (phase / "fixtures").mkdir()
            executable = phase / "renderer.exe"
            executable.write_bytes(b"not executed")
            name = matrix.SCENE_NAMES[0]
            folder = phase / "renders" / "baseline-a" / name
            folder.mkdir(parents=True)
            stale = [folder / f"{name}.png", folder / f"{name}.rgba", folder / f"{name}.report.json"]
            for path in stale:
                path.write_bytes(b"stale")
            completed = subprocess.CompletedProcess([], 0, json.dumps({
                "frames": 32, "preseed": 5, "renderPasses": 1,
            }), "")
            with mock.patch.object(matrix.subprocess, "run", return_value=completed):
                with self.assertRaisesRegex(RuntimeError, "did not produce both PNG and RGBA"):
                    matrix.capture_label(executable, "baseline-a", phase, phase, 360, 270, 32)
            for path in stale:
                self.assertFalse(path.exists(), "stale output remained at the capture destination")
            failed = json.loads((phase / "reports" / "capture-baseline-a.json").read_text())
            self.assertEqual(failed["status"], "failed")
            quarantined = failed["captures"][0]["quarantined_outputs"]
            self.assertEqual(len(quarantined), 3)
            for path in quarantined:
                self.assertEqual(Path(path).read_bytes(), b"stale")


if __name__ == "__main__":
    unittest.main()
