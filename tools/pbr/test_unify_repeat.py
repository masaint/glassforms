from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image

import unify_repeat as repeat


def hdr_image(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 4.0, size=(6, 8, 4)).astype("<f4")


def png_image():
    return np.arange(6 * 8 * 4, dtype=np.uint8).reshape(6, 8, 4)


def write_capture(root, scene, label, png, hdr):
    files = repeat.capture_files(root, scene, label)
    files["png"].parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(png, "RGBA").save(files["png"])
    height, width = hdr.shape[:2]
    files["hdr"].write_bytes(struct.pack("<II", width, height) + hdr.astype("<f4").tobytes())


def one_ulp(hdr, y=2, x=3, c=1):
    out = hdr.copy()
    out[y, x, c] = np.nextafter(out[y, x, c], np.float32(np.inf))
    return out


class DiffTests(unittest.TestCase):
    def test_identical_arrays(self):
        stats = repeat.diff_arrays(hdr_image(), hdr_image())
        self.assertTrue(stats["identical"])
        self.assertEqual(stats["differingPixels"], 0)
        self.assertEqual(stats["maxAbs"], 0.0)

    def test_one_ulp_is_a_difference_with_statistics(self):
        a = hdr_image()
        stats = repeat.diff_arrays(a, one_ulp(a))
        self.assertFalse(stats["identical"])
        self.assertEqual(stats["differingPixels"], 1)
        self.assertEqual(stats["differingValues"], 1)
        self.assertGreater(stats["maxAbs"], 0.0)
        self.assertLess(stats["maxRel"], 1e-6)
        self.assertFalse(repeat.within(stats, None))
        self.assertTrue(repeat.within(stats, 1e-6))
        self.assertFalse(repeat.within(stats, 1e-9))

    def test_nan_at_the_same_place_is_identical(self):
        a = hdr_image()
        a[1, 1, 0] = np.nan
        b = a.copy()
        b[1, 1, 0] = np.float32(np.nan) * -1   # another NaN bit pattern
        self.assertTrue(repeat.diff_arrays(a, b)["identical"])

    def test_nan_against_a_number_differs_and_no_tolerance_covers_it(self):
        a = hdr_image()
        b = a.copy()
        b[1, 1, 0] = np.nan
        stats = repeat.diff_arrays(a, b)
        self.assertFalse(stats["identical"])
        self.assertEqual(stats["nanMismatches"], 1)
        self.assertEqual(stats["differingPixels"], 1)
        self.assertTrue(np.isfinite(stats["maxAbs"]))
        self.assertFalse(repeat.within(stats, 1.0))

    def test_signed_zero_is_not_byte_identical(self):
        a = np.zeros((2, 2, 4), "<f4")
        b = a.copy()
        b[0, 0, 0] = -0.0
        self.assertFalse(repeat.diff_arrays(a, b)["identical"])

    def test_shape_mismatch(self):
        stats = repeat.diff_arrays(hdr_image(), hdr_image()[:, :7])
        self.assertTrue(stats["shapeMismatch"])
        self.assertFalse(repeat.within(stats, 1.0))

    def test_tolerances_are_by_scene_name(self):
        self.assertEqual(repeat.parse_tolerances(["glass_thomas=1e-6"], ["glass_thomas"]),
                         {"glass_thomas": 1e-6})
        for bad in (["nope=1e-6"], ["glass_thomas"], ["glass_thomas=-1"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                repeat.parse_tolerances(bad, ["glass_thomas"])


class SceneVerdictTests(unittest.TestCase):
    def run_scene(self, candidate_png, candidate_hdr, baseline_b=None, tolerance=None):
        png, hdr = png_image(), hdr_image()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_capture(root, "s", "baseline-a", png, hdr)
            write_capture(root, "s", "candidate", candidate_png, candidate_hdr)
            b_png, b_hdr = baseline_b or (png, hdr)
            write_capture(root, "s", "baseline-b", b_png, b_hdr)
            return repeat.evaluate_scene(root, "s", tolerance)

    def test_identical_passes(self):
        result = self.run_scene(png_image(), hdr_image())
        self.assertEqual(result["verdict"], "IDENTICAL")
        self.assertIn(result["verdict"], repeat.PASS)

    def test_one_ulp_hdr_fails_by_default_and_passes_with_tolerance(self):
        hdr = one_ulp(hdr_image())
        self.assertEqual(self.run_scene(png_image(), hdr)["verdict"], "DIFFERENT")
        self.assertEqual(self.run_scene(png_image(), hdr, tolerance=1e-6)["verdict"], "WITHIN_TOLERANCE")

    def test_png_change_fails_even_with_hdr_tolerance(self):
        png = png_image()
        png[0, 0, 0] ^= 1
        result = self.run_scene(png, hdr_image(), tolerance=1e-6)
        self.assertEqual(result["verdict"], "DIFFERENT")
        self.assertEqual(result["candidate_vs_baselineA"]["png"]["differingPixels"], 1)

    def test_depth_change_is_reported_apart_from_colour(self):
        result = self.run_scene(png_image(), one_ulp(hdr_image(), c=3))
        self.assertEqual(result["verdict"], "DIFFERENT")
        cand = result["candidate_vs_baselineA"]
        self.assertEqual(cand["depth"]["differingPixels"], 1)
        self.assertTrue(cand["colour"]["identical"])

    def test_shape_mismatch_fails(self):
        hdr = hdr_image()[:5]
        self.assertEqual(self.run_scene(png_image(), hdr, tolerance=1.0)["verdict"], "SHAPE_MISMATCH")

    def test_baseline_a_against_b_mismatch_is_nondeterministic(self):
        result = self.run_scene(png_image(), hdr_image(), baseline_b=(png_image(), one_ulp(hdr_image())),
                                tolerance=1e-6)
        self.assertEqual(result["verdict"], "NONDETERMINISTIC")
        self.assertNotIn(result["verdict"], repeat.PASS)

    def test_missing_and_truncated_files_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_capture(root, "s", "baseline-a", png_image(), hdr_image())
            write_capture(root, "s", "candidate", png_image(), hdr_image())
            self.assertEqual(repeat.evaluate_scene(root, "s")["verdict"], "MISSING")
            write_capture(root, "s", "baseline-b", png_image(), hdr_image())
            hdr = repeat.capture_files(root, "s", "candidate")["hdr"]
            hdr.write_bytes(hdr.read_bytes()[:-4])
            self.assertEqual(repeat.evaluate_scene(root, "s")["verdict"], "UNREADABLE")

    def test_default_scenes_are_the_gate_scenes(self):
        scenes = repeat.gate_scenes()
        self.assertIn("glass_thomas", scenes)
        self.assertIn("rough_assembly", scenes)


if __name__ == "__main__":
    unittest.main()
