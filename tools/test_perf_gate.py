import csv
from pathlib import Path
import tempfile
import unittest

from perf_gate import restored_sample, trace_summary


class PerformanceTraceTests(unittest.TestCase):
    def test_held_updates_contribute_time_without_completing_samples(self):
        def row(phase, update, half, passes, native):
            w, h = (3840, 2160) if native else (1920, 1080)
            return [phase, update, 2, 1, 1, 4, 3, 'None', 1, 0, 6,
                    w, h, 0, 0, 0, 0, half, 768, passes]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'session.tsv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream, delimiter='\t')
                for phase in ('turn', 'play'):
                    writer.writerows(row(phase, i, 0, 1, False) for i in range(5))
                writer.writerows([
                    row('settle', 0, 0, 0, True),   # staged build, previous image still visible
                    row('settle', 1, 1, 1, True),
                    row('settle', 2, 2, 0, True),   # no second half was actually submitted
                    row('settle', 3, 2, 1, True),
                    row('settle', 4, 0, 0, False),  # idle, last full-quality sample remains
                ])
            settle = trace_summary(path, 5, 3840, 2160)['settle']
            recovery = restored_sample(path, settle['finalQuality'])

        self.assertEqual(settle['heldUpdates'], 3)
        self.assertEqual(settle['tracePasses'], 2)
        self.assertEqual(settle['tracedUpdateP80Ms'], 2.0)
        self.assertEqual(settle['firstNativeUpdate'], 2)
        self.assertEqual(settle['nativePairP80Ms'], 6.0)
        self.assertEqual(settle['finalQuality']['width'], 3840)
        self.assertEqual(recovery['firstCompleteUpdate'], 4)
        self.assertEqual(recovery['msToFirstCompleteSample'], 8.0)
        self.assertTrue(recovery['endsAtReferenceQuality'])
