r"""tools/perf_gate.py with its jobs sent through main's shared queue (D:\glassforms\native\tools\gpu_run.py),
not this copy's. Same arguments as perf_gate.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_gate

perf_gate.HERE = r'D:\glassforms\native\tools'
perf_gate.main(sys.argv)
