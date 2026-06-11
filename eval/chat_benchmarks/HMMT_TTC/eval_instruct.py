import os, sys
# Add HMMT dir to sys.path so matharena package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HMMT"))

from eval.chat_benchmarks.HMMT.eval_instruct import HMMTBenchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

HMMTTTCBenchmark = make_ttc_benchmark(HMMTBenchmark)
HMMTTTCBenchmark.__module__ = __name__
