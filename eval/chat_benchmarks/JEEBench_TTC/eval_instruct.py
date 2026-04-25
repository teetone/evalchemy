from eval.chat_benchmarks.JEEBench.eval_instruct import JEEBenchBenchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

JEEBenchTTCBenchmark = make_ttc_benchmark(JEEBenchBenchmark)
JEEBenchTTCBenchmark.__module__ = __name__
