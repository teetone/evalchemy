from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

MATH500TTCBenchmark = make_ttc_benchmark(MATH500Benchmark)
MATH500TTCBenchmark.__module__ = __name__
