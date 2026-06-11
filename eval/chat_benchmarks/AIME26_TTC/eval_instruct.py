from eval.chat_benchmarks.AIME26.eval_instruct import AIME26Benchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

AIME26TTCBenchmark = make_ttc_benchmark(AIME26Benchmark)
AIME26TTCBenchmark.__module__ = __name__
