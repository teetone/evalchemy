from eval.chat_benchmarks.AIME25.eval_instruct import AIME25Benchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

AIME25TTCBenchmark = make_ttc_benchmark(AIME25Benchmark)
AIME25TTCBenchmark.__module__ = __name__
