from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

AIME24TTCBenchmark = make_ttc_benchmark(AIME24Benchmark)
AIME24TTCBenchmark.__module__ = __name__
