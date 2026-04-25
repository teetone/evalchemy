from eval.chat_benchmarks.HMMT.eval_instruct import HMMTBenchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

HMMTTTCBenchmark = make_ttc_benchmark(HMMTBenchmark)
HMMTTTCBenchmark.__module__ = __name__
