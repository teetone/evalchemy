from eval.chat_benchmarks.HLE.eval_instruct import HLESubsetBenchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

HLESubsetTTCBenchmark = make_ttc_benchmark(HLESubsetBenchmark)
HLESubsetTTCBenchmark.__module__ = __name__
