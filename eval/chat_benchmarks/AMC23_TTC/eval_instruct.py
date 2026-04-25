from eval.chat_benchmarks.AMC23.eval_instruct import AMC23Benchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark

AMC23TTCBenchmark = make_ttc_benchmark(AMC23Benchmark)
AMC23TTCBenchmark.__module__ = __name__
