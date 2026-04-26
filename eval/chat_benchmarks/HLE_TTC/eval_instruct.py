from eval.chat_benchmarks.HLE.eval_instruct import HLESubsetBenchmark, format_message
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark


HLESubsetTTCBenchmark = make_ttc_benchmark(HLESubsetBenchmark)
HLESubsetTTCBenchmark.__module__ = __name__


# HLE uses its own format_message(question) which handles system prompt + question text
def _build_messages_hle(self, example):
    return format_message(example)
HLESubsetTTCBenchmark._build_messages = _build_messages_hle
