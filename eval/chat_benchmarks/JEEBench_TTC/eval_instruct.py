import os, sys
# Add JEEBench dir to sys.path so utils module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "JEEBench"))

from eval.chat_benchmarks.JEEBench.eval_instruct import JEEBenchBenchmark, format_message
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark


JEEBenchTTCBenchmark = make_ttc_benchmark(JEEBenchBenchmark)
JEEBenchTTCBenchmark.__module__ = __name__


# JEEBench uses format_message(question, prompt_library) for per-type prompts.
# self.prompt_library is set during __init__ by prompt_for_boxed_answer(PROMPT_LIBRARY).
def _build_messages_jeebench(self, example):
    return format_message(example, self.prompt_library)
JEEBenchTTCBenchmark._build_messages = _build_messages_jeebench
