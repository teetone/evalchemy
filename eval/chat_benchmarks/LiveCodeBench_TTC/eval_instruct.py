from eval.chat_benchmarks.LiveCodeBench.eval_instruct import LiveCodeBenchBenchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark


LiveCodeBenchTTCBenchmark = make_ttc_benchmark(LiveCodeBenchBenchmark)
LiveCodeBenchTTCBenchmark.__module__ = __name__


# LiveCodeBench builds prompts based on is_stdin flag
def _build_messages_lcb(self, example):
    if example["is_stdin"]:
        prompt_text = (
            "Generate an executable Python function generated from the given prompt. "
            "The function should take stdin as input and print the output. "
            "Simply call the function after the definition."
            + example["prompt"]
        )
    else:
        prompt_text = (
            "Generate an executable Python function generated from the given prompt. "
            "Return the function body without invoking it at the final solution."
            + example["prompt"]
        )
    return [{"role": "user", "content": prompt_text}]
LiveCodeBenchTTCBenchmark._build_messages = _build_messages_lcb
