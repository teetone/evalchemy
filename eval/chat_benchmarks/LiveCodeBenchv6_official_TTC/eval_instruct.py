from eval.chat_benchmarks.LiveCodeBenchv6_official.eval_instruct import LiveCodeBenchV6OfficialBenchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark


LiveCodeBenchV6OfficialTTCBenchmark = make_ttc_benchmark(LiveCodeBenchV6OfficialBenchmark)
LiveCodeBenchV6OfficialTTCBenchmark.__module__ = __name__


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
LiveCodeBenchV6OfficialTTCBenchmark._build_messages = _build_messages_lcb
