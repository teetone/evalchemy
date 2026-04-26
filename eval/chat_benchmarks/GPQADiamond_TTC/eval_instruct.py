from eval.chat_benchmarks.GPQADiamond.eval_instruct import GPQADiamondBenchmark
from eval.chat_benchmarks.ttc_utils import make_ttc_benchmark


GPQADiamondTTCBenchmark = make_ttc_benchmark(GPQADiamondBenchmark)
GPQADiamondTTCBenchmark.__module__ = __name__


# GPQADiamond needs preprocessing (shuffle options) and custom prompt (problem + options)
_original_preprocess = GPQADiamondTTCBenchmark._preprocess_examples
def _preprocess_gpqa(self, examples):
    for example in examples:
        if "multiple_choice_string" not in example:
            mc_string, correct = self.generate_multiple_choice_answers(example)
            example["multiple_choice_string"] = mc_string
            example["answer"] = correct
GPQADiamondTTCBenchmark._preprocess_examples = _preprocess_gpqa
