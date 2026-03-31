import json
import logging
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class OlympiadBenchPhysicsBenchmark(BaseBenchmark):
    """
    OlympiadBench Physics Benchmark for evaluating physics reasoning of LLMs.
    Link: https://huggingface.co/datasets/Hothan/OlympiadBench

    233 text-only, English, open-ended physics competition problems.
    Follows the same evaluation logic as OlympiadBench (math) using
    hendrycks_math answer extraction and is_equiv scoring.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/OlympiadBench_Physics/data/olympiadbench_physics.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.max_new_tokens = max_tokens
        self.seed = seed

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        examples = self.load_questions()

        all_instances = []
        for idx, example in enumerate(examples):
            messages = [
                {"role": "user", "content": PROMPT.format(problem=example["problem"])},
            ]

            templated_messages = self._prepare_messages(messages, model)

            instance = Instance(
                "generate_until",
                example,
                (
                    templated_messages,
                    {
                        "do_sample": False,
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": 0.7,
                        "seed": self.seed,
                    },
                ),
                idx,
            )

            all_instances.append(instance)

        self.logger.info("Generating responses for OlympiadBench Physics...")
        outputs = self.compute(model, all_instances)

        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = self.extract_answer(output)

        return {"examples": examples}

    def _check_answer(self, expected: str, predicted: str) -> bool:
        return is_equiv(expected, predicted)

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = sum(self._check_answer(str(example["answer"]), example["model_answer"]) for example in examples)

        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": solved / total,
            }
        )

        return results

    def load_questions(self) -> List[Dict[str, str]]:
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]

        if self.debug:
            questions = questions[:2]
            self.logger.info(f"Debug mode enabled. Using only {len(questions)} questions.")

        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        try:
            answer = remove_boxed(last_boxed_only_string(output))
            return answer
        except:
            return ""
