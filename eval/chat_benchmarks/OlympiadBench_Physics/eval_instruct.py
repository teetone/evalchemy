import json
import logging
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark
from matharena.parser import check_answers, extract_answer, parse_answer

PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class OlympiadBenchPhysicsBenchmark(BaseBenchmark):
    """
    OlympiadBench Physics Benchmark for evaluating physics reasoning of LLMs.
    Link: https://huggingface.co/datasets/Hothan/OlympiadBench

    233 text-only, English, open-ended physics competition problems.
    Uses matharena's sympy-based symbolic equivalence checker (same as HMMT)
    instead of hendrycks_math's string-based is_equiv, since physics answers
    often involve scientific notation, decimal/fraction equivalence, and
    reordered terms that require symbolic comparison.
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
            # Use matharena's sympy-based answer extraction
            parsed, _ = extract_answer(output, strict_parsing=False, parse=True)
            example["model_answer"] = parsed

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = 0
        for example in examples:
            gold, _ = parse_answer(str(example["answer"]))
            model = example["model_answer"]
            is_correct = check_answers(model, gold)
            example["is_correct"] = is_correct
            solved += is_correct

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
