import json
import logging
import os
import sys
import traceback
import warnings
from typing import Any, Dict, List, Optional

import sympy
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import last_boxed_only_string, remove_boxed
from sympy.parsing.latex import parse_latex

from eval.task import BaseBenchmark

PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


def _parse(x: str) -> list:
    """Parse a latex string into sympy expressions. Uses LiveBench's approach:
    lark backend first, then default backend as fallback."""
    try:
        import lark

        try:
            parsed = parse_latex(x, backend="lark")
        except Exception:
            try:
                parsed = parse_latex(x.replace("\\\\", "\\"), backend="lark")
            except Exception:
                try:
                    parsed = parse_latex(x)
                except Exception:
                    return []

        if isinstance(parsed, lark.Tree):
            return parsed.children
        return [parsed]
    except ImportError:
        # lark not installed, use default backend
        try:
            return [parse_latex(x)]
        except Exception:
            return []


def _is_equiv(x1: str, x2: str) -> bool:
    """Symbolic equivalence check using sympy. Based on LiveBench's is_equiv:
    parses both strings as latex, computes diff, checks if simplify(diff) == 0
    or |simplify(diff)| < 0.001."""
    try:
        parsed_x1s = _parse(x1)
        parsed_x2s = _parse(x2)

        if not parsed_x1s or not parsed_x2s:
            return False

        for p1 in parsed_x1s:
            for p2 in parsed_x2s:
                try:
                    diff = p1 - p2
                except Exception:
                    continue

                try:
                    if sympy.simplify(diff) == 0:
                        return True
                except Exception:
                    pass

                try:
                    if sympy.Abs(sympy.simplify(diff)) < 0.001:
                        return True
                except Exception:
                    pass

        return False
    except Exception as e:
        warnings.warn(f"Failed comparing {x1} and {x2}: {e}")
        return False


class OlympiadBenchPhysicsBenchmark(BaseBenchmark):
    """
    OlympiadBench Physics Benchmark for evaluating physics reasoning of LLMs.
    Link: https://huggingface.co/datasets/Hothan/OlympiadBench

    233 text-only, English, open-ended physics competition problems.
    Uses sympy-based symbolic equivalence checker (based on LiveBench's approach)
    which handles decimal/fraction equivalence, scientific notation, and
    numerical tolerance (< 0.001 absolute difference after simplification).
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
            example["model_answer"] = self._extract_answer(output)

        return {"examples": examples}

    def _extract_answer(self, output: str) -> str:
        try:
            return remove_boxed(last_boxed_only_string(output))
        except Exception:
            return ""

    def _check_answer(self, expected: str, predicted: str) -> bool:
        if not predicted:
            return False
        return _is_equiv(expected, predicted)

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
