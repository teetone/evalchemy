import json
import logging
import re
import traceback
import warnings
from typing import Any, Dict, List, Optional

import sympy
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from sympy.parsing.latex import parse_latex

from eval.task import BaseBenchmark

PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


def _strip_var_assignment(x: str) -> str:
    """Strip leading variable assignment like 'v=' or 'E_{min}=' from an expression."""
    # Match patterns like: x=, v=, F=, E_{min}=, t_{r \rightarrow 0}=, A(\mu, \Omega, L)=
    m = re.match(r'^[A-Za-z_{}\\,\s\(\)]+\s*=\s*', x)
    if m:
        stripped = x[m.end():]
        if stripped:
            return stripped
    return x


def _normalize_latex(x: str) -> str:
    """Normalize physics latex for better parsing. Replaces common physics
    symbols with sympy-friendly names and cleans up formatting."""
    x = x.strip()
    # Strip \text{...} and \mathrm{...} (units, labels)
    x = re.sub(r'\\text\{[^}]*\}', '', x)
    x = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', x)
    # Remove \left \right \, \; \! \quad
    x = re.sub(r'\\(left|right|,|;|!|quad|qquad)\b', '', x)
    # \dfrac -> \frac, \tfrac -> \frac
    x = x.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # Remove \approx (treat as =)
    x = x.replace('\\approx', '=')
    # Strip %, $
    x = x.replace('\\%', '').replace('%', '').replace('$', '')
    return x.strip()


def _parse(x: str) -> list:
    """Parse a latex string into sympy expressions. Uses lark backend first,
    then default backend as fallback."""
    x = _normalize_latex(x)

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
        try:
            return [parse_latex(x)]
        except Exception:
            return []


def _check_parsed(parsed_x1s: list, parsed_x2s: list, rel_tol: float) -> bool:
    """Check if any pair of parsed expressions are equivalent."""
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

            # Relative tolerance check for numerical values
            try:
                v1 = float(sympy.N(p1))
                v2 = float(sympy.N(p2))
                denom = max(abs(v1), abs(v2))
                if denom > 0 and abs(v1 - v2) / denom < rel_tol:
                    return True
            except Exception:
                pass

    return False


def _is_equiv(x1: str, x2: str, rel_tol: float = 0.001) -> bool:
    """Symbolic equivalence check using sympy. Parses both strings as latex,
    computes diff, checks if simplify(diff) == 0 or within relative tolerance.

    Also tries stripping variable assignments (e.g. 'v=' prefix) from both
    sides before comparing.

    rel_tol: relative tolerance (0.001 = 0.1%). Two values a, b are considered
    equal if |a - b| / max(|a|, |b|) < rel_tol.
    """
    try:
        # Try all combinations: original and variable-stripped versions
        variants_x1 = [x1]
        stripped_x1 = _strip_var_assignment(x1)
        if stripped_x1 != x1:
            variants_x1.append(stripped_x1)

        variants_x2 = [x2]
        stripped_x2 = _strip_var_assignment(x2)
        if stripped_x2 != x2:
            variants_x2.append(stripped_x2)

        for v1 in variants_x1:
            for v2 in variants_x2:
                if _check_parsed(_parse(v1), _parse(v2), rel_tol):
                    return True

        return False
    except Exception:
        return False


def _extract_all_boxed(text: str) -> list[str]:
    """Extract all \\boxed{...} contents from text, handling nested braces."""
    results = []
    idx = 0
    while True:
        idx = text.find("\\boxed{", idx)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    content = text[start:i].strip()
                    if content:
                        results.append(content)
                    break
                depth -= 1
        idx = start
    return results


def _deduplicate(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class OlympiadBenchPhysicsBenchmark(BaseBenchmark):
    """
    OlympiadBench Physics Benchmark for evaluating physics reasoning of LLMs.
    Link: https://huggingface.co/datasets/Hothan/OlympiadBench

    236 text-only, English, open-ended physics competition problems.
    Uses sympy-based symbolic equivalence checker with parse_latex.

    Multi-part answers: Some problems expect multiple sub-answers. The checker
    extracts ALL \\boxed{} contents from the model output and checks if each
    expected sub-answer matches any of them. A problem is scored correct only
    if ALL expected parts are matched.
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
            example["model_answers"] = _deduplicate(_extract_all_boxed(output))

        return {"examples": examples}

    def _check_answer(self, expected_parts: list[str], predicted_parts: list[str]) -> bool:
        """Check if ALL expected answer parts match some predicted part.

        Each expected part must match at least one predicted part via symbolic
        equivalence. A predicted part can only match one expected part.
        """
        if not predicted_parts:
            return False

        # For each expected part, find a matching predicted part
        used = set()
        for exp in expected_parts:
            found = False
            for i, pred in enumerate(predicted_parts):
                if i in used:
                    continue
                if _is_equiv(exp, pred):
                    used.add(i)
                    found = True
                    break
            if not found:
                return False
        return True

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = 0
        for example in examples:
            expected_parts = example["answers"]
            predicted_parts = example.get("model_answers", [])
            is_correct = self._check_answer(expected_parts, predicted_parts)
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
