"""
Test-Time Compute (TTC) utilities: UQ validation prompts, decision extraction,
and a factory for creating TTC benchmark wrappers.

Ports validation logic from virtual-world-data/sdg/validation.py and sdg/prompts.py.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Type

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

logger = logging.getLogger(__name__)

# =============================================================================
# Config defaults
# =============================================================================

DEFAULT_N_CANDIDATES = 8
N_VALIDATION_VOTES = 3
VAL_TEMPERATURE = 0.6
VAL_TOP_P = 0.95
VAL_MAX_TOKENS = 32768

# =============================================================================
# UQ Validation Prompts (verbatim from sdg/prompts.py)
# =============================================================================

CYCLE_QUESTION_GENERATION_PROMPT = """Given an answer, please generate the most likely question that would have prompted this answer. Focus on inferring the core question that this answer is addressing. Output only the inferred question, without any additional explanation.

Answer:
{answer}

Inferred Question:"""

CYCLE_COMPARISON_PROMPT = """You are evaluating whether an answer is relevant to the original question and touches the core of the question by comparing the original question with an inferred question derived only from the answer.

Original Question: {original_question}
Inferred Question: {inferred_question}

Compare the two questions and determine:
1. If the original question and inferred question are asking about the same core topic
2. If the original question and inferred question share the same key elements and requirements
3. If answering one question would effectively address the other question

After your analysis, provide your decision: [[Y]] if the questions are semantically equivalent and address the same core problem, or [[N]] if they are asking about different things."""

FACTUAL_ERROR_PROMPT = """Please act as an impartial judge and carefully analyze the following answer for any factual errors, logical flaws, or misleading information.

Question: {question}
Answer: {answer}

Consider the credibility of the claims made in the answer and determine if they align with established knowledge. Evaluate:
1. Are there any incorrect facts, dates, numbers, formulas, or claims?
2. Is there any faulty logic, reasoning, or problem-solving approach?
3. Are there any misleading, incomplete, or ambiguous explanations?
4. Does the answer introduce any misconceptions or propagate common errors?

Minor typos or grammatical errors are acceptable. But be strict about any factual error, calculation error, or logical flaw. When unsure, lean toward accepting statements unless they contain clear errors.

After a thorough analysis, provide your decision: [[Y]] if the answer has no factual errors or major flaws, or [[N]] if it contains important factual errors or logical flaws that would mislead the user."""

TOTAL_CORRECTNESS_PROMPT = """Please act as an impartial judge and evaluate whether the response is completely correct in both process and conclusion.

Question: {question}
Answer: {answer}

Consider correctness, usefulness, completeness and depth in your assessment. Consider whether this answer completely solves the question.

You should rely on your own reasoning to form a reference solution and compare the answer to your reasoning.

Begin your evaluation by giving a brief summary of your thoughts on the response. Focus on whether it is accurate, addresses the question well, and is reasonably detailed. Be precise about any errors or gaps you notice.

Notes:
1. If the answer is partial, high-level, or just states that this is an open problem, you should not accept it.
2. If the answer lacks details or is not comprehensive, you should not accept it.
3. If the answer contains any errors, you should not accept it.
4. You should only accept the answer if it is at least 95% correct and solves the question.

After providing your explanation, decide whether this answer is correct. Think twice about whether this answer solves the question.
Format: Accepted: [[Y]] if you accept the answer or Accepted: [[N]] if you do not accept."""


# =============================================================================
# Decision extraction (from sdg/validation.py)
# =============================================================================

DECISION_RE = re.compile(r"\[\[\s*([YN])\s*\]\]", re.IGNORECASE)


def extract_decision_robust(text: str) -> bool:
    """Priority: [[Y]]/[[N]] -> standalone Y/N -> YES/NO -> False."""
    if not text:
        return False
    text_upper = text.upper()

    matches = list(DECISION_RE.finditer(text_upper))
    if matches:
        return matches[-1].group(1) == "Y"

    standalone = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text_upper)
    if standalone:
        return standalone[-1] == "Y"

    yesno = re.findall(r"\b(YES|NO)\b", text_upper)
    if yesno:
        return yesno[-1] == "YES"

    return False


def check_unanimous(vote_texts: list[str]) -> bool:
    if not vote_texts:
        return False
    return all(extract_decision_robust(t) for t in vote_texts)


# =============================================================================
# Text helpers (from sdg/validation.py)
# =============================================================================

def extract_final_answer(answer: str) -> str:
    """Content after </think> (or full text if no tag)."""
    if "</think>" in answer:
        return answer.split("</think>")[-1].strip()
    return answer


def clean_inferred_question(text: str) -> str:
    """After </think>, first line only."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip().split("\n")[0].strip()
    else:
        text = text.strip().split("\n")[0].strip()
    return text


# =============================================================================
# UQ validation via evalchemy model interface
# =============================================================================

def _generate_validation(
    model: LM,
    benchmark: BaseBenchmark,
    prompts: list[str],
    num_outputs: int,
    seed: list[int],
) -> list[list[str]]:
    """Generate validation responses through evalchemy's compute() interface.

    Args:
        model: evalchemy LM model
        benchmark: BaseBenchmark instance (for _prepare_messages and compute)
        prompts: list of prompt strings
        num_outputs: number of outputs per prompt (1 for single, N_VALIDATION_VOTES for votes)
        seed: base seed list

    Returns:
        list[list[str]] — outer list per prompt, inner list per output
    """
    all_results: list[list[str]] = []

    for vote_idx in range(num_outputs):
        vote_seed = [s + vote_idx + 100 for s in seed]  # offset to avoid collision with generation seeds
        instances = []
        for idx, prompt_text in enumerate(prompts):
            messages = [{"role": "user", "content": prompt_text}]
            templated = benchmark._prepare_messages(messages, model)
            instance = Instance(
                "generate_until",
                {},
                (
                    templated,
                    {
                        "do_sample": True,
                        "max_new_tokens": VAL_MAX_TOKENS,
                        "temperature": VAL_TEMPERATURE,
                        "seed": vote_seed,
                    },
                ),
                idx,
            )
            instances.append(instance)

        outputs = benchmark.compute(model, instances)

        if not all_results:
            all_results = [[o] for o in outputs]
        else:
            for i, o in enumerate(outputs):
                all_results[i].append(o)

    return all_results


def uq_filter_candidates(
    model: LM,
    benchmark: BaseBenchmark,
    question: str,
    candidates: list[str],
    seed: list[int],
) -> tuple[str, int, dict]:
    """Run 3-stage UQ validation on candidates, return first valid.

    Args:
        model: evalchemy LM
        benchmark: BaseBenchmark instance
        question: the original problem/question text
        candidates: list of generated candidate answers
        seed: base seed for validation sampling

    Returns:
        (selected_answer, selected_index, info_dict)
        info_dict contains validation statistics
    """
    info = {
        "n_candidates": len(candidates),
        "n_passed_cycle": 0,
        "n_passed_factual": 0,
        "n_passed_correctness": 0,
        "selected_index": 0,
        "selection_method": "fallback",
    }

    for idx, candidate in enumerate(candidates):
        answer_text = extract_final_answer(candidate)

        # Stage 1: Cycle consistency
        # Step 1a: Generate inferred question
        gen_prompt = CYCLE_QUESTION_GENERATION_PROMPT.format(answer=answer_text)
        inferred_raw = _generate_validation(model, benchmark, [gen_prompt], 1, seed)
        inferred_q = clean_inferred_question(inferred_raw[0][0])

        # Step 1b: Compare original vs inferred
        compare_prompt = CYCLE_COMPARISON_PROMPT.format(
            original_question=question, inferred_question=inferred_q
        )
        vote_outputs = _generate_validation(
            model, benchmark, [compare_prompt], N_VALIDATION_VOTES, seed
        )
        if not check_unanimous(vote_outputs[0]):
            continue
        info["n_passed_cycle"] += 1

        # Stage 2: Factual error check
        factual_prompt = FACTUAL_ERROR_PROMPT.format(question=question, answer=answer_text)
        vote_outputs = _generate_validation(
            model, benchmark, [factual_prompt], N_VALIDATION_VOTES, seed
        )
        if not check_unanimous(vote_outputs[0]):
            continue
        info["n_passed_factual"] += 1

        # Stage 3: Total correctness
        correctness_prompt = TOTAL_CORRECTNESS_PROMPT.format(question=question, answer=answer_text)
        vote_outputs = _generate_validation(
            model, benchmark, [correctness_prompt], N_VALIDATION_VOTES, seed
        )
        if not check_unanimous(vote_outputs[0]):
            continue
        info["n_passed_correctness"] += 1

        # All 3 stages passed — select this candidate
        info["selected_index"] = idx
        info["selection_method"] = "uq_validated"
        return candidate, idx, info

    # No candidate passed all stages — fall back to first candidate
    info["selected_index"] = 0
    info["selection_method"] = "fallback"
    return candidates[0], 0, info


# =============================================================================
# Factory: create TTC wrapper from any existing benchmark
# =============================================================================

def make_ttc_benchmark(
    original_class: Type[BaseBenchmark],
    default_n_candidates: int = DEFAULT_N_CANDIDATES,
) -> Type[BaseBenchmark]:
    """Create a TTC (Test-Time Compute) wrapper class for an existing benchmark.

    The wrapper:
    - Reuses the original benchmark's data loading, prompts, answer extraction, AND scoring
    - Only overrides generate_responses() to do N candidates + UQ filtering
    - Sets n_repeat=1 so the parent's evaluate_responses() scores a single run
    - Output format matches what the parent expects

    Args:
        original_class: The original benchmark class to wrap
        default_n_candidates: Default number of candidates per problem
    """

    class TTCBenchmark(original_class):

        def __init__(self, n_candidates: int = default_n_candidates, **kwargs):
            super().__init__(**kwargs)
            self.n_candidates = n_candidates
            # Force n_repeat=1 — TTC does its own multi-generation + filtering
            if hasattr(self, "n_repeat"):
                self._original_n_repeat = self.n_repeat
                self.n_repeat = 1
            # Storage for prompts captured during parent's generation
            self._captured_prompts: Dict[int, str] = {}
            self._capturing: bool = False

        def compute(self, model: LM, inputs: List[Instance], do_slice: bool = True) -> List[str]:
            """Wrapper around parent compute that captures templated prompts during generation."""
            if self._capturing:
                for inst in inputs:
                    if isinstance(inst.args, tuple) and len(inst.args) >= 1:
                        self._captured_prompts[inst.idx] = inst.args[0]
            return super().compute(model, inputs, do_slice)

        def generate_responses(self, model: LM) -> Dict[str, Any]:
            """Generate N candidates per problem, UQ-filter, return in parent's format."""

            # Step 1: Run parent's generate_responses (with n_repeat=1)
            # This produces one output per example using the benchmark's exact prompt logic.
            # Our compute() wrapper captures the templated prompt for each example idx.
            self._captured_prompts.clear()
            self._capturing = True
            parent_result = original_class.generate_responses(self, model)
            self._capturing = False

            if model.rank != 0:
                return None

            # Step 2: For each example, generate N-1 more candidates and UQ-filter
            examples = parent_result["examples"]

            for prob_idx, example in enumerate(examples):
                question_text = _get_question_text(example)

                # Get the first candidate from parent's generation
                if "model_outputs" in example:
                    first_output = example["model_outputs"][0]
                elif "model_output" in example:
                    first_output = example["model_output"]
                else:
                    continue

                candidates = [first_output]

                # Reuse the exact templated prompt captured during parent's generation
                templated_messages = self._captured_prompts.get(prob_idx)
                if templated_messages is None:
                    logger.warning(f"No captured prompt for example {prob_idx}, skipping TTC")
                    continue

                # Generate N-1 more candidates with different seeds
                for j in range(1, self.n_candidates):
                    seed = [s + j for s in self.seed]
                    instance = Instance(
                        "generate_until",
                        example,
                        (
                            templated_messages,
                            {
                                "do_sample": False,
                                "max_new_tokens": self.max_new_tokens,
                                "temperature": 0.7,
                                "seed": seed,
                            },
                        ),
                        0,
                    )
                    outputs = super().compute(model, [instance])
                    candidates.append(outputs[0])

                # UQ filter: pick first valid candidate
                selected, sel_idx, ttc_info = uq_filter_candidates(
                    model, self, question_text, candidates, self.seed,
                )

                # Replace the example's output with the selected candidate
                # Match the format the parent's evaluate_responses expects
                if "model_outputs" in example:
                    example["model_outputs"] = [selected]
                    example["model_answers"] = [self.extract_answer(selected)]
                elif "model_output" in example:
                    example["model_output"] = selected
                    example["model_answer"] = self.extract_answer(selected)

                example["ttc_info"] = ttc_info
                example["all_candidates"] = candidates

                if (prob_idx + 1) % 5 == 0 or prob_idx == 0:
                    logger.info(
                        f"TTC [{prob_idx + 1}/{len(examples)}] "
                        f"selected candidate {sel_idx}/{self.n_candidates} "
                        f"via {ttc_info['selection_method']}"
                    )

            return parent_result

        # evaluate_responses is NOT overridden — uses parent's exact scoring logic

    # Set class name for TaskManager discovery
    ttc_name = original_class.__name__.replace("Benchmark", "TTCBenchmark")
    TTCBenchmark.__name__ = ttc_name
    TTCBenchmark.__qualname__ = ttc_name
    # __module__ must be set by caller for TaskManager discovery:
    #   ClassName = make_ttc_benchmark(OriginalClass)
    #   ClassName.__module__ = __name__

    return TTCBenchmark


def _get_question_text(example: dict) -> str:
    """Extract the question text from an example for UQ validation."""
    for key in ("problem", "question", "Question", "prompt"):
        if key in example:
            return example[key]
    return str(example)


