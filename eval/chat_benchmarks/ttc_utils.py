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
N_VALIDATION_VOTES = 5
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

def _generate_n_candidates(
    model: LM,
    templated_prompt: str,
    n: int,
    max_new_tokens: int,
    temperature: float = 0.7,
) -> list[str]:
    """Generate n diverse candidates for a prompt using vLLM's native n parameter.

    Calls model.model.generate() directly with SamplingParams(n=n) to get
    n completions in a single request. This shares the prompt KV cache and
    guarantees diversity via independent temperature sampling.

    Args:
        model: lm-eval LM model (must be a vLLM model with model.model attribute)
        templated_prompt: the prompt string (already chat-templated)
        n: number of candidates to generate
        max_new_tokens: max generation length
        temperature: sampling temperature (default 0.7)

    Returns:
        list of n generated text strings
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    # Tokenize the prompt the same way lm-eval does
    prompt_tokens = model.tok_encode(templated_prompt)

    # Truncate prompt if it exceeds max_model_len - max_new_tokens,
    # same as lm-eval's generate_until does (left truncation).
    # Use model.max_length property (same as lm-eval line 619).
    max_model_len = getattr(model, 'max_length', None)
    if max_model_len:
        max_ctx_len = max_model_len - max_new_tokens
        if len(prompt_tokens) > max_ctx_len:
            logger.warning(
                f"Prompt length {len(prompt_tokens)} exceeds max context "
                f"({max_ctx_len}={max_model_len}-{max_new_tokens}). Truncating."
            )
            prompt_tokens = prompt_tokens[-max_ctx_len:]

    # Call vLLM directly
    from vllm import TokensPrompt
    outputs = model.model.generate(
        [TokensPrompt(prompt_token_ids=prompt_tokens)],
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    # Extract all n completions from the single request
    return [output.text for output in outputs[0].outputs]


def _generate_validation(
    model: LM,
    benchmark: BaseBenchmark,
    prompts: list[str],
    num_outputs: int,
    seed: list[int],
) -> list[list[str]]:
    """Generate validation responses using vLLM's native n parameter.

    Single vLLM call per prompt with n=num_outputs for diverse votes.

    Args:
        model: evalchemy LM model
        benchmark: BaseBenchmark instance (for _prepare_messages)
        prompts: list of prompt strings
        num_outputs: number of outputs per prompt (1 for single, N_VALIDATION_VOTES for votes)
        seed: base seed list (unused — diversity comes from temperature sampling via n)

    Returns:
        list[list[str]] — outer list per prompt, inner list per output
    """
    all_results: list[list[str]] = []

    for prompt_text in prompts:
        messages = [{"role": "user", "content": prompt_text}]
        templated = benchmark._prepare_messages(messages, model)
        outputs = _generate_n_candidates(
            model, templated, num_outputs, VAL_MAX_TOKENS, VAL_TEMPERATURE,
        )
        all_results.append(outputs)

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
            logger.debug(f"  Candidate {idx}: FAILED cycle consistency")
            continue
        info["n_passed_cycle"] += 1
        logger.debug(f"  Candidate {idx}: passed cycle consistency")

        # Stage 2: Factual error check
        factual_prompt = FACTUAL_ERROR_PROMPT.format(question=question, answer=answer_text)
        vote_outputs = _generate_validation(
            model, benchmark, [factual_prompt], N_VALIDATION_VOTES, seed
        )
        if not check_unanimous(vote_outputs[0]):
            logger.debug(f"  Candidate {idx}: FAILED factual check")
            continue
        info["n_passed_factual"] += 1
        logger.debug(f"  Candidate {idx}: passed factual check")

        # Stage 3: Total correctness
        correctness_prompt = TOTAL_CORRECTNESS_PROMPT.format(question=question, answer=answer_text)
        vote_outputs = _generate_validation(
            model, benchmark, [correctness_prompt], N_VALIDATION_VOTES, seed
        )
        if not check_unanimous(vote_outputs[0]):
            logger.debug(f"  Candidate {idx}: FAILED correctness check")
            continue
        info["n_passed_correctness"] += 1

        # All 3 stages passed — select this candidate
        info["selected_index"] = idx
        info["selection_method"] = "uq_validated"
        logger.info(f"  UQ VALIDATED: candidate {idx}/{len(candidates)} passed all 3 stages")
        return candidate, idx, info

    # No candidate passed all stages — fall back to first candidate
    info["selected_index"] = 0
    info["selection_method"] = "fallback"
    logger.warning(
        f"  UQ FALLBACK: no candidate passed all 3 stages "
        f"(cycle={info['n_passed_cycle']}, factual={info['n_passed_factual']}, "
        f"correctness={info['n_passed_correctness']} out of {len(candidates)} candidates). "
        f"Using first candidate."
    )
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

    Note: Adds the original benchmark's directory to sys.path so that local imports
    (e.g., matharena in HMMT, utils in JEEBench) resolve correctly when the TTC
    wrapper is loaded from a different directory.

    Args:
        original_class: The original benchmark class to wrap
        default_n_candidates: Default number of candidates per problem
    """
    import os
    import sys

    # Add the original benchmark's directory to sys.path so local imports
    # (matharena, utils, run_judge_results, etc.) resolve correctly.
    original_module_file = sys.modules[original_class.__module__].__file__
    if original_module_file:
        original_dir = os.path.dirname(os.path.abspath(original_module_file))
        if original_dir not in sys.path:
            sys.path.insert(0, original_dir)

    class TTCBenchmark(original_class):

        def __init__(self, n_candidates: int = default_n_candidates, **kwargs):
            super().__init__(**kwargs)
            self.n_candidates = n_candidates
            # Force n_repeat=1 — TTC does its own multi-generation + filtering
            if hasattr(self, "n_repeat"):
                self._original_n_repeat = self.n_repeat
                self.n_repeat = 1

        def _build_messages(self, example: dict) -> list[dict]:
            """Build chat messages for a single example.

            Default: uses module-level PROMPT.format(problem=...).
            Override in thin wrappers for benchmarks with complex prompt logic.
            """
            import sys
            mod = sys.modules.get(original_class.__module__)
            prompt_template = getattr(mod, "PROMPT", None) if mod else None
            if prompt_template is None:
                raise NotImplementedError(
                    f"No PROMPT found in {original_class.__module__}. "
                    f"Override _build_messages() in the TTC wrapper."
                )
            # Most benchmarks use PROMPT.format(problem=...) where the problem text
            # comes from different field names ("problem", "question", "Question").
            # AMC23 uses example["question"] but the template variable is {problem}.
            question_text = _get_question_text(example)

            # GPQADiamond uses problem + options
            if "multiple_choice_string" in example:
                content = prompt_template.format(
                    problem=question_text,
                    options=example["multiple_choice_string"],
                )
            else:
                content = prompt_template.format(problem=question_text)

            return [{"role": "user", "content": content}]

        def generate_responses(self, model: LM) -> Dict[str, Any]:
            """Generate N candidates per problem via single vLLM call, UQ-filter."""

            examples = self.load_questions()

            # Some benchmarks do preprocessing in generate_responses before building prompts.
            # Call _preprocess_examples to handle that (e.g., GPQADiamond shuffles options).
            self._preprocess_examples(examples)

            if model.rank != 0:
                return None

            for prob_idx, example in enumerate(examples):
                question_text = _get_question_text(example)

                # Build prompt using benchmark-specific logic
                messages = self._build_messages(example)
                templated = self._prepare_messages(messages, model)

                # Single vLLM call: generate all N candidates with n=N
                candidates = _generate_n_candidates(
                    model, templated, self.n_candidates, self.max_new_tokens,
                )

                # UQ filter: pick first valid candidate
                selected, sel_idx, ttc_info = uq_filter_candidates(
                    model, self, question_text, candidates, self.seed,
                )

                # Store in the format parent's evaluate_responses expects
                example["model_outputs"] = [selected]
                example["model_answers"] = [_re_extract_answer(self, selected, example)]
                example["model_output"] = selected
                example["model_answer"] = _re_extract_answer(self, selected, example)
                example["ttc_info"] = ttc_info
                example["all_candidates"] = candidates

                n_unique = len(set(candidates))
                if (prob_idx + 1) % 5 == 0 or prob_idx == 0:
                    logger.info(
                        f"TTC [{prob_idx + 1}/{len(examples)}] "
                        f"{n_unique}/{len(candidates)} unique candidates, "
                        f"selected candidate {sel_idx}/{self.n_candidates} "
                        f"via {ttc_info['selection_method']}"
                    )

            return {"examples": examples}

        def _preprocess_examples(self, examples: list[dict]) -> None:
            """Hook for benchmark-specific preprocessing before prompt building.

            Default is no-op. Override for benchmarks that need preprocessing
            (e.g., GPQADiamond shuffles multiple choice options).
            """
            pass

        # evaluate_responses is NOT overridden — uses parent's exact scoring logic

    # Set class name for TaskManager discovery
    ttc_name = original_class.__name__.replace("Benchmark", "TTCBenchmark")
    TTCBenchmark.__name__ = ttc_name
    TTCBenchmark.__qualname__ = ttc_name
    # __module__ must be set by caller for TaskManager discovery:
    #   ClassName = make_ttc_benchmark(OriginalClass)
    #   ClassName.__module__ = __name__

    return TTCBenchmark


def _re_extract_answer(benchmark: BaseBenchmark, output: str, example: dict):
    """Re-extract the model answer from output using the benchmark's extraction method.

    Checks for extract_answer() method first (math benchmarks like AIME, MATH500).
    Otherwise, looks at the original model_answers format to determine the extraction method.
    """
    # Math benchmarks define extract_answer (AIME24, AIME25, AIME26, AMC23, MATH500, OlympiadBench, JEEBench)
    if hasattr(benchmark, "extract_answer"):
        return benchmark.extract_answer(output)

    # For benchmarks without extract_answer, inspect what the parent produced
    # and use the same extraction.
    class_name = type(benchmark).__name__

    # Multiple choice: GPQADiamond
    if "GPQADiamond" in class_name:
        from eval.chat_benchmarks.GPQADiamond.testing_utils import get_multiple_choice_answer
        return get_multiple_choice_answer(output)

    # Multiple choice: HLE (has its own testing_utils)
    if "HLE" in class_name:
        from eval.chat_benchmarks.HLE.testing_utils import get_multiple_choice_answer
        return get_multiple_choice_answer(output)

    # HMMT uses matharena's extract_answer
    if "HMMT" in class_name:
        from matharena.parser import extract_answer
        list_answer = "," in str(example.get("answer", ""))
        return extract_answer(output, False, True, list_answer)[0]

    # OlympiadBench_Physics extracts boxed answers
    if "OlympiadBenchPhysics" in class_name:
        from eval.chat_benchmarks.OlympiadBench_Physics.eval_instruct import _extract_all_boxed, _deduplicate
        return _deduplicate(_extract_all_boxed(output))

    # LiveCodeBench extracts code blocks
    if "LiveCodeBench" in class_name:
        from eval.chat_benchmarks.LiveCodeBench.eval_instruct import has_code
        return has_code(output)

    # Fallback: return raw output
    return output


def _get_question_text(example: dict) -> str:
    """Extract the question text from an example for UQ validation."""
    for key in ("problem", "question", "Question", "prompt"):
        if key in example:
            return example[key]
    return str(example)


