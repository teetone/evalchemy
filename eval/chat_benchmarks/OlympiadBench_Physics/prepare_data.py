#!/usr/bin/env python3
"""
Prepare OlympiadBench Physics (text-only, English) dataset.

Source: https://huggingface.co/datasets/Hothan/OlympiadBench

The HF dataset has 18 subsets with the following naming convention:
  * OE: Open-ended questions
  * TP: Theorem proof problems
  * MM: Multimodal
  * TO: Text-only
  * physics: Physics problems
  * maths: Math problems
  * en: English
  * zh: Chinese
  * COMP: Competition problems
  * CEE: Chinese College Entrance Exam problems

We want: OE (open-ended) + TO (text-only) + physics + en (English).
The only matching subset is OE_TO_physics_en_COMP (236 problems).
3 problems have multiple answers and are skipped.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

SUBSET = "OE_TO_physics_en_COMP"
OUTPUT_PATH = Path(__file__).parent / "data" / "olympiadbench_physics.json"


def strip_latex_dollars(s: str) -> str:
    """Strip surrounding $ delimiters from a LaTeX answer string."""
    s = s.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    return s


def main():
    ds = load_dataset("Hothan/OlympiadBench", SUBSET, split="train")
    print(f"Loaded {len(ds)} problems from {SUBSET}")

    problems = []
    skipped = 0
    for row in ds:
        final_answer = row["final_answer"]
        if len(final_answer) != 1:
            print(f"Skipping problem (multi-answer, {len(final_answer)} answers): {final_answer}")
            skipped += 1
            continue

        answer = strip_latex_dollars(final_answer[0])

        problems.append({
            "id": len(problems),
            "problem": row["question"],
            "solution": row.get("solution", ""),
            "answer": answer,
            "answer_type": row.get("answer_type", ""),
            "subfield": "Physics",
        })

    print(f"Skipped {skipped} multi-answer problems")
    print(f"Total problems: {len(problems)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(problems)} problems to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
