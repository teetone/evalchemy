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

Multi-part answers come in two forms:
1. Multiple elements in the final_answer list (3 problems)
2. A single string with sub-answers delimited by "$ , $" or "$,$"
Both are split into a list of individual answer parts with $ stripped.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

SUBSET = "OE_TO_physics_en_COMP"
OUTPUT_PATH = Path(__file__).parent / "data" / "olympiadbench_physics.json"


def split_multi_part_answer(s: str) -> list[str]:
    """Split a multi-part answer string on '$ , $' or '$,$' delimiters.

    Each part has surrounding $ stripped. Single-part answers return a
    one-element list.
    """
    # Split on $ , $ or $,$ (with optional whitespace)
    parts = re.split(r"\$\s*,\s*\$", s)
    cleaned = []
    for p in parts:
        p = p.strip()
        # Strip leading/trailing $ that remain after splitting
        if p.startswith("$"):
            p = p[1:]
        if p.endswith("$"):
            p = p[:-1]
        p = p.strip()
        if p:
            cleaned.append(p)
    return cleaned


def main():
    ds = load_dataset("Hothan/OlympiadBench", SUBSET, split="train")
    print(f"Loaded {len(ds)} problems from {SUBSET}")

    problems = []
    for row in ds:
        final_answer = row["final_answer"]

        # Collect all answer parts from all elements in the list
        all_parts = []
        for ans_str in final_answer:
            all_parts.extend(split_multi_part_answer(ans_str))

        problems.append({
            "id": len(problems),
            "problem": row["question"],
            "solution": row.get("solution", ""),
            "answers": all_parts,
            "answer_type": row.get("answer_type", ""),
            "subfield": "Physics",
        })

    multi_part = sum(1 for p in problems if len(p["answers"]) > 1)
    print(f"Total problems: {len(problems)} ({multi_part} multi-part)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(problems)} problems to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
