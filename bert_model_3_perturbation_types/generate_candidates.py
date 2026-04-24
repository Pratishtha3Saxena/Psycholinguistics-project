#!/usr/bin/env python3

import pandas as pd
from datasets import load_dataset

# -----------------------------
# Simple perturbations (reuse logic)
# -----------------------------

def adjective_polarity_change(sentence):
    mapping = {
        "big": "small", "small": "big",
        "large": "small", "tiny": "large",
        "old": "young", "young": "old",
        "strong": "weak", "weak": "strong",
        "rich": "poor", "poor": "rich",
    }

    for src, tgt in mapping.items():
        if src in sentence.lower():
            return sentence.replace(src, tgt)
    return None


def structural_reordering(sentence):
    if " because " not in sentence.lower():
        return None

    parts = sentence.split(" because ")
    if len(parts) != 2:
        return None

    return f"Because {parts[1].strip()}, {parts[0].strip()}."


def safe_lexical_swap(sentence):
    mapping = {
        "easy": "simple",
        "hard": "difficult",
        "big": "large",
        "small": "tiny",
    }

    for src, tgt in mapping.items():
        if src in sentence.lower():
            return sentence.replace(src, tgt)
    return None


def map_category(ptype):
    if ptype == "adjective_polarity":
        return "lexical"
    if ptype == "structural":
        return "structural"
    if ptype == "lexical_swap":
        return "lexical"
    return "semantic"


# -----------------------------
# Main
# -----------------------------

def main():
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    rows = list(ds)[:300]

    output = []
    count = 0

    for idx, row in enumerate(rows):
        sentence = row["sentence"]

        # Focus on "because" sentences (better quality)
        if " because " not in sentence.lower():
            continue

        perturbations = {
            "adjective_polarity": adjective_polarity_change(sentence),
            "structural": structural_reordering(sentence),
            "lexical_swap": safe_lexical_swap(sentence),
        }

        for ptype, psentence in perturbations.items():
            if psentence and psentence != sentence:
                output.append({
                    "item_id": idx,
                    "cue_category": map_category(ptype),
                    "perturbation_type": ptype,
                    "original_sentence": sentence,
                    "perturbed_sentence": psentence,
                    "grammar_valid": "",
                    "meaning_preserved": "",
                    "gold_preserved": "",
                    "keep_for_analysis": "",
                    "validator_notes": "",
                })

        count += 1
        if count >= 120:
            break

    df = pd.DataFrame(output)
    df.to_csv("candidate_perturbations.csv", index=False)

    print(f"Generated {len(df)} candidates → candidate_perturbations.csv")


if __name__ == "__main__":
    main()