#!/usr/bin/env python3
"""
Example usage for reference (to change and copy paste before running):
----------------
python3 psycholinguistics_pronoun_baseline.py \
  --model bert-base-uncased \
  --split validation \
  --max-items 300 \
  --run-perturbations \
  --output-dir outputs_baseline

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from nltk.corpus import wordnet as wn
except Exception:
    wn = None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Example:
    sentence: str
    option1: str
    option2: str
    answer: str  # "1" or "2"
    example_id: Optional[str] = None


@dataclass
class CandidateScore:
    candidate_text: str
    pll: float


@dataclass
class BaselineResult:
    example_id: Optional[str]
    sentence: str
    option1: str
    option2: str
    gold: str
    candidate1_text: str
    candidate2_text: str
    candidate1_pll: float
    candidate2_pll: float
    predicted: str
    correct: bool


@dataclass
class PerturbationResult:
    example_id: Optional[str]
    perturbation_type: str
    original_sentence: str
    perturbed_sentence: str
    gold: str
    original_predicted: str
    perturbed_predicted: str
    changed_prediction: bool
    original_correct: bool
    perturbed_correct: bool


# -----------------------------
# Utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()



def replace_blank(sentence: str, replacement: str) -> str:
    """
    WinoGrande sentences typically contain a single underscore `_` placeholder.
    Replace the first standalone underscore with the option text.
    """
    if "_" not in sentence:
        raise ValueError(f"No underscore placeholder found in sentence: {sentence}")
    out = sentence.replace("_", replacement, 1)
    return normalize_space(out)


# -----------------------------
# Dataset loading
# -----------------------------


def load_winogrande_examples(split: str, max_items: Optional[int], seed: int) -> List[Example]:
    """
    We load the official WinoGrande dataset from Hugging Face datasets.

    Reference fields for info:
    - sentence
    - option1
    - option2
    - answer
    - qID
    """
    ds = load_dataset("winogrande", "winogrande_xl", split=split)
    rows = list(ds)
    if max_items is not None and max_items < len(rows):
        rows = rows[:max_items]

    examples: List[Example] = []
    for row in rows:
        examples.append(
            Example(
                sentence=row["sentence"],
                option1=row["option1"],
                option2=row["option2"],
                answer=str(row["answer"]),
                example_id=row.get("qID"),
            )
        )
    return examples


# -----------------------------
# PLL scorer for masked LMs
# -----------------------------

class MaskedLMPLLScorer:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()

        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"Tokenizer for {model_name} has no mask token.")

    @torch.no_grad()
    def sentence_pll(self, sentence: str) -> float:
        """
        Pseudo-log-likelihood for a full sentence.

        For each non-special token position i:
        - mask token i
        - ask the model for log-probability of the original token at i
        - sum across positions
        """
        enc = self.tokenizer(sentence, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        seq_len = input_ids.size(1)
        pll = 0.0

        # Skip special tokens [CLS] and [SEP]-like boundaries by checking special mask.
        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )

        for pos in range(seq_len):
            if special_mask[pos] == 1:
                continue

            original_token_id = input_ids[0, pos].item()
            masked_ids = input_ids.clone()
            masked_ids[0, pos] = self.tokenizer.mask_token_id

            outputs = self.model(input_ids=masked_ids, attention_mask=attention_mask)
            logits = outputs.logits[0, pos]
            log_probs = torch.log_softmax(logits, dim=-1)
            pll += float(log_probs[original_token_id].item())

        return pll

    def compare_candidates(self, candidate1: str, candidate2: str) -> Tuple[CandidateScore, CandidateScore, str]:
        score1 = self.sentence_pll(candidate1)
        score2 = self.sentence_pll(candidate2)
        predicted = "1" if score1 >= score2 else "2"
        return (
            CandidateScore(candidate_text=candidate1, pll=score1),
            CandidateScore(candidate_text=candidate2, pll=score2),
            predicted,
        )


# -----------------------------
# Baseline experiment
# -----------------------------


def evaluate_baseline(examples: List[Example], scorer: MaskedLMPLLScorer) -> List[BaselineResult]:
    results: List[BaselineResult] = []

    for ex in tqdm(examples, desc="Baseline evaluation"):
        cand1 = replace_blank(ex.sentence, ex.option1)
        cand2 = replace_blank(ex.sentence, ex.option2)

        score1, score2, predicted = scorer.compare_candidates(cand1, cand2)
        correct = predicted == ex.answer

        results.append(
            BaselineResult(
                example_id=ex.example_id,
                sentence=ex.sentence,
                option1=ex.option1,
                option2=ex.option2,
                gold=ex.answer,
                candidate1_text=score1.candidate_text,
                candidate2_text=score2.candidate_text,
                candidate1_pll=score1.pll,
                candidate2_pll=score2.pll,
                predicted=predicted,
                correct=correct,
            )
        )
    return results


# -----------------------------
# Perturbations
# -----------------------------

ADJ_POLARITY_MAP: Dict[str, str] = {
    "big": "small",
    "small": "big",
    "large": "small",
    "tiny": "large",
    "old": "young",
    "young": "old",
    "strong": "weak",
    "weak": "strong",
    "heavy": "light",
    "light": "heavy",
    "wide": "narrow",
    "narrow": "wide",
    "long": "short",
    "short": "long",
    "high": "low",
    "low": "high",
    "rich": "poor",
    "poor": "rich",
}



def synonym_substitution(sentence: str) -> Optional[str]:
    """
    Heuristic: replace the first adjective/adverb/verb/noun token we can find with a
    WordNet synonym that is a single word and not identical.

    This is intentionally simple and should be manually checked for research use.
    """
    if wn is None:
        return None

    tokens = re.findall(r"\w+|[^\w\s]", sentence)
    for i, tok in enumerate(tokens):
        if not re.match(r"^[A-Za-z]+$", tok):
            continue
        lower = tok.lower()
        synsets = wn.synsets(lower)
        candidates = []
        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != lower and " " not in name and name.isalpha():
                    candidates.append(name)
        if candidates:
            replacement = candidates[0]
            if tok[0].isupper():
                replacement = replacement.capitalize()
            new_tokens = tokens[:]
            new_tokens[i] = replacement
            out = "".join(
                [
                    (t if re.match(r"[^\w\s]", t) else (" " + t))
                    for t in new_tokens
                ]
            ).strip()
            return normalize_space(out)
    return None



def adjective_polarity_change(sentence: str) -> Optional[str]:
    """
    Heuristic adjective flip using a small hand-built polarity map.
    Replaces the first matching whole word.
    """
    for src, tgt in ADJ_POLARITY_MAP.items():
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        match = pattern.search(sentence)
        if match:
            matched = match.group(0)
            replacement = tgt.capitalize() if matched[0].isupper() else tgt
            return pattern.sub(replacement, sentence, count=1)
    return None



def structural_reordering(sentence: str) -> Optional[str]:
    """
    Very simple reordering heuristic for sentences containing "because".

    Example:
    "A ... because B ..." -> "Because B ..., A ..."

    This is only a rough baseline and may produce awkward results for some items.
    """
    lowered = sentence.lower()
    idx = lowered.find(" because ")
    if idx == -1:
        return None

    left = sentence[:idx].strip().rstrip(".,;: ")
    right = sentence[idx + len(" because "):].strip().rstrip(".,;: ")
    if not left or not right:
        return None

    reordered = f"Because {right}, {left}."
    return normalize_space(reordered)



def apply_perturbations(sentence: str) -> Dict[str, str]:
    outputs: Dict[str, str] = {}

    syn = synonym_substitution(sentence)
    if syn and syn != sentence:
        outputs["synonym_substitution"] = syn

    pol = adjective_polarity_change(sentence)
    if pol and pol != sentence:
        outputs["adjective_polarity_change"] = pol

    reo = structural_reordering(sentence)
    if reo and reo != sentence:
        outputs["structural_reordering"] = reo

    return outputs


# -----------------------------
# Perturbation evaluation
# -----------------------------


def build_prediction_for_sentence(
    sentence: str,
    option1: str,
    option2: str,
    scorer: MaskedLMPLLScorer,
) -> str:
    cand1 = replace_blank(sentence, option1)
    cand2 = replace_blank(sentence, option2)
    _, _, predicted = scorer.compare_candidates(cand1, cand2)
    return predicted



def evaluate_perturbations(
    examples: List[Example],
    baseline_results: List[BaselineResult],
    scorer: MaskedLMPLLScorer,
    max_items_for_perturbation: Optional[int] = None,
) -> List[PerturbationResult]:
    base_map = {r.example_id if r.example_id is not None else str(i): r for i, r in enumerate(baseline_results)}

    selected_examples = examples
    if max_items_for_perturbation is not None:
        selected_examples = selected_examples[:max_items_for_perturbation]

    perturbation_results: List[PerturbationResult] = []

    for idx, ex in enumerate(tqdm(selected_examples, desc="Perturbation evaluation")):
        base_key = ex.example_id if ex.example_id is not None else str(idx)
        base = base_map[base_key]

        perturbed_versions = apply_perturbations(ex.sentence)
        for ptype, psentence in perturbed_versions.items():
            try:
                perturbed_pred = build_prediction_for_sentence(psentence, ex.option1, ex.option2, scorer)
            except Exception:
                # Some perturbations may break placeholder logic or produce unusable sentences.
                continue

            perturbation_results.append(
                PerturbationResult(
                    example_id=ex.example_id,
                    perturbation_type=ptype,
                    original_sentence=ex.sentence,
                    perturbed_sentence=psentence,
                    gold=ex.answer,
                    original_predicted=base.predicted,
                    perturbed_predicted=perturbed_pred,
                    changed_prediction=(base.predicted != perturbed_pred),
                    original_correct=base.correct,
                    perturbed_correct=(perturbed_pred == ex.answer),
                )
            )

    return perturbation_results


# -----------------------------
# Reporting
# -----------------------------


def baseline_summary(results: List[BaselineResult]) -> Dict[str, float]:
    total = len(results)
    correct = sum(r.correct for r in results)
    return {
        "n_items": total,
        "n_correct": correct,
        "accuracy": (correct / total) if total > 0 else float("nan"),
    }



def perturbation_summary(results: List[PerturbationResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(asdict(r) for r in results)
    grouped = (
        df.groupby("perturbation_type")
        .agg(
            n=("perturbation_type", "count"),
            changed_prediction_rate=("changed_prediction", "mean"),
            original_accuracy=("original_correct", "mean"),
            perturbed_accuracy=("perturbed_correct", "mean"),
        )
        .reset_index()
    )
    return grouped



def save_outputs(
    output_dir: str,
    baseline_results: List[BaselineResult],
    perturbation_results: List[PerturbationResult],
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    baseline_df = pd.DataFrame(asdict(r) for r in baseline_results)
    baseline_df.to_csv(os.path.join(output_dir, "baseline_predictions.csv"), index=False)

    with open(os.path.join(output_dir, "baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_summary(baseline_results), f, indent=2)

    if perturbation_results:
        perturb_df = pd.DataFrame(asdict(r) for r in perturbation_results)
        perturb_df.to_csv(os.path.join(output_dir, "perturbation_predictions.csv"), index=False)

        summary_df = perturbation_summary(perturbation_results)
        summary_df.to_csv(os.path.join(output_dir, "perturbation_summary.csv"), index=False)


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline Winograd-style pronoun resolution with masked LM PLL scoring.")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Hugging Face masked LM model name")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split: train / validation / test")
    parser.add_argument("--max-items", type=int, default=300, help="Number of items to take sequentially from the start of the split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs_baseline")
    parser.add_argument("--run-perturbations", action="store_true")
    parser.add_argument(
        "--max-perturb-items",
        type=int,
        default=50,
        help="How many baseline items to attempt perturbation on",
    )
    return parser.parse_args()



def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    print(f"Using device: {device}")
    print(f"Loading WinoGrande split='{args.split}' with max_items={args.max_items}")

    examples = load_winogrande_examples(args.split, args.max_items, args.seed)
    print(f"Loaded {len(examples)} examples")

    # ------------------------------------------------------------------
    # Print a few sample examples so we can verify the data looks right
    # ------------------------------------------------------------------
    section("SAMPLE LOADED EXAMPLES (first 5)")
    for ex in examples[:5]:
        print(f"\n  ID      : {ex.example_id}")
        print(f"  Sentence: {ex.sentence}")
        print(f"  Option 1: {ex.option1}")
        print(f"  Option 2: {ex.option2}")
        print(f"  Gold    : {ex.answer}")

    scorer = MaskedLMPLLScorer(model_name=args.model, device=device)
    baseline_results = evaluate_baseline(examples, scorer)
    base_stats = baseline_summary(baseline_results)

    section("BASELINE SUMMARY")
    print(json.dumps(base_stats, indent=2))

    # INFO: Printing for manual inspection purpoeses
    correct_results   = [r for r in baseline_results if r.correct]
    incorrect_results = [r for r in baseline_results if not r.correct]

    section("SAMPLE CORRECT PREDICTIONS (first 5)")
    for r in correct_results[:5]:
        print(f"\n  Sentence : {r.sentence}")
        print(f"  Cand 1   : {r.candidate1_text}  [PLL: {r.candidate1_pll:.2f}]")
        print(f"  Cand 2   : {r.candidate2_text}  [PLL: {r.candidate2_pll:.2f}]")
        print(f"  Gold: {r.gold}  |  Predicted: {r.predicted}  ✓")

    section("SAMPLE INCORRECT PREDICTIONS (first 5)")
    for r in incorrect_results[:5]:
        print(f"\n  Sentence : {r.sentence}")
        print(f"  Cand 1   : {r.candidate1_text}  [PLL: {r.candidate1_pll:.2f}]")
        print(f"  Cand 2   : {r.candidate2_text}  [PLL: {r.candidate2_pll:.2f}]")
        print(f"  Gold: {r.gold}  |  Predicted: {r.predicted}  ✗")

    # INFO: PLL score gap distribution- how confident are correct vs wrong calls
    section("PLL SCORE GAP (|cand1_pll - cand2_pll|)")
    gaps_correct   = [abs(r.candidate1_pll - r.candidate2_pll) for r in correct_results]
    gaps_incorrect = [abs(r.candidate1_pll - r.candidate2_pll) for r in incorrect_results]
    if gaps_correct:
        print(f"  Correct predictions   — mean gap: {sum(gaps_correct)/len(gaps_correct):.2f}  "
              f"min: {min(gaps_correct):.2f}  max: {max(gaps_correct):.2f}")
    if gaps_incorrect:
        print(f"  Incorrect predictions — mean gap: {sum(gaps_incorrect)/len(gaps_incorrect):.2f}  "
              f"min: {min(gaps_incorrect):.2f}  max: {max(gaps_incorrect):.2f}")

    perturbation_results: List[PerturbationResult] = []
    if args.run_perturbations:
        perturbation_results = evaluate_perturbations(
            examples=examples,
            baseline_results=baseline_results,
            scorer=scorer,
            max_items_for_perturbation=args.max_perturb_items,
        )
        if perturbation_results:
            section("PERTURBATION AGGREGATE SUMMARY")
            print(perturbation_summary(perturbation_results).to_string(index=False))


            # INFO: Few Cases where pertubation flipped for each pertubation type 
            perturb_df = pd.DataFrame(asdict(r) for r in perturbation_results)
            flipped_df = perturb_df[perturb_df["changed_prediction"] == True]

            section("SAMPLE PREDICTION FLIPS BY PERTURBATION TYPE")
            for ptype in perturb_df["perturbation_type"].unique():
                subset = flipped_df[flipped_df["perturbation_type"] == ptype].head(3)
                print(f"\n  -- {ptype} ({len(flipped_df[flipped_df['perturbation_type']==ptype])} flips total) --")
                if subset.empty:
                    print("  (no flips for this type)")
                    continue
                for _, row in subset.iterrows():
                    print(f"\n    Original : {row['original_sentence']}")
                    print(f"    Perturbed: {row['perturbed_sentence']}")
                    print(f"    Gold: {row['gold']}  |  {row['original_predicted']} → {row['perturbed_predicted']}"
                          f"  ({'was correct, now wrong' if row['original_correct'] and not row['perturbed_correct'] else 'was wrong, now correct' if not row['original_correct'] and row['perturbed_correct'] else 'correctness unchanged'})")

            # INFO: Few cases where pert. did not flip for each pertubation type (stability examples)
            stable_df = perturb_df[perturb_df["changed_prediction"] == False]
            section("SAMPLE STABLE PREDICTIONS (perturbation had no effect, first 5)")
            for _, row in stable_df.head(5).iterrows():
                print(f"\n  [{row['perturbation_type']}]")
                print(f"  Original : {row['original_sentence']}")
                print(f"  Perturbed: {row['perturbed_sentence']}")
                print(f"  Prediction held at: {row['original_predicted']}  (Gold: {row['gold']})")
        else:
            print("\nNo valid perturbation results were generated.")

    save_outputs(args.output_dir, baseline_results, perturbation_results)
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()