#!/usr/bin/env python3
"""
improved Winogrande baseline + curated perturbation workflow (via import of a curated perturbation CSV)

Commandss:
python3 psycholinguistics_curated_eval.py \
  --model bert-base-uncased \
  --split validation \
  --max-items 300 \
  --output-dir outputs_curated \
  --export-dataset-sheet

After reviewing and editing the seed sheet:
python3 psycholinguistics_curated_eval.py \
  --model bert-base-uncased \
  --split validation \
  --max-items 300 \
  --output-dir outputs_curated \
  --curated-perturbations curated_perturbation_seed_sheet_validated.csv \
  --only-keep-validated
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass
class Example:
    item_id: int
    sentence: str
    option1: str
    option2: str
    answer: str
    example_id: Optional[str] = None


@dataclass
class BaselineResult:
    item_id: int
    example_id: Optional[str]
    sentence: str
    option1: str
    option2: str
    gold: str
    candidate1_text: str
    candidate2_text: str
    candidate1_pll: float
    candidate2_pll: float
    score_gap: float
    predicted: str
    correct: bool


@dataclass
class CuratedPerturbationResult:
    item_id: int
    cue_category: str
    perturbation_type: str
    original_sentence: str
    perturbed_sentence: str
    gold: str
    original_predicted: str
    perturbed_predicted: str
    changed_prediction: bool
    original_correct: bool
    perturbed_correct: bool
    grammar_valid: Optional[str] = None
    meaning_preserved: Optional[str] = None
    gold_preserved: Optional[str] = None
    keep_for_analysis: Optional[str] = None
    validator_notes: Optional[str] = None


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def replace_blank(sentence: str, replacement: str) -> str:
    if "_" not in sentence:
        raise ValueError(f"No underscore placeholder found in sentence: {sentence}")
    return normalize_space(sentence.replace("_", replacement, 1))


def load_examples(split: str, max_items: int) -> List[Example]:
    ds = load_dataset("winogrande", "winogrande_xl", split=split)
    rows = ds.select(range(min(max_items, len(ds))))
    examples: List[Example] = []
    for i, row in enumerate(rows, start=1):
        examples.append(
            Example(
                item_id=i,
                sentence=row["sentence"],
                option1=row["option1"],
                option2=row["option2"],
                answer=str(row["answer"]),
                example_id=row.get("qID"),
            )
        )
    return examples


def export_dataset_sheet(examples: List[Example], output_dir: str) -> str:
    out = os.path.join(output_dir, "winogrande_frozen_subset.csv")
    df = pd.DataFrame([asdict(x) for x in examples])
    df.to_csv(out, index=False)
    return out


class MaskedLMPLLScorer:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.model.eval()
        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"Tokenizer for {model_name} has no mask token.")

    @torch.no_grad()
    def sentence_pll(self, sentence: str) -> float:
        enc = self.tokenizer(sentence, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        seq_len = input_ids.size(1)
        pll = 0.0
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

    def predict(self, sentence: str, option1: str, option2: str) -> Tuple[str, float, float, str, str]:
        cand1 = replace_blank(sentence, option1)
        cand2 = replace_blank(sentence, option2)
        score1 = self.sentence_pll_fast(cand1)
        score2 = self.sentence_pll_fast(cand2)
        predicted = "1" if score1 >= score2 else "2"
        return predicted, score1, score2, cand1, cand2

    @torch.no_grad()
    def sentence_pll_fast(self, sentence: str) -> float:
        enc = self.tokenizer(sentence, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        seq_len = input_ids.size(1)
        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )

        # Build a batch where each row masks one position
        mask_positions = [i for i in range(seq_len) if special_mask[i] == 0]
        if not mask_positions:
            return 0.0

        batch_ids = input_ids.repeat(len(mask_positions), 1)
        for row, pos in enumerate(mask_positions):
            batch_ids[row, pos] = self.tokenizer.mask_token_id

        outputs = self.model(input_ids=batch_ids, attention_mask=attention_mask.repeat(len(mask_positions), 1))
        logits = outputs.logits  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=-1)

        pll = 0.0
        for row, pos in enumerate(mask_positions):
            original_token_id = input_ids[0, pos].item()
            pll += float(log_probs[row, pos, original_token_id].item())

        return pll   

def evaluate_baseline(examples: List[Example], scorer: MaskedLMPLLScorer) -> List[BaselineResult]:
    results = []
    for ex in tqdm(examples, desc="Baseline evaluation"):
        predicted, score1, score2, cand1, cand2 = scorer.predict(ex.sentence, ex.option1, ex.option2)
        results.append(
            BaselineResult(
                item_id=ex.item_id,
                example_id=ex.example_id,
                sentence=ex.sentence,
                option1=ex.option1,
                option2=ex.option2,
                gold=ex.answer,
                candidate1_text=cand1,
                candidate2_text=cand2,
                candidate1_pll=score1,
                candidate2_pll=score2,
                score_gap=abs(score1 - score2),
                predicted=predicted,
                correct=(predicted == ex.answer),
            )
        )
    return results


def load_curated_perturbations(path: str, only_keep_validated: bool=False) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["item_id", "cue_category", "perturbation_type", "perturbed_sentence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in curated perturbation CSV: {missing}")
    if only_keep_validated and "keep_for_analysis" in df.columns:
        keep = df["keep_for_analysis"].astype(str).str.strip().str.lower().isin({"yes", "y", "true", "1"})
        df = df[keep].copy()
    return df


def evaluate_curated_perturbations(
    examples: List[Example],
    baseline_results: List[BaselineResult],
    perturb_df: pd.DataFrame,
    scorer: MaskedLMPLLScorer,
) -> List[CuratedPerturbationResult]:
    ex_map = {e.item_id: e for e in examples}
    base_map = {r.item_id: r for r in baseline_results}
    results = []
    for _, row in tqdm(perturb_df.iterrows(), total=len(perturb_df), desc="Curated perturbation evaluation"):
        item_id = int(row["item_id"])
        ex = ex_map[item_id]
        base = base_map[item_id]
        pert_sentence = str(row["perturbed_sentence"])
        predicted, _, _, _, _ = scorer.predict(pert_sentence, ex.option1, ex.option2)
        results.append(
            CuratedPerturbationResult(
                item_id=item_id,
                cue_category=str(row.get("cue_category", "")),
                perturbation_type=str(row.get("perturbation_type", "")),
                original_sentence=ex.sentence,
                perturbed_sentence=pert_sentence,
                gold=ex.answer,
                original_predicted=base.predicted,
                perturbed_predicted=predicted,
                changed_prediction=(base.predicted != predicted),
                original_correct=base.correct,
                perturbed_correct=(predicted == ex.answer),
                grammar_valid=None if pd.isna(row.get("grammar_valid")) else str(row.get("grammar_valid")),
                meaning_preserved=None if pd.isna(row.get("meaning_preserved")) else str(row.get("meaning_preserved")),
                gold_preserved=None if pd.isna(row.get("gold_preserved")) else str(row.get("gold_preserved")),
                keep_for_analysis=None if pd.isna(row.get("keep_for_analysis")) else str(row.get("keep_for_analysis")),
                validator_notes=None if pd.isna(row.get("validator_notes")) else str(row.get("validator_notes")),
            )
        )
    return results


def baseline_summary(results: List[BaselineResult]) -> Dict[str, float]:
    total = len(results)
    correct = sum(r.correct for r in results)
    gaps_correct = [r.score_gap for r in results if r.correct]
    gaps_incorrect = [r.score_gap for r in results if not r.correct]
    return {
        "n_items": total,
        "n_correct": correct,
        "accuracy": correct / total if total else float("nan"),
        "mean_gap_correct": sum(gaps_correct) / len(gaps_correct) if gaps_correct else float("nan"),
        "mean_gap_incorrect": sum(gaps_incorrect) / len(gaps_incorrect) if gaps_incorrect else float("nan"),
    }


def curated_summary(results: List[CuratedPerturbationResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(r) for r in results])
    return (
        df.groupby(["cue_category", "perturbation_type"])
        .agg(
            n=("item_id", "count"),
            changed_prediction_rate=("changed_prediction", "mean"),
            original_accuracy=("original_correct", "mean"),
            perturbed_accuracy=("perturbed_correct", "mean"),
        )
        .reset_index()
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--split", default="validation")
    p.add_argument("--max-items", type=int, default=300)
    p.add_argument("--output-dir", default="outputs_curated")
    p.add_argument("--export-dataset-sheet", action="store_true")
    p.add_argument("--curated-perturbations", type=str, default=None)
    p.add_argument("--only-keep-validated", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    examples = load_examples(args.split, args.max_items)
    if args.export_dataset_sheet:
        path = export_dataset_sheet(examples, args.output_dir)
        print(f"Exported frozen subset sheet to: {path}")

    scorer = MaskedLMPLLScorer(args.model, choose_device())
    baseline_results = evaluate_baseline(examples, scorer)
    baseline_df = pd.DataFrame([asdict(r) for r in baseline_results])
    baseline_df.to_csv(os.path.join(args.output_dir, "baseline_predictions.csv"), index=False)

    with open(os.path.join(args.output_dir, "baseline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_summary(baseline_results), f, indent=2)

    print(json.dumps(baseline_summary(baseline_results), indent=2))

    if args.curated_perturbations:
        perturb_df = load_curated_perturbations(args.curated_perturbations, only_keep_validated=args.only_keep_validated)
        perturb_results = evaluate_curated_perturbations(examples, baseline_results, perturb_df, scorer)
        perturb_out = pd.DataFrame([asdict(r) for r in perturb_results])
        perturb_out.to_csv(os.path.join(args.output_dir, "curated_perturbation_predictions.csv"), index=False)
        summary = curated_summary(perturb_results)
        summary.to_csv(os.path.join(args.output_dir, "curated_perturbation_summary.csv"), index=False)
        if not summary.empty:
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
