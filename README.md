# Psycholinguistic Analysis: Do Language Models Resolve Pronoun Ambiguity Using Human-like Linguistic Cues?

**Authors:** Namrata Baliga (2022101021) & Pratishtha Saxena (2022113008)  
**Course:** Computational Psycholinguistics, Spring 2026  
**Institution:** IIIT Hyderabad

---

## Overview

This project investigates whether transformer-based language models resolve Winograd-style pronoun ambiguity using cognitive mechanisms similar to those employed by human readers. We evaluate six model configurations—**N-gram (Trigram baseline)**, **BERT (Full PLL)**, **RoBERTa-base (Partial MLM)**, **GPT-2 (Causal LM)**, and centering-augmented variants—on 300 WinoGrande validation items. 



We conduct controlled perturbation experiments (synonym substitution, adjective polarity flips, negation insertion) to probe whether models rely on genuine semantic reasoning or shallow lexical and statistical associations. Our theoretical framework draws on **Centering Theory** (Grosz, Joshi & Weinstein, 1995) as operationalized in Traxler's psycholinguistics textbook.



## Research Questions

1. Do transformer language models resolve Winograd-style pronoun ambiguity using semantic reasoning similar to human cognitive processes? 
2. Which linguistic cues—semantic reasoning, syntactic structure, or lexical associations—most strongly influence model disambiguation decisions?

## Models Used

| Scorer | Parameters | Description |
| :--- | :--- | :--- |
| **N-gram Baseline** | - | Trigram LM; corpus log-probability of candidate sentence. |
| **BERT (Full PLL)** | 110M | All non-special tokens masked in turn; log-probs summed. |
| **RoBERTa (Partial PLL)** | 125M | Only candidate tokens masked; focuses on the disambiguating signal. |
| **GPT-2 (CLM)** | 117M | Single forward pass; total causal LM log-probability. |
| **Centering+RoBERTa** | 125M+ | RoBERTa score + $\alpha \times prom$ ($\alpha=1.5$). |
| **Centering+GPT-2** | 117M+ | GPT-2 score + $\alpha \times prom$ ($\alpha=1.5$). |

## Key Results

### Baseline Accuracy (300 items)
Transformer models generally perform near chance (50%), while the N-gram baseline falls below, confirming that surface co-occurrence statistics are actively misleading on WinoGrande.

| Model | Accuracy | Correct/Total |
| :--- | :--- | :--- |
| N-gram Baseline | 48.67% | 146/300  |
| BERT (Full PLL) | 50.00% | 150/300  |
| **RoBERTa-base (Partial PLL)** | **54.00%** | 162/300  |
| GPT-2 (CLM) | 52.33% | 157/300  |
| Centering+RoBERTa | 53.00% | 159/300  |

### Perturbation Flip Rates
High synonym flip rates reveal a reliance on lexical forms over semantic meaning. BERT shows a distinct profile, being most sensitive to structural reordering.

| Model | Synonym | Negation | Polarity | Structural |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | 5.0%  | - | - | **15.0%**  |
| **RoBERTa** | **17.3%**  | 7.3%  | 11.8%  | - |
| **GPT-2** | 7.4%  | 5.9%  | 0.0%  | - |

### Centering Theory Impact (RoBERTa)
Model accuracy is significantly higher when the gold referent aligns with discourse prominence cues.
* **Gold referent is Subject**: 66.7% accuracy.
* **Gold referent is First-mentioned**: 72.5% accuracy.
* **Statistical Significance**: Mann-Whitney U test ($p=0.0006$).

## Repository Structure

```text
Psycholinguistics-project/
├── Main_model/
│   ├── final_model.ipynb          # Final experiment notebook (RoBERTa + GPT-2 + Centering)
│   └── main_model.ipynb           # Development notebook with ngram, Bert and RoBERTa-large tests
├── Ngram-Baseline/
│   ├── baseline_experiment.ipynb  # Trigram implementation
│   └── baseline_experiment_v2.ipynb # Baseline on frozen 300 items
├── bert_model_3_perturbation_types/
│   ├── outputs_curated_balanced_60/ # BERT results from balanced 60-item run
│   └── psycholinguistics_curated_eval.py # BERT evaluation script
├── report.tex                     # Final report (ACL Conference format)
└── README.md                      # This file

## Setup & Reproduction

1. **Install Dependencies**:
   ```bash
   pip install torch transformers datasets nltk spacy pandas numpy matplotlib scipy tqdm
   python -m spacy download en_core_web_sm
2. **Run Analysis**: Use Main_model/final_model.ipynb for the primary transformer and centering analysis. N-gram results are contained within the Ngram-Baseline/ directory.

