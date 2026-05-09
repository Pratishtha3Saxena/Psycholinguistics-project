# Psycholinguistic Analysis: Do Language Models Resolve Pronoun Ambiguity Using Human-like Linguistic Cues?

**Authors:** Namrata Baliga (2022101021) & Pratishtha Saxena (2022113008)  
**Course:** Computational Psycholinguistics, Spring 2026  
**Institution:** IIIT Hyderabad

---

## Overview

[cite_start]This project investigates whether transformer-based language models resolve Winograd-style pronoun ambiguity using cognitive mechanisms similar to those employed by human readers[cite: 213]. [cite_start]We evaluate six model configurations—**N-gram (Trigram baseline)**, **BERT (Full PLL)**, **RoBERTa-base (Partial MLM)**, **GPT-2 (Causal LM)**, and centering-augmented variants—on 300 WinoGrande validation items[cite: 10, 31]. 



[cite_start]We conduct controlled perturbation experiments (synonym substitution, adjective polarity flips, negation insertion) to probe whether models rely on genuine semantic reasoning or shallow lexical and statistical associations[cite: 11, 25]. [cite_start]Our theoretical framework draws on **Centering Theory** (Grosz, Joshi & Weinstein, 1995) as operationalized in Traxler's psycholinguistics textbook[cite: 8, 35].



## Research Questions

1. [cite_start]Do transformer language models resolve Winograd-style pronoun ambiguity using semantic reasoning similar to human cognitive processes? [cite: 213]
2. [cite_start]Which linguistic cues—semantic reasoning, syntactic structure, or lexical associations—most strongly influence model disambiguation decisions? [cite: 214]

## Models Used

| Scorer | Parameters | Description |
| :--- | :--- | :--- |
| **N-gram Baseline** | - | [cite_start]Trigram LM; corpus log-probability of candidate sentence[cite: 60]. |
| **BERT (Full PLL)** | 110M | [cite_start]All non-special tokens masked in turn; log-probs summed[cite: 60]. |
| **RoBERTa (Partial PLL)** | 125M | [cite_start]Only candidate tokens masked; focuses on the disambiguating signal[cite: 60]. |
| **GPT-2 (CLM)** | 117M | [cite_start]Single forward pass; total causal LM log-probability[cite: 60]. |
| **Centering+RoBERTa** | 125M+ | [cite_start]RoBERTa score + $\alpha \times prom$ ($\alpha=1.5$)[cite: 60]. |
| **Centering+GPT-2** | 117M+ | [cite_start]GPT-2 score + $\alpha \times prom$ ($\alpha=1.5$)[cite: 60]. |

## Key Results

### Baseline Accuracy (300 items)
[cite_start]Transformer models generally perform near chance (50%), while the N-gram baseline falls below, confirming that surface co-occurrence statistics are actively misleading on WinoGrande[cite: 90, 106].

| Model | Accuracy | Correct/Total |
| :--- | :--- | :--- |
| N-gram Baseline | 48.67% | [cite_start]146/300 [cite: 89] |
| BERT (Full PLL) | 50.00% | [cite_start]150/300 [cite: 89] |
| **RoBERTa-base (Partial PLL)** | **54.00%** | [cite_start]162/300 [cite: 89] |
| GPT-2 (CLM) | 52.33% | [cite_start]157/300 [cite: 89] |
| Centering+RoBERTa | 53.00% | [cite_start]159/300 [cite: 89] |

### Perturbation Flip Rates
[cite_start]High synonym flip rates reveal a reliance on lexical forms over semantic meaning[cite: 15, 191]. [cite_start]BERT shows a distinct profile, being most sensitive to structural reordering[cite: 154].

| Model | Synonym | Negation | Polarity | Structural |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | [cite_start]5.0% [cite: 127] | - | - | [cite_start]**15.0%** [cite: 127] |
| **RoBERTa** | [cite_start]**17.3%** [cite: 127] | [cite_start]7.3% [cite: 127] | [cite_start]11.8% [cite: 127] | - |
| **GPT-2** | [cite_start]7.4% [cite: 127] | [cite_start]5.9% [cite: 127] | [cite_start]0.0% [cite: 127] | - |

### Centering Theory Impact (RoBERTa)
[cite_start]Model accuracy is significantly higher when the gold referent aligns with discourse prominence cues[cite: 119, 142].
* [cite_start]**Gold referent is Subject**: 66.7% accuracy[cite: 126].
* [cite_start]**Gold referent is First-mentioned**: 72.5% accuracy[cite: 126].
* [cite_start]**Statistical Significance**: Mann-Whitney U test ($p=0.0006$)[cite: 123].

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

