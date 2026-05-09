# Psycholinguistic Analysis: Do Language Models Resolve Pronoun Ambiguity Using Human-like Linguistic Cues?

**Authors:** Namrata Baliga (2022101021) & Pratishtha Saxena (2022113008)  
**Course:** Computational Psycholinguistics, Spring 2026  
**Institution:** IIIT Hyderabad

---

## Overview

This project investigates whether transformer-based language models resolve Winograd-style pronoun ambiguity using cognitive mechanisms similar to those employed by human readers. We evaluate four scoring configurations—RoBERTa-base (Partial MLM), GPT-2 (Causal LM), Centering+RoBERTa, and Centering+GPT-2—on 300 WinoGrande validation items and conduct controlled perturbation experiments (synonym substitution, adjective polarity flips, negation insertion) to probe the linguistic cues driving model decisions. Our theoretical framework draws on Centering Theory (Grosz, Joshi & Weinstein, 1995) as operationalized in Traxler's psycholinguistics textbook (Chapter 6).

## Research Questions

1. Do transformer language models resolve Winograd-style pronoun ambiguity using semantic reasoning similar to human cognitive processes?
2. Which linguistic cues—semantic reasoning, syntactic structure, or lexical associations—most strongly influence model disambiguation decisions?

## Repository Structure

```
Psycholinguistics-project/
├── Main_model/
│   ├── final_model.ipynb          # Final experiment notebook (RoBERTa + GPT-2 + Centering + Perturbations)
│   └── main_model.ipynb           # Earlier development notebook
├── Ngram-Baseline/
│   ├── baseline_experiment.ipynb  # N-gram baseline (v1)
│   └── baseline_experiment_v2.ipynb # N-gram baseline (v2, frozen 300 items)
├── bert_model_3_perturbation_types/
│   ├── balanced_perturbations_validated_60.csv   # 60-item balanced perturbation set
│   ├── candidate_perturbations.csv               # Auto-generated perturbation candidates
│   ├── generate_candidates.py                     # Perturbation generation script
│   ├── psycholinguistics_curated_eval.py          # Curated evaluation script
│   └── outputs_curated_balanced_60/               # Results from balanced BERT perturbation run
├── outputs_baseline/
│   ├── baseline_predictions.csv       # Per-item baseline predictions
│   ├── baseline_summary.json          # Aggregate baseline accuracy
│   ├── perturbation_predictions.csv   # Perturbation experiment predictions
│   └── perturbation_summary.csv       # Perturbation summary statistics
├── psycholinguistics_pronoun_baseline.py  # Standalone baseline evaluation script
├── code.ipynb                             # Supplementary analysis notebook
├── main_perturbation_model.ipynb          # RoBERTa perturbation experiments
├── base.png                               # Base accuracy bar chart
├── graph.png                              # Centering theory + perturbation robustness plots
├── report.tex                             # LaTeX source for the final report
├── Psycholinguistics project proposal (3).pdf  # Project proposal
├── pre_submission.pdf                     # Mid-evaluation submission
└── README.md                              # This file
```

## Models Used

| Model | Type | Parameters | Source |
|-------|------|-----------|--------|
| RoBERTa-base | Masked LM (Partial PLL) | 125M | HuggingFace `roberta-base` |
| GPT-2 | Causal LM | 117M | HuggingFace `gpt2` |
| Centering+RoBERTa | MLM + discourse features | 125M + heuristic | Custom (α=1.5) |
| Centering+GPT-2 | CLM + discourse features | 117M + heuristic | Custom (α=1.5) |

## Dataset

- **WinoGrande** (Sakaguchi et al., 2019): First 300 validation items from `winogrande_xl` split
- Each item: sentence with blank (`_`), two candidate referents, gold answer
- Perturbations: 456 automatically generated and grammaticality-filtered variants

## Key Results

### Baseline Accuracy (300 items)
| Model | Accuracy |
|-------|----------|
| RoBERTa-base (Partial MLM) | **54.0%** |
| GPT-2 (CLM) | 52.3% |
| Centering+RoBERTa | 53.0% |
| Centering+GPT-2 | 52.7% |

### Perturbation Flip Rates
| Perturbation | RoBERTa Flip Rate | GPT-2 Flip Rate |
|-------------|-------------------|-----------------|
| Negation | 7.3% | 5.9% |
| Synonym | **17.3%** | 7.4% |
| Polarity | 11.8% | 0.0% |

### Centering Theory Impact (RoBERTa)
- Gold referent is subject: **66.7%** accuracy
- Gold referent is non-subject: 46.2% accuracy
- Gold referent is first-mentioned: **72.5%** accuracy
- Gold referent is not first-mentioned: 32.9% accuracy
- Mann-Whitney U test: U=12768, p=0.0006 (significant)

## Setup & Reproduction

### Requirements
```
torch
transformers
datasets
nltk
spacy
pandas
numpy
matplotlib
scipy
tqdm
```

### Installation
```bash
pip install torch transformers datasets nltk spacy pandas numpy matplotlib scipy tqdm
python -m spacy download en_core_web_sm
```

### Running
Open and run `Main_model/final_model.ipynb` in Google Colab (GPU recommended) or Jupyter.

## Sampling Note

Items are taken as the first 300 rows sequentially (NOT randomly) from the WinoGrande validation split. This matches the partner's notebook (`baseline_experiment_v2.ipynb`) so both pipelines operate on the same 300 sentences.

## References

1. Sakaguchi, K., et al. (2019). WinoGrande: An Adversarial Winograd Schema Challenge at Scale. *arXiv:1907.10641*.
2. Traxler, M. J. (2012). *Introduction to Psycholinguistics: Understanding Language Science*. Wiley-Blackwell.
3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL-HLT*.
4. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.
5. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI*.
6. Grosz, B. J., Joshi, A. K., & Weinstein, S. (1995). Centering: A Framework for Modeling the Local Coherence of Discourse. *Computational Linguistics*, 21(2).

## License

This project is for academic purposes (IIIT Hyderabad, Computational Psycholinguistics course).