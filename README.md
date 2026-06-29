# Language & Culture Benchmarking and LLM-as-a-Judge Taskforce

This repo contains scripts for generating model answers to questions in the sample datasets, and for scoring and evaluating those answers using an LLM-as-a-judge approach.

---

## Repository Structure

```
.
├── common_data/
│   ├── lang_prompts.json                 # Per-language prompt templates
│   ├── LAAJ_updated_criteria_tabular.tsv # Evaluation criteria
├── local_llm_multilang.py                # Generate model answers
├── LaaJ_scoring.py                       # LLM-as-a-judge scoring
├── laaj_correlations.py                  # LLM-human correlation analysis
├── .data/
│   ├── samples/                          # Per-language datasets (unversioned)
│   └── laaj_scores/                      # Scoring outputs
└── results/                              # Correlation results
```

---

## Scripts

### 1. Generating Model Answers
**Script:** `local_llm_multilang.py`

Generates model answers for per-language/culture/region datasets.

- **Input:** Dataset files in `.data/samples/`, following the naming convention:
  `Lang&Cult_<Language>_dataset.tsv` (e.g., `Lang&Cult_Slovenian_dataset.tsv`)
- **Prompts:** Models are prompted in each respective language using templates from `lang_prompts.json`

---

### 2. LLM-as-a-Judge Scoring
**Script:** `Laaj_scoring.py`

Scores model outputs using evaluation criteria from `LAAJ_updated_criteria_tabular.tsv`.

- **Output:** `.data/laaj_scores/LAAJ_{safe_model}_{lang}.tsv`

---

### 3. LLM-Human Correlation Analysis
**Script:** `laaj_correlations.py`

Computes correlations between LLM scores and human judgments, and summarizes them.

- **Output:** `results/LAAJ-human_correlation_{lang}.tsv`
