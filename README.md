# Language & Culture Benchmarking and LLM-as-a-judge Taskforce

This repo contains a few scripts used to generate model answers to the questions in the sample datasets.

## Generating model answers
Script: ``local_llm_multilang.py``

Per-language/culture/region datasets are expected to reside in the (unversioned) ``.data/samples`` folder, and the dataset name should follow ``Lang&Cult_Language_dataset.tsv`` (e.g., _Lang&Cult_Slovenian_dataset.tsv_).
Models are prompted in each respective language by looking up the prompt in ``lang_prompts.json``.

## LLM-as-a-judge scoring
``Laaj_scoring.py`` produces LLM scores according to the evaluation criteria defined in ``LAAJ_updated_criteria_tabular.tsv`` >>  ``.data/laaj_scores/LAAJ_{safe_model}_{lang}.tsv``
``laaj_correlations.py``computes LLM-human correlations and summarizes them into >> ``results/LAAJ-human_correlation_{lang}.tsv``





