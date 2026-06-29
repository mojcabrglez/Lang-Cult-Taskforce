import re
import os
import pandas as pd
import numpy as np
from scipy import stats

eval_model = 'gemma3:4b-it-qat'
safe_model = eval_model.replace(':', '-')

laaj_scores_path = '.data/laaj_scores/'

def load_and_merge(model_scores, manual_scores):
    """
    Load both files and merge on a shared ID column.
    Assumes both CSVs have columns like: id, Readability, Informativeness, Faithfulness

    ## criteria in LLM prompt: Readability and Linguistic acceptability, Adequacy, General quality
    """
    model_df = pd.read_csv(model_scores,sep='\t')
    manual_df = pd.read_excel(manual_scores)
    if 'Fluency' in manual_df.keys():
        manual_df.rename(columns={'Fluency':"Readability and Linguistic acceptability", 'General_Quality':'General quality'}, inplace=True)

    model_df.columns = model_df.columns.str.strip()
    manual_df.columns = manual_df.columns.str.strip()

    # Build a rename map from full name -> first 3 letters
    #rename_map = {criteria[m]['name']: criteria[m]['name'][:3] for m in criteria}

    #model_df = model_df.rename(columns=rename_map)
    #manual_df = manual_df.rename(columns=rename_map)

    merged = pd.merge(model_df, manual_df, left_index=True, right_index=True,
                      suffixes=('_model', '_manual'))
    return merged


def correlations_per_criterion(merged_df, criteria_names):
    """
    For each criterion, compute Pearson, Spearman, and Kendall correlations
    between model and manual scores.
    """
    results = []

    for criterion in criteria_names:
        model_col = f'{criterion}_model'
        manual_col = f'{criterion}_manual'

        if model_col not in merged_df or manual_col not in merged_df:
            print(f"Warning: columns for '{criterion}' not found, skipping.")
            continue

        model_scores  = merged_df[model_col].dropna()
        manual_scores = merged_df[manual_col].dropna()

        # Align indices after dropna
        common_idx    = model_scores.index.intersection(manual_scores.index)
        model_scores  = model_scores.loc[common_idx]
        manual_scores = manual_scores.loc[common_idx]

        pearson_r,  pearson_p  = stats.pearsonr(model_scores, manual_scores)
        spearman_r, spearman_p = stats.spearmanr(model_scores, manual_scores)
        kendall_r,  kendall_p  = stats.kendalltau(model_scores, manual_scores)

        results.append({
            'criterion':   criterion,
            'n':           len(common_idx),
            'pearson_r':   round(pearson_r,  3),
            'pearson_p':   round(pearson_p,  4),
            'spearman_r':  round(spearman_r, 3),
            'spearman_p':  round(spearman_p, 4),
            'kendall_r':   round(kendall_r,  3),
            'kendall_p':   round(kendall_p,  4),
            'mean_model':  round(model_scores.mean(),  2),
            'mean_manual': round(manual_scores.mean(), 2),
            'mae':         round((model_scores - manual_scores).abs().mean(), 3),
        })

    return pd.DataFrame(results)


def exact_and_adjacent_agreement(merged_df, criteria_names):
    """
    For ordinal scores 1-5, also compute:
    - Exact agreement rate (model == manual)
    - Adjacent agreement rate (|model - manual| <= 1)
    """
    rows = []
    for criterion in criteria_names:
        model_col  = f'{criterion}_model'
        manual_col = f'{criterion}_manual'

        diff = (merged_df[model_col] - merged_df[manual_col]).dropna()

        rows.append({
            'criterion':          criterion,
            'exact_agreement':    round((diff == 0).mean(), 3),
            'adjacent_agreement': round((diff.abs() <= 1).mean(), 3),
        })

    return pd.DataFrame(rows)


## criteria in LLM prompt: Readability and Linguistic acceptability, Adequacy, General quality

for laaj_score in os.listdir(laaj_scores_path):
    if f"LAAJ_{safe_model}_scores_" in laaj_score:
        lang = laaj_score.split('_')[-2]
        model_scores = pd.read_excel(os.path.join(laaj_scores_path, laaj_score))
        # should have parallel manual scores, named by this convention:
        manual_scores = f".data/manual_evals/manual_evaluation_{lang}.xlsx"

        criteria_df = pd.read_csv('LAAJ_criteria_tabular.tsv', sep='\t', encoding='UTF-8')
        criteria = criteria_df.to_dict(orient='index')

        # --- Run it ---
        criteria_names = [criteria[m]['name'] for m in criteria]  # from your parsing script

        merged = load_and_merge(model_scores, manual_scores)

        corr_df = correlations_per_criterion(merged, criteria_names)
        agreement_df = exact_and_adjacent_agreement(merged, criteria_names)

        summary = pd.merge(corr_df, agreement_df, on='criterion')
        print(f'Correlation for scores on {lang} data:')
        print(summary.to_string(index=False))

        summary.to_csv(f'results/LAAJ-human_correlation_{lang}.tsv', index=False,sep='\t')
        print(f'Wrote correlation statistics to results/LAAJ-human_correlation_{lang}.tsv')


