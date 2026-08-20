import re
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt

eval_model = 'gemma3:4b-it-qat'
safe_model = eval_model.replace(':', '-')

laaj_scores_path = '.data/laaj_scores/'

manual_scores_path = '.data/manual_evals/'
def load_and_merge(scorespath1, scorespath2, evaluators=None):
    """
    Load both files and merge on a shared ID column.
    Assumes both files have columns like: id, Fluency, Adequacy, General_Quality
    ## criteria in LLM prompt: Readability and Linguistic acceptability, Adequacy, General quality
    """
    if evaluators == 'human':
        model_df = pd.read_excel(scorespath1)
        manual_df = pd.read_excel(scorespath2)
    else:
        model_df = pd.read_csv(scorespath1, sep='\t')
        manual_df = pd.read_excel(scorespath2)

    model_df.columns = model_df.columns.str.strip()
    manual_df.columns = manual_df.columns.str.strip()

    # Build a rename map from full name -> first 3 letters
    #rename_map = {criteria[m]['name']: criteria[m]['name'][:3] for m in criteria}

    #model_df = model_df.rename(columns=rename_map)
    #manual_df = manual_df.rename(columns=rename_map)
    if evaluators:
        merged = pd.merge(model_df, manual_df, left_index=True, right_index=True,
                          suffixes=('_ann1', '_ann2'))
    else:
        merged = pd.merge(model_df, manual_df, left_index=True, right_index=True,
                      suffixes=('_model', '_human'))
    return merged

def pairwise_ordinal_krippendorff_alpha(x, y, min_score=1, max_score=5):
    """
    Pairwise Krippendorff's alpha for ordinal ratings.

    This implementation is for two raters/evaluators.
    It uses squared ordinal distance:
        distance(a, b) = ((a - b) / (max_score - min_score)) ** 2

    Returns np.nan if alpha cannot be computed.
    """
    paired = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    })

    paired = paired.replace([np.inf, -np.inf], np.nan)
    paired = paired.dropna()

    paired = paired[
        paired["x"].between(min_score, max_score) &
        paired["y"].between(min_score, max_score)
    ]

    if len(paired) < 2:
        return np.nan

    x_clean = paired["x"].round().astype(int)
    y_clean = paired["y"].round().astype(int)

    # Observed disagreement
    observed_disagreement = (((x_clean - y_clean) / (max_score - min_score)) ** 2).mean()

    # Expected disagreement from pooled marginal distribution
    pooled = pd.concat([x_clean, y_clean], ignore_index=True)
    values = np.arange(min_score, max_score + 1)

    probs = pooled.value_counts(normalize=True).reindex(values, fill_value=0).values

    expected_disagreement = 0.0

    for i, score_i in enumerate(values):
        for j, score_j in enumerate(values):
            distance = ((score_i - score_j) / (max_score - min_score)) ** 2
            expected_disagreement += probs[i] * probs[j] * distance

    if expected_disagreement == 0:
        return np.nan

    alpha = 1 - (observed_disagreement / expected_disagreement)

    return alpha

def icc_2_1_absolute_agreement(x, y):
    """
    Compute ICC(2,1): two-way random effects, single rater,
    absolute agreement.

    Suitable for pairwise evaluator agreement when both evaluators
    are considered exchangeable/random raters.

    Returns np.nan if ICC cannot be computed.
    """
    paired = pd.DataFrame({
        "rater1": pd.to_numeric(x, errors="coerce"),
        "rater2": pd.to_numeric(y, errors="coerce"),
    })

    paired = paired.replace([np.inf, -np.inf], np.nan)
    paired = paired.dropna()

    if len(paired) < 2:
        return np.nan

    ratings = paired.to_numpy(dtype=float)

    n, k = ratings.shape  # n items, k raters

    if k != 2:
        raise ValueError("This helper expects exactly two raters.")

    grand_mean = ratings.mean()

    mean_per_item = ratings.mean(axis=1)
    mean_per_rater = ratings.mean(axis=0)

    # Sum of squares
    ss_items = k * np.sum((mean_per_item - grand_mean) ** 2)
    ss_raters = n * np.sum((mean_per_rater - grand_mean) ** 2)
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_error = ss_total - ss_items - ss_raters

    # Degrees of freedom
    df_items = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)

    if df_items <= 0 or df_raters <= 0 or df_error <= 0:
        return np.nan

    ms_items = ss_items / df_items
    ms_raters = ss_raters / df_raters
    ms_error = ss_error / df_error

    denominator = (
        ms_items +
        (k - 1) * ms_error +
        (k * (ms_raters - ms_error) / n)
    )

    if denominator == 0:
        return np.nan

    icc = (ms_items - ms_error) / denominator

    return icc

def correlations_per_criterion(merged_df, criteria_names):
    """
    For each criterion, compute Spearman correlation
    between scores.
    """
    results = []


    if 'Fluency_ann2' in merged_df.columns:
        a = 'ann1'
        b = 'ann2'
    else:
        a = 'model'
        b = 'human'
    for criterion in criteria_names:
        a_col  = f'{criterion}_{a}'
        b_col = f'{criterion}_{b}'


        if a_col not in merged_df or b_col not in merged_df:
            print(f"Warning: columns for '{criterion}' not found, skipping.")
            continue

        model_scores  = merged_df[a_col].dropna()
        manual_scores = merged_df[b_col].dropna()

        # Align indices after dropna
        common_idx    = model_scores.index.intersection(manual_scores.index)
        model_scores  = model_scores.loc[common_idx]
        manual_scores = manual_scores.loc[common_idx]

        spearman_r, spearman_p = stats.spearmanr(model_scores, manual_scores)

        results.append({
            'criterion':   criterion,
            'n_examples':           len(common_idx),
            'spearman_r':  round(spearman_r, 3),
            'spearman_p':  round(spearman_p, 4),
            'mean_a':  round(model_scores.mean(),  2),
            'mean_b': round(manual_scores.mean(), 2),
            #'mae':         round((model_scores - manual_scores).abs().mean(), 3),
        })

    return pd.DataFrame(results)


def safe_weighted_kappa(x, y, weights="quadratic"):
    """
    Compute Cohen's weighted kappa safely after:
    - coercing scores to numeric
    - removing NaN / inf values
    - aligning x and y pairwise
    """
    paired = pd.DataFrame({
        "x": pd.to_numeric(x, errors="coerce"),
        "y": pd.to_numeric(y, errors="coerce"),
    })

    paired = paired.replace([np.inf, -np.inf], np.nan)
    paired = paired.dropna()

    if len(paired) < 2:
        return np.nan

    # Optional but useful if scores should be 1-5
    paired = paired[
        paired["x"].between(1, 5) &
        paired["y"].between(1, 5)
    ]

    if len(paired) < 2:
        return np.nan

    x_clean = paired["x"].round().astype(int)
    y_clean = paired["y"].round().astype(int)

    return cohen_kappa_score(
        x_clean,
        y_clean,
        weights=weights,
        labels=[1, 2, 3, 4, 5],
    )


def agreement_scores(merged_df, criteria_names):
    rows = []

    for criterion in criteria_names:
        possible_pairs = [
            (f"{criterion}_ann1", f"{criterion}_ann2"),
            (f"{criterion}_model", f"{criterion}_human"),
        ]

        found_pair = None

        for a_col, b_col in possible_pairs:
            if a_col in merged_df.columns and b_col in merged_df.columns:
                found_pair = (a_col, b_col)
                break

        if found_pair is None:
            print(f"Warning: columns for '{criterion}' not found, skipping.")
            continue

        a_col, b_col = found_pair

        paired = pd.DataFrame({
            "a": pd.to_numeric(merged_df[a_col], errors="coerce"),
            "b": pd.to_numeric(merged_df[b_col], errors="coerce"),
        })

        paired = paired.replace([np.inf, -np.inf], np.nan)
        paired = paired.dropna()

        if len(paired) == 0:
            print(f"Warning: no valid paired scores for '{criterion}', skipping.")
            continue

        diff = paired["a"] - paired["b"]

        linear_kappa = safe_weighted_kappa(
            paired["a"],
            paired["b"],
            weights="linear",
        )

        quadratic_kappa = safe_weighted_kappa(
            paired["a"],
            paired["b"],
            weights="quadratic",
        )

        ordinal_alpha = pairwise_ordinal_krippendorff_alpha(
            paired["a"],
            paired["b"],
            min_score=1,
            max_score=5,
        )

        icc_2_1 = icc_2_1_absolute_agreement(
            paired["a"],
            paired["b"],
        )

        rows.append({
            "criterion": criterion,

            "exact_agreement": round((diff == 0).mean(), 3),
            "adjacent_agreement": round((diff.abs() <= 1).mean(), 3),

            #"linear_CohenK": (
           #     round(linear_kappa, 3) if pd.notna(linear_kappa) else np.nan
            #),
            "quadratic_CohenK": (
                round(quadratic_kappa, 3) if pd.notna(quadratic_kappa) else np.nan
            ),

            #"ordinal_Krippendorff_alpha": (
            #    round(ordinal_alpha, 3) if pd.notna(ordinal_alpha) else np.nan
            #),
            "ICC_2_1_absolute": (
                round(icc_2_1, 3) if pd.notna(icc_2_1) else np.nan
            ),
        })

    return pd.DataFrame(rows)


def plot_krippendorff_alpha_by_language(
    summary_df,
    output_dir="results/plots",
    alpha_col="ordinal_Krippendorff_alpha",
):
    """
    Create one bar plot per language showing Krippendorff's alpha
    by criterion and annotator/evaluator pair.
    """
    os.makedirs(output_dir, exist_ok=True)

    required_cols = {"language", "comparison", "criterion", alpha_col}
    missing_cols = required_cols - set(summary_df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    plot_df = summary_df.copy()
    plot_df[alpha_col] = pd.to_numeric(plot_df[alpha_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[alpha_col])

    for language in sorted(plot_df["language"].unique()):
        lang_df = plot_df[plot_df["language"] == language].copy()

        criteria = list(lang_df["criterion"].drop_duplicates())
        comparisons = list(lang_df["comparison"].drop_duplicates())

        x = np.arange(len(criteria))
        width = 0.8 / max(len(comparisons), 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, comparison in enumerate(comparisons):
            comp_df = lang_df[lang_df["comparison"] == comparison]

            values = []
            for criterion in criteria:
                value = comp_df.loc[
                    comp_df["criterion"] == criterion,
                    alpha_col,
                ]

                if len(value) == 0:
                    values.append(np.nan)
                else:
                    values.append(value.iloc[0])

            offset = (i - (len(comparisons) - 1) / 2) * width

            ax.bar(
                x + offset,
                values,
                width,
                label=comparison,
            )

        ax.axhline(0.667, linestyle="--", linewidth=1, label="Tentative threshold α = 0.667")
        ax.axhline(0.800, linestyle=":", linewidth=1, label="Reliable threshold α = 0.800")

        ax.set_title(f"Krippendorff's alpha by criterion — {language}, {safe_model}")
        ax.set_xlabel("Criterion")
        ax.set_ylabel("Krippendorff's alpha")
        ax.set_xticks(x)
        ax.set_xticklabels(criteria, rotation=30, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        safe_language = str(language).replace(" ", "_").replace("/", "_")
        output_path = os.path.join(
            output_dir,
            f"krippendorff_alpha_{safe_language}.png",
        )

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote plot to {output_path}")



def plot_cohen_K_by_language(
    summary_df,
    output_dir="results/plots",
    alpha_col="quadratic_CohenK",
):
    """
    Create one bar plot per language showing Cohen's Kappa
    by criterion and annotator/evaluator pair.
    """
    os.makedirs(output_dir, exist_ok=True)

    required_cols = {"language", "comparison", "criterion", alpha_col}
    missing_cols = required_cols - set(summary_df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    plot_df = summary_df.copy()
    plot_df[alpha_col] = pd.to_numeric(plot_df[alpha_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[alpha_col])

    for language in sorted(plot_df["language"].unique()):
        lang_df = plot_df[plot_df["language"] == language].copy()

        criteria = list(lang_df["criterion"].drop_duplicates())
        comparisons = list(lang_df["comparison"].drop_duplicates())

        x = np.arange(len(criteria))
        width = 0.8 / max(len(comparisons), 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, comparison in enumerate(comparisons):
            comp_df = lang_df[lang_df["comparison"] == comparison]

            values = []
            for criterion in criteria:
                value = comp_df.loc[
                    comp_df["criterion"] == criterion,
                    alpha_col,
                ]

                if len(value) == 0:
                    values.append(np.nan)
                else:
                    values.append(value.iloc[0])

            offset = (i - (len(comparisons) - 1) / 2) * width

            ax.bar(
                x + offset,
                values,
                width,
                label=comparison,
            )

        ax.axhline(0.61, linestyle="--", linewidth=1, label="Substantial agreement κ = 0.61")
        ax.axhline(0.81, linestyle=":", linewidth=1, label="Almost perfect agreement κ= 0.81")

        ax.set_title(f"Quadratic weighed Cohen's κ by criterion — {language}, {safe_model}")
        ax.set_xlabel("Criterion")
        ax.set_ylabel("κ")
        ax.set_xticks(x)
        ax.set_xticklabels(criteria, rotation=30, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        safe_language = str(language).replace(" ", "_").replace("/", "_")
        output_path = os.path.join(
            output_dir,
            f"quadratic_kappa_{safe_language}.png",
        )

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Wrote plot to {output_path}")


criteria_names = ["Fluency", "Adequacy", "General_Quality"]

all_summaries = []

for laaj_score in os.listdir(laaj_scores_path):
    if f"LAAJ_{safe_model}" in laaj_score:
        # Safer language extraction:
        # If filename is LAAJ_model_scores_sl.xlsx, this gives "sl"
        lang = os.path.splitext(laaj_score)[0].split("_")[-1].strip('.xlsx')
        model_scores_path = os.path.join(laaj_scores_path, laaj_score)

        manual_scores1 = f".data/manual_evals/Manual_evaluation_{lang}_1.xlsx"
        manual_scores2 = f".data/manual_evals/Manual_evaluation_{lang}_2.xlsx"

        if not os.path.exists(manual_scores1):
            print(f"Warning: missing manual eval 1 for {lang}: {manual_scores1}")
            continue

        if not os.path.exists(manual_scores2):
            print(f"Warning: missing manual eval 2 for {lang}: {manual_scores2}")
            manual_scores2 = None


        comparisons = [
            {
                "comparison": "human_1_vs_human_2",
                "file1": manual_scores1,
                "file2": manual_scores2,
                "evaluators": "human",
            },
            {
                "comparison": "model_vs_human_1",
                "file1": model_scores_path,
                "file2": manual_scores1,
                "evaluators": None,
            },
            {
                "comparison": "model_vs_human_2",
                "file1": model_scores_path,
                "file2": manual_scores2,
                "evaluators": None,
            },
        ]

        print(f"\n{'=' * 80}")
        print(f"Language: {lang}")
        print(f"{'=' * 80}")

        for comparison in comparisons:
            comparison_name = comparison["comparison"]
            if comparison["file2"] == None:
                continue
            if comparison["file1"] == None:
                continue

            merged = load_and_merge(
                comparison["file1"],
                comparison["file2"],
                evaluators=comparison["evaluators"],
            )

            corr_df = correlations_per_criterion(
                merged,
                criteria_names,
            )

            agreement_df = agreement_scores(
                merged,
                criteria_names,
            )

            summary = pd.merge(
                corr_df,
                agreement_df,
                on="criterion",
                how="outer",
            )

            summary.insert(0, "language", lang)
            summary.insert(1, "comparison", comparison_name)
            summary.insert(2, "file1", os.path.basename(comparison["file1"]))
            summary.insert(3, "file2", os.path.basename(comparison["file2"]))

            all_summaries.append(summary)

            print(f"\nComparison: {comparison_name}")
            print(summary.to_string(index=False))


if all_summaries:
    final_summary = pd.concat(all_summaries, ignore_index=True)

    output_path = "results/LAAJ_Agreement_Correlation_short.tsv"
    final_summary.to_csv(output_path, index=False, sep="\t",
                         columns=['language',
                                  'comparison',
                                  'criterion',
                                  'spearman_r',
                                  'spearman_p',
                                  'quadratic_CohenK',
                                  #  'ICC_2_1_absolute'
                                  ])

    print(f"\n{'=' * 80}")
    print(f"Wrote all pairwise statistics to {output_path}")
    print(f"{'=' * 80}")

else:
    print("No summaries were generated.")

#plot_krippendorff_alpha_by_language(
#        final_summary,
#        output_dir="results/plots",
#        alpha_col="ordinal_Krippendorff_alpha",
#    )

plot_cohen_K_by_language(
        final_summary,
        output_dir="results/plots",
        alpha_col="quadratic_CohenK",
    )