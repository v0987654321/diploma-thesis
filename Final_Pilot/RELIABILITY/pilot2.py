import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import cohen_kappa_score
import krippendorff

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REL_DIR = os.path.join(BASE_DIR, "REL")
REL_DATA_DIR = os.path.join(REL_DIR, "data")

INPUT_MASTER_LONG = os.path.join(REL_DATA_DIR, "pilot_comparison_master_long.xlsx")

OUT_ALPHA_XLSX = os.path.join(REL_DATA_DIR, "reliability_human_alpha.xlsx")
OUT_PAIRWISE_AGREEMENT_XLSX = os.path.join(REL_DATA_DIR, "reliability_pairwise_agreement.xlsx")
OUT_PAIRWISE_KAPPA_XLSX = os.path.join(REL_DATA_DIR, "reliability_pairwise_kappa.xlsx")

OUT_MODEL_VS_PIPELINE_XLSX = os.path.join(REL_DATA_DIR, "reliability_model_vs_pipeline.xlsx")
OUT_MODEL_VS_HUMAN_MAJ_XLSX = os.path.join(REL_DATA_DIR, "reliability_model_vs_human_majority.xlsx")

OUT_BRANCH_VS_PIPELINE_XLSX = os.path.join(REL_DATA_DIR, "reliability_operational_branches_vs_pipeline.xlsx")
OUT_BRANCH_VS_HUMAN_MAJ_XLSX = os.path.join(REL_DATA_DIR, "reliability_operational_branches_vs_human_majority.xlsx")

OUT_HIGHCONF_VS_PIPELINE_XLSX = os.path.join(REL_DATA_DIR, "reliability_highconf_vs_pipeline.xlsx")
OUT_HIGHCONF_VS_HUMAN_MAJ_XLSX = os.path.join(REL_DATA_DIR, "reliability_highconf_vs_human_majority.xlsx")

OUT_VARIABLE_SUMMARY_XLSX = os.path.join(REL_DATA_DIR, "reliability_variable_summary.xlsx")
OUT_NA_DIAGNOSTICS_XLSX = os.path.join(REL_DATA_DIR, "reliability_na_diagnostics.xlsx")
OUT_COVERAGE_DIAGNOSTICS_XLSX = os.path.join(REL_DATA_DIR, "reliability_coverage_diagnostics.xlsx")

N_A_CODE = 99

FINAL_VARS = [
    "V04_Relevance_Final",
    "V05_Actor_Mention_Final",
    "V06_Successor_Frame_Final",
    "V07_Dominant_Label_Final",
    "V08_Stance_Final",
    "V09_Dominant_Location_Final",
    "V10_Ambivalence_Final",
    "V11_Legitimation_Final",
    "V12_Counterterrorism_Final",
    "V13_Sovereignty_Final",
    "V14_Human_Rights_Abuse_Final",
    "V15_Anti_or_Neocolonialism_Final",
    "V16_Western_Failure_Final",
    "V17_Security_Effectiveness_Final",
    "V18_Economic_Interests_Final",
    "V19_Geopolitical_Rivalry_Final",
    "V20_Main_Associated_Actor_Final",
    "V21_Dominant_Discourse_Final",
]

SUBSTANTIVE_VARS = [v for v in FINAL_VARS if v != "V04_Relevance_Final"]

# ============================================================
# HELPERS
# ============================================================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def normalize_article_id(x):
    if pd.isna(x) or x is None:
        return None
    s = str(x).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.zfill(6) if digits else None

def to_int_or_nan(x):
    if pd.isna(x) or x is None:
        return np.nan
    try:
        return int(float(x))
    except Exception:
        return np.nan

def percent_agreement(a, b):
    if len(a) == 0:
        return np.nan
    a = np.array(a)
    b = np.array(b)
    return float(np.mean(a == b))

def safe_kappa(a, b):
    if len(a) == 0:
        return np.nan
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return np.nan
    try:
        return float(cohen_kappa_score(a, b, labels=labels))
    except Exception:
        return np.nan

def mode_with_tie_handling(values):
    vals = [v for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    c = Counter(vals)
    mc = c.most_common()
    if len(mc) == 1:
        return mc[0][0]
    if mc[0][1] > mc[1][1]:
        return mc[0][0]
    return np.nan

def prepare_variable_subset(df, var, exclude_na=True):
    cols = ["article_id", "coder_id", "coder_type", "source_group", var]
    cols = [c for c in cols if c in df.columns]
    tmp = df[cols].copy()
    tmp = tmp.rename(columns={var: "value"})
    tmp["value"] = tmp["value"].apply(to_int_or_nan)
    tmp = tmp.dropna(subset=["article_id"])

    if exclude_na and var in SUBSTANTIVE_VARS:
        tmp = tmp[tmp["value"] != N_A_CODE]

    tmp = tmp.dropna(subset=["value"])
    return tmp

def krippendorff_alpha_from_long(df_long, var, coder_filter=None, exclude_na=True):
    tmp = prepare_variable_subset(df_long, var, exclude_na=exclude_na)
    if coder_filter is not None:
        tmp = coder_filter(tmp)

    if tmp.empty:
        return np.nan, 0, 0, "empty"

    pivot = tmp.pivot_table(index="article_id", columns="coder_id", values="value", aggfunc="first")
    if pivot.shape[1] < 2:
        return np.nan, pivot.shape[0], pivot.shape[1], "fewer_than_2_coders"

    matrix = pivot.to_numpy().T

    try:
        alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement="nominal")
        return alpha, pivot.shape[0], pivot.shape[1], "ok"
    except Exception as e:
        return np.nan, pivot.shape[0], pivot.shape[1], f"error: {e}"

def build_human_majority(df_long, exclude_na_main=True):
    rows = []
    humans = df_long[df_long["coder_type"] == "human"].copy()

    if humans.empty:
        return pd.DataFrame(columns=["article_id"] + FINAL_VARS)

    for article_id, g in humans.groupby("article_id"):
        row = {"article_id": article_id}
        for var in FINAL_VARS:
            vals = g[var].apply(to_int_or_nan).tolist()

            if var in SUBSTANTIVE_VARS and exclude_na_main:
                vals = [v for v in vals if not pd.isna(v) and v != N_A_CODE]
            else:
                vals = [v for v in vals if not pd.isna(v)]

            row[var] = mode_with_tie_handling(vals)
        rows.append(row)

    return pd.DataFrame(rows)

def pairwise_against_reference(df_targets, df_reference, var, comparison_type, target_label_col="coder_id"):
    results = []

    if df_targets.empty or df_reference.empty:
        return pd.DataFrame()

    tmp_t = prepare_variable_subset(df_targets, var, exclude_na=(var in SUBSTANTIVE_VARS))
    if tmp_t.empty:
        return pd.DataFrame()

    tmp_r = df_reference[["article_id", var]].copy()
    tmp_r[var] = tmp_r[var].apply(to_int_or_nan)

    if var in SUBSTANTIVE_VARS:
        tmp_r = tmp_r[tmp_r[var] != N_A_CODE]

    tmp_r = tmp_r.dropna(subset=[var]).rename(columns={var: "ref_value"})
    if tmp_r.empty:
        return pd.DataFrame()

    ref_total_articles = tmp_r["article_id"].nunique()
    merged = tmp_t.merge(tmp_r, on="article_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    for target_id, g in merged.groupby(target_label_col):
        vals_t = g["value"].tolist()
        vals_r = g["ref_value"].tolist()

        target_total_articles = tmp_t[tmp_t[target_label_col] == target_id]["article_id"].nunique()
        n_compared = len(vals_t)

        agree = percent_agreement(vals_t, vals_r)
        kappa = safe_kappa(vals_t, vals_r)

        results.append({
            "Comparison_Type": comparison_type,
            "Variable": var,
            "Target_ID": target_id,
            "N_Compared": n_compared,
            "N_Target_Total": target_total_articles,
            "N_Reference_Total": ref_total_articles,
            "Coverage_of_Target_Compared": n_compared / target_total_articles if target_total_articles > 0 else np.nan,
            "Coverage_of_Reference_Compared": n_compared / ref_total_articles if ref_total_articles > 0 else np.nan,
            "Percent_Agreement": agree,
            "Cohen_Kappa": kappa,
        })

    return pd.DataFrame(results)

def pairwise_between_targets_and_pipeline(df_targets, df_pipeline, var, comparison_type, label_col="coder_id"):
    results = []

    if df_targets.empty or df_pipeline.empty:
        return pd.DataFrame()

    tmp_t = prepare_variable_subset(df_targets, var, exclude_na=(var in SUBSTANTIVE_VARS))
    tmp_p = prepare_variable_subset(df_pipeline, var, exclude_na=(var in SUBSTANTIVE_VARS))

    if tmp_t.empty or tmp_p.empty:
        return pd.DataFrame()

    tmp_p = tmp_p[["article_id", "value"]].drop_duplicates().rename(columns={"value": "pipeline_value"})
    pipeline_total_articles = tmp_p["article_id"].nunique()

    merged = tmp_t.merge(tmp_p, on="article_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    for target_id, g in merged.groupby(label_col):
        vals_t = g["value"].tolist()
        vals_p = g["pipeline_value"].tolist()

        target_total_articles = tmp_t[tmp_t[label_col] == target_id]["article_id"].nunique()
        n_compared = len(vals_t)

        agree = percent_agreement(vals_t, vals_p)
        kappa = safe_kappa(vals_t, vals_p)

        results.append({
            "Comparison_Type": comparison_type,
            "Variable": var,
            "Target_ID": target_id,
            "N_Compared": n_compared,
            "N_Target_Total": target_total_articles,
            "N_Reference_Total": pipeline_total_articles,
            "Coverage_of_Target_Compared": n_compared / target_total_articles if target_total_articles > 0 else np.nan,
            "Coverage_of_Reference_Compared": n_compared / pipeline_total_articles if pipeline_total_articles > 0 else np.nan,
            "Percent_Agreement": agree,
            "Cohen_Kappa": kappa,
        })

    return pd.DataFrame(results)

def na_diagnostics(df_long):
    rows = []
    for var in FINAL_VARS:
        for (coder_type, coder_id), g in df_long.groupby(["coder_type", "coder_id"]):
            vals = g[var].apply(to_int_or_nan)
            n_total = vals.notna().sum()
            n_na = (vals == N_A_CODE).sum()
            share_na = (n_na / n_total) if n_total > 0 else np.nan

            rows.append({
                "Variable": var,
                "Coder_Type": coder_type,
                "Coder_ID": coder_id,
                "N_Total_NonMissing": n_total,
                "N_Explicit_NA_99": n_na,
                "Share_Explicit_NA_99": share_na,
            })
    return pd.DataFrame(rows)

def coverage_diagnostics(df_long):
    rows = []
    for (coder_type, coder_id), g in df_long.groupby(["coder_type", "coder_id"]):
        rows.append({
            "Coder_Type": coder_type,
            "Coder_ID": coder_id,
            "N_Unique_Articles": g["article_id"].nunique(),
            "N_Rows": len(g),
        })
    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(INPUT_MASTER_LONG):
        raise FileNotFoundError(f"Missing input file: {INPUT_MASTER_LONG}")

    df = pd.read_excel(INPUT_MASTER_LONG, dtype={"article_id": str})
    df["article_id"] = df["article_id"].apply(normalize_article_id)

    for var in FINAL_VARS:
        if var in df.columns:
            df[var] = df[var].apply(to_int_or_nan)
        else:
            df[var] = np.nan

    if "coder_type" not in df.columns:
        raise ValueError("Expected column 'coder_type' in pilot_comparison_master_long.xlsx")
    if "coder_id" not in df.columns:
        raise ValueError("Expected column 'coder_id' in pilot_comparison_master_long.xlsx")

    # Main groups
    df_humans = df[df["coder_type"] == "human"].copy()
    df_benchmark_models = df[df["coder_type"] == "benchmark_model"].copy()
    df_pipeline = df[df["coder_type"] == "pipeline"].copy()
    df_operational = df[df["coder_type"] == "llm_operational"].copy()
    df_highconf = df[df["coder_type"] == "high_confidence_branch"].copy()

    # Human majority
    df_human_majority = build_human_majority(df, exclude_na_main=True)

    # ========================================================
    # 1. HUMAN-ONLY KRIPPENDORFF ALPHA
    # ========================================================
    alpha_rows = []
    for var in FINAL_VARS:
        alpha, n_articles, n_coders, status = krippendorff_alpha_from_long(
            df_humans, var, coder_filter=None, exclude_na=True
        )
        alpha_rows.append({
            "Variable": var,
            "Krippendorff_Alpha": alpha,
            "N_Articles_Used": n_articles,
            "N_Coders_Used": n_coders,
            "Status": status,
        })

    df_alpha = pd.DataFrame(alpha_rows)
    df_alpha.to_excel(OUT_ALPHA_XLSX, index=False)

    # ========================================================
    # 2. BENCHMARK MODELS VS PIPELINE
    # ========================================================
    model_vs_pipeline = []
    for var in FINAL_VARS:
        res = pairwise_between_targets_and_pipeline(
            df_benchmark_models, df_pipeline, var, "benchmark_model_vs_pipeline", label_col="coder_id"
        )
        if not res.empty:
            model_vs_pipeline.append(res)

    df_model_vs_pipeline = pd.concat(model_vs_pipeline, ignore_index=True) if model_vs_pipeline else pd.DataFrame()
    if df_model_vs_pipeline.empty:
        df_model_vs_pipeline = pd.DataFrame(columns=[
            "Comparison_Type", "Variable", "Target_ID", "N_Compared", "N_Target_Total", "N_Reference_Total",
            "Coverage_of_Target_Compared", "Coverage_of_Reference_Compared", "Percent_Agreement", "Cohen_Kappa"
        ])
    df_model_vs_pipeline.to_excel(OUT_MODEL_VS_PIPELINE_XLSX, index=False)

    # ========================================================
    # 3. BENCHMARK MODELS VS HUMAN MAJORITY
    # ========================================================
    model_vs_human_maj = []
    for var in FINAL_VARS:
        res = pairwise_against_reference(
            df_benchmark_models, df_human_majority, var, "benchmark_model_vs_human_majority", target_label_col="coder_id"
        )
        if not res.empty:
            model_vs_human_maj.append(res)

    df_model_vs_human_maj = pd.concat(model_vs_human_maj, ignore_index=True) if model_vs_human_maj else pd.DataFrame()
    if df_model_vs_human_maj.empty:
        df_model_vs_human_maj = pd.DataFrame(columns=[
            "Comparison_Type", "Variable", "Target_ID", "N_Compared", "N_Target_Total", "N_Reference_Total",
            "Coverage_of_Target_Compared", "Coverage_of_Reference_Compared", "Percent_Agreement", "Cohen_Kappa"
        ])
    df_model_vs_human_maj.to_excel(OUT_MODEL_VS_HUMAN_MAJ_XLSX, index=False)

    # ========================================================
    # 4. OPERATIONAL BRANCHES VS PIPELINE
    # ========================================================
    branch_vs_pipeline = []
    for var in FINAL_VARS:
        res = pairwise_between_targets_and_pipeline(
            df_operational, df_pipeline, var, "operational_branch_vs_pipeline", label_col="coder_id"
        )
        if not res.empty:
            branch_vs_pipeline.append(res)

    df_branch_vs_pipeline = pd.concat(branch_vs_pipeline, ignore_index=True) if branch_vs_pipeline else pd.DataFrame()
    if df_branch_vs_pipeline.empty:
        df_branch_vs_pipeline = pd.DataFrame(columns=[
            "Comparison_Type", "Variable", "Target_ID", "N_Compared", "N_Target_Total", "N_Reference_Total",
            "Coverage_of_Target_Compared", "Coverage_of_Reference_Compared", "Percent_Agreement", "Cohen_Kappa"
        ])
    df_branch_vs_pipeline.to_excel(OUT_BRANCH_VS_PIPELINE_XLSX, index=False)

    # ========================================================
    # 5. OPERATIONAL BRANCHES VS HUMAN MAJORITY
    # ========================================================
    branch_vs_human_maj = []
    for var in FINAL_VARS:
        res = pairwise_against_reference(
            df_operational, df_human_majority, var, "operational_branch_vs_human_majority", target_label_col="coder_id"
        )
        if not res.empty:
            branch_vs_human_maj.append(res)

    df_branch_vs_human_maj = pd.concat(branch_vs_human_maj, ignore_index=True) if branch_vs_human_maj else pd.DataFrame()
    if df_branch_vs_human_maj.empty:
        df_branch_vs_human_maj = pd.DataFrame(columns=[
            "Comparison_Type", "Variable", "Target_ID", "N_Compared", "N_Target_Total", "N_Reference_Total",
            "Coverage_of_Target_Compared", "Coverage_of_Reference_Compared", "Percent_Agreement", "Cohen_Kappa"
        ])
    df_branch_vs_human_maj.to_excel(OUT_BRANCH_VS_HUMAN_MAJ_XLSX, index=False)

    # ========================================================
    # 6. HIGH-CONFIDENCE VS PIPELINE
    # ========================================================
    highconf_vs_pipeline = []
    for var in FINAL_VARS:
        res = pairwise_between_targets_and_pipeline(
            df_highconf, df_pipeline, var, "high_confidence_branch_vs_pipeline", label_col="coder_id"
        )
        if not res.empty:
            highconf_vs_pipeline.append(res)

    df_highconf_vs_pipeline = pd.concat(highconf_vs_pipeline, ignore_index=True) if highconf_vs_pipeline else pd.DataFrame()
    if df_highconf_vs_pipeline.empty:
        df_highconf_vs_pipeline = pd.DataFrame(columns=[
            "Comparison_Type", "Variable", "Target_ID", "N_Compared", "N_Target_Total", "N_Reference_Total",
            "Coverage_of_Target_Compared", "Coverage_of_Reference_Compared", "Percent_Agreement", "Cohen_Kappa"
        ])
    df_highconf_vs_pipeline.to_excel(OUT_HIGHCONF_VS_PIPELINE_XLSX, index=False)

    # ========================================================
    # 7. HIGH-CONFIDENCE VS HUMAN MAJORITY
    # ========================================================
    highconf_vs_human_maj = []
    for var in FINAL_VARS:
        res = pairwise_against_reference(
            df_highconf, df_human_majority, var, "high_confidence_branch_vs_human_majority", target_label_col="coder_id"
        )
        if not res.empty:
            highconf_vs_human_maj.append(res)

    df_highconf_vs_human_maj = pd.concat(highconf_vs_human_maj, ignore_index=True) if highconf_vs_human_maj else pd.DataFrame()
    if df_highconf_vs_human_maj.empty:
        df_highconf_vs_human_maj = pd.DataFrame(columns=[
            "Comparison_Type", "Variable", "Target_ID", "N_Compared", "N_Target_Total", "N_Reference_Total",
            "Coverage_of_Target_Compared", "Coverage_of_Reference_Compared", "Percent_Agreement", "Cohen_Kappa"
        ])
    df_highconf_vs_human_maj.to_excel(OUT_HIGHCONF_VS_HUMAN_MAJ_XLSX, index=False)

    # ========================================================
    # 8. COMBINED PAIRWISE TABLES
    # ========================================================
    df_pairwise_agreement = pd.concat(
        [
            df_model_vs_pipeline,
            df_model_vs_human_maj,
            df_branch_vs_pipeline,
            df_branch_vs_human_maj,
            df_highconf_vs_pipeline,
            df_highconf_vs_human_maj,
        ],
        ignore_index=True
    )

    if df_pairwise_agreement.empty:
        df_pairwise_agreement = pd.DataFrame(columns=[
            "Comparison_Type", "Variable", "Target_ID", "N_Compared", "N_Target_Total", "N_Reference_Total",
            "Coverage_of_Target_Compared", "Coverage_of_Reference_Compared", "Percent_Agreement", "Cohen_Kappa"
        ])

    df_pairwise_agreement.to_excel(OUT_PAIRWISE_AGREEMENT_XLSX, index=False)
    df_pairwise_agreement.to_excel(OUT_PAIRWISE_KAPPA_XLSX, index=False)

    # ========================================================
    # 9. N/A + COVERAGE diagnostics
    # ========================================================
    df_na = na_diagnostics(df)
    df_na.to_excel(OUT_NA_DIAGNOSTICS_XLSX, index=False)

    df_cov = coverage_diagnostics(df)
    df_cov.to_excel(OUT_COVERAGE_DIAGNOSTICS_XLSX, index=False)

    # ========================================================
    # 10. VARIABLE SUMMARY
    # ========================================================
    summary_rows = []

    for var in FINAL_VARS:
        alpha_val = df_alpha.loc[df_alpha["Variable"] == var, "Krippendorff_Alpha"]
        alpha_val = alpha_val.iloc[0] if not alpha_val.empty else np.nan

        tmp = df_pairwise_agreement[df_pairwise_agreement["Variable"] == var].copy()

        summary_rows.append({
            "Variable": var,
            "Human_Alpha": alpha_val,
            "Mean_Pairwise_Agreement": tmp["Percent_Agreement"].mean() if not tmp.empty else np.nan,
            "Mean_Pairwise_Kappa": tmp["Cohen_Kappa"].mean() if not tmp.empty else np.nan,
            "Mean_N_Compared": tmp["N_Compared"].mean() if not tmp.empty else np.nan,
            "Best_Agreement_Target": tmp.sort_values("Percent_Agreement", ascending=False)["Target_ID"].iloc[0] if not tmp.empty else "",
            "Worst_Agreement_Target": tmp.sort_values("Percent_Agreement", ascending=True)["Target_ID"].iloc[0] if not tmp.empty else "",
            "N_Pairwise_Comparisons": len(tmp),
        })

    df_var_summary = pd.DataFrame(summary_rows)
    df_var_summary.to_excel(OUT_VARIABLE_SUMMARY_XLSX, index=False)

    # ========================================================
    # 11. CONSOLE SUMMARY
    # ========================================================
    print("\n================ RELIABILITY SUMMARY ================\n")

    print("BASIC SOURCE COUNTS")
    print("-------------------")
    print(f"Unique pilot articles in master table: {df['article_id'].nunique()}")
    print(f"Human coder rows: {len(df_humans)}")
    print(f"Benchmark model rows: {len(df_benchmark_models)}")
    print(f"Pipeline rows: {len(df_pipeline)}")
    print(f"Operational LLM rows: {len(df_operational)}")
    print(f"High-confidence rows: {len(df_highconf)}")
    print(f"High-confidence unique articles: {df_highconf['article_id'].nunique() if not df_highconf.empty else 0}")

    print("\nCoverage diagnostics by source:")
    print(df_cov.sort_values(["Coder_Type", "Coder_ID"]).to_string(index=False))

    print("\nHUMAN-ONLY KRIPPENDORFF ALPHA (per variable)")
    print("--------------------------------------------")
    print(df_alpha.sort_values("Krippendorff_Alpha", ascending=False).to_string(index=False))

    weak_067 = df_alpha[df_alpha["Krippendorff_Alpha"] < 0.67]
    weak_050 = df_alpha[df_alpha["Krippendorff_Alpha"] < 0.50]

    print("\nVariables with alpha < 0.67")
    print("---------------------------")
    if weak_067.empty:
        print("None")
    else:
        print(weak_067[["Variable", "Krippendorff_Alpha", "N_Articles_Used"]].to_string(index=False))

    print("\nVariables with alpha < 0.50")
    print("---------------------------")
    if weak_050.empty:
        print("None")
    else:
        print(weak_050[["Variable", "Krippendorff_Alpha", "N_Articles_Used"]].to_string(index=False))

    print("\nBENCHMARK MODELS VS PIPELINE (average across variables)")
    print("------------------------------------------------------")
    if not df_model_vs_pipeline.empty:
        model_pipe_avg = (
            df_model_vs_pipeline.groupby("Target_ID")[["Percent_Agreement", "Cohen_Kappa", "N_Compared"]]
            .mean()
            .sort_values("Percent_Agreement", ascending=False)
        )
        print(model_pipe_avg.to_string())
    else:
        print("No benchmark model vs pipeline comparisons available.")

    print("\nBENCHMARK MODELS VS HUMAN MAJORITY (average across variables)")
    print("------------------------------------------------------------")
    if not df_model_vs_human_maj.empty:
        model_hmaj_avg = (
            df_model_vs_human_maj.groupby("Target_ID")[["Percent_Agreement", "Cohen_Kappa", "N_Compared"]]
            .mean()
            .sort_values("Percent_Agreement", ascending=False)
        )
        print(model_hmaj_avg.to_string())
    else:
        print("No benchmark model vs human majority comparisons available.")

    print("\nOPERATIONAL BRANCHES VS PIPELINE (average across variables)")
    print("----------------------------------------------------------")
    if not df_branch_vs_pipeline.empty:
        branch_pipe_avg = (
            df_branch_vs_pipeline.groupby("Target_ID")[["Percent_Agreement", "Cohen_Kappa", "N_Compared"]]
            .mean()
            .sort_values("Percent_Agreement", ascending=False)
        )
        print(branch_pipe_avg.to_string())
    else:
        print("No operational branch vs pipeline comparisons available.")

    print("\nOPERATIONAL BRANCHES VS HUMAN MAJORITY (average across variables)")
    print("----------------------------------------------------------------")
    if not df_branch_vs_human_maj.empty:
        branch_hmaj_avg = (
            df_branch_vs_human_maj.groupby("Target_ID")[["Percent_Agreement", "Cohen_Kappa", "N_Compared"]]
            .mean()
            .sort_values("Percent_Agreement", ascending=False)
        )
        print(branch_hmaj_avg.to_string())
    else:
        print("No operational branch vs human majority comparisons available.")

    print("\nHIGH-CONFIDENCE BRANCHES VS PIPELINE (average across variables)")
    print("--------------------------------------------------------------")
    if not df_highconf_vs_pipeline.empty:
        high_pipe_avg = (
            df_highconf_vs_pipeline.groupby("Target_ID")[["Percent_Agreement", "Cohen_Kappa", "N_Compared"]]
            .mean()
            .sort_values("Percent_Agreement", ascending=False)
        )
        print(high_pipe_avg.to_string())
    else:
        print("No high-confidence vs pipeline comparisons available.")

    print("\nHIGH-CONFIDENCE BRANCHES VS HUMAN MAJORITY (average across variables)")
    print("--------------------------------------------------------------------")
    if not df_highconf_vs_human_maj.empty:
        high_hmaj_avg = (
            df_highconf_vs_human_maj.groupby("Target_ID")[["Percent_Agreement", "Cohen_Kappa", "N_Compared"]]
            .mean()
            .sort_values("Percent_Agreement", ascending=False)
        )
        print(high_hmaj_avg.to_string())
    else:
        print("No high-confidence vs human majority comparisons available.")

    print("\nVARIABLES SORTED BY MEAN PAIRWISE AGREEMENT")
    print("-------------------------------------------")
    print(df_var_summary.sort_values("Mean_Pairwise_Agreement", ascending=False)[
        ["Variable", "Human_Alpha", "Mean_Pairwise_Agreement", "Mean_Pairwise_Kappa", "Mean_N_Compared"]
    ].to_string(index=False))

    print("\nTOP N/A DIAGNOSTICS")
    print("-------------------")
    na_summary = (
        df_na.groupby("Variable")[["N_Explicit_NA_99", "N_Total_NonMissing"]]
        .sum()
        .reset_index()
    )
    na_summary["Share_Explicit_NA_99"] = na_summary["N_Explicit_NA_99"] / na_summary["N_Total_NonMissing"]
    print(na_summary.sort_values("Share_Explicit_NA_99", ascending=False).to_string(index=False))

    print("\nSaved files:")
    print(f"- {OUT_ALPHA_XLSX}")
    print(f"- {OUT_PAIRWISE_AGREEMENT_XLSX}")
    print(f"- {OUT_MODEL_VS_PIPELINE_XLSX}")
    print(f"- {OUT_MODEL_VS_HUMAN_MAJ_XLSX}")
    print(f"- {OUT_BRANCH_VS_PIPELINE_XLSX}")
    print(f"- {OUT_BRANCH_VS_HUMAN_MAJ_XLSX}")
    print(f"- {OUT_HIGHCONF_VS_PIPELINE_XLSX}")
    print(f"- {OUT_HIGHCONF_VS_HUMAN_MAJ_XLSX}")
    print(f"- {OUT_VARIABLE_SUMMARY_XLSX}")
    print(f"- {OUT_NA_DIAGNOSTICS_XLSX}")
    print(f"- {OUT_COVERAGE_DIAGNOSTICS_XLSX}")

if __name__ == "__main__":
    main()