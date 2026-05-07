import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REL_DIR = os.path.join(BASE_DIR, "REL")
REL_DATA_DIR = os.path.join(REL_DIR, "data")

INPUT_MASTER_LONG = os.path.join(REL_DATA_DIR, "pilot_comparison_master_long.xlsx")

OUT_SUMMARY_XLSX = os.path.join(REL_DATA_DIR, "output_layer_alignment_summary.xlsx")
OUT_SUMMARY_CSV = os.path.join(REL_DATA_DIR, "output_layer_alignment_summary.csv")
OUT_PNG = os.path.join(REL_DATA_DIR, "output_layer_dumbbell_plot.png")
OUT_SVG = os.path.join(REL_DATA_DIR, "output_layer_dumbbell_plot.svg")

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
N_A_CODE = 99

TARGET_OUTPUTS = [
    "local_pipeline_draft",
    "gemini_working",
    "gemini_authoritative",
    "gemini_conservative",
    "local_working",
    "local_authoritative",
    "local_conservative",
    "gemini_high_confidence",
    "local_high_confidence",
]

DISPLAY_NAMES = {
    "local_pipeline_draft": "Pipeline draft",
    "gemini_working": "Gemini working",
    "gemini_authoritative": "Gemini authoritative",
    "gemini_conservative": "Gemini conservative",
    "local_working": "Local working",
    "local_authoritative": "Local authoritative",
    "local_conservative": "Local conservative",
    "gemini_high_confidence": "Gemini high-confidence",
    "local_high_confidence": "Local high-confidence",
}

# ============================================================
# HELPERS
# ============================================================
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

def mode_with_tie_handling(values):
    vals = [v for v in values if not pd.isna(v)]
    if not vals:
        return np.nan
    counts = pd.Series(vals).value_counts()
    if len(counts) == 1:
        return counts.index[0]
    if counts.iloc[0] > counts.iloc[1]:
        return counts.index[0]
    return np.nan

def build_human_majority(df):
    humans = df[df["coder_type"] == "human"].copy()
    rows = []

    for article_id, g in humans.groupby("article_id"):
        row = {"article_id": article_id}
        for var in FINAL_VARS:
            vals = g[var].apply(to_int_or_nan).tolist()
            if var in SUBSTANTIVE_VARS:
                vals = [v for v in vals if not pd.isna(v) and v != N_A_CODE]
            else:
                vals = [v for v in vals if not pd.isna(v)]
            row[var] = mode_with_tie_handling(vals)
        rows.append(row)

    return pd.DataFrame(rows)

def prepare_source_table(df, coder_id):
    sub = df[df["coder_id"] == coder_id].copy()
    if sub.empty:
        return pd.DataFrame(columns=["article_id"] + FINAL_VARS)

    keep = ["article_id"] + FINAL_VARS
    keep = [c for c in keep if c in sub.columns]
    sub = sub[keep].drop_duplicates(subset=["article_id"]).copy()

    for var in FINAL_VARS:
        if var in sub.columns:
            sub[var] = sub[var].apply(to_int_or_nan)
        else:
            sub[var] = np.nan

    return sub

def compute_mean_agreement(source_df, ref_df):
    agreements = []
    ns = []

    for var in FINAL_VARS:
        left = source_df[["article_id", var]].copy()
        right = ref_df[["article_id", var]].copy()

        left[var] = left[var].apply(to_int_or_nan)
        right[var] = right[var].apply(to_int_or_nan)

        if var in SUBSTANTIVE_VARS:
            left = left[left[var] != N_A_CODE]
            right = right[right[var] != N_A_CODE]

        left = left.dropna(subset=[var])
        right = right.dropna(subset=[var])

        merged = left.merge(right, on="article_id", suffixes=("_src", "_ref"), how="inner")
        if merged.empty:
            continue

        agree = (merged[f"{var}_src"] == merged[f"{var}_ref"]).mean()
        agreements.append(float(agree))
        ns.append(len(merged))

    if not agreements:
        return np.nan, np.nan, 0

    mean_agree = float(np.mean(agreements))
    mean_distance = 1.0 - mean_agree
    mean_n = float(np.mean(ns))
    return mean_agree, mean_distance, mean_n

# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(INPUT_MASTER_LONG):
        raise FileNotFoundError(f"Missing input: {INPUT_MASTER_LONG}")

    df = pd.read_excel(INPUT_MASTER_LONG, dtype={"article_id": str})
    df["article_id"] = df["article_id"].apply(normalize_article_id)

    for var in FINAL_VARS:
        if var in df.columns:
            df[var] = df[var].apply(to_int_or_nan)
        else:
            df[var] = np.nan

    # build references
    human_majority = build_human_majority(df)
    pipeline_df = prepare_source_table(df, "local_pipeline_draft")

    rows = []

    for coder_id in TARGET_OUTPUTS:
        src = prepare_source_table(df, coder_id)
        if src.empty:
            continue

        agree_h, dist_h, n_h = compute_mean_agreement(src, human_majority)
        agree_p, dist_p, n_p = compute_mean_agreement(src, pipeline_df)

        rows.append({
            "coder_id": coder_id,
            "display_name": DISPLAY_NAMES.get(coder_id, coder_id),
            "mean_agreement_human_majority": agree_h,
            "mean_distance_human_majority": dist_h,
            "mean_n_human_majority": n_h,
            "mean_agreement_pipeline": agree_p,
            "mean_distance_pipeline": dist_p,
            "mean_n_pipeline": n_p,
        })

    df_out = pd.DataFrame(rows)

    # export tables
    df_out.to_excel(OUT_SUMMARY_XLSX, index=False)
    df_out.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # plot
    plot_df = df_out.copy()
    plot_df = plot_df.sort_values("mean_agreement_human_majority", ascending=True).reset_index(drop=True)

    y = np.arange(len(plot_df))

    plt.figure(figsize=(10, 6.5))
    ax = plt.gca()

    # dumbbell lines
    for i, row in plot_df.iterrows():
        ax.plot(
            [row["mean_agreement_pipeline"], row["mean_agreement_human_majority"]],
            [i, i],
            color="black",
            linewidth=1.8,
            zorder=1
        )

    # points
    ax.scatter(
        plot_df["mean_agreement_pipeline"],
        y,
        color="white",
        edgecolor="black",
        s=90,
        zorder=3,
        label="Agreement with pipeline"
    )

    ax.scatter(
        plot_df["mean_agreement_human_majority"],
        y,
        color="black",
        s=55,
        zorder=4,
        label="Agreement with human majority"
    )

    # labels
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"], fontsize=10)
    ax.set_xlabel("Mean agreement", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_title("Comparative alignment of output layers", fontsize=12)

    # style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.6)

    # legend
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_SVG, bbox_inches="tight")
    plt.close()

    print("\n=== OUTPUT LAYER ALIGNMENT SUMMARY ===")
    print(df_out[[
        "display_name",
        "mean_agreement_human_majority",
        "mean_distance_human_majority",
        "mean_agreement_pipeline",
        "mean_distance_pipeline"
    ]].to_string(index=False))

    print(f"\nSaved:")
    print(f"- {OUT_SUMMARY_XLSX}")
    print(f"- {OUT_SUMMARY_CSV}")
    print(f"- {OUT_PNG}")
    print(f"- {OUT_SVG}")

if __name__ == "__main__":
    main()