import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REL_DIR = os.path.join(BASE_DIR, "REL")
REL_DATA_DIR = os.path.join(REL_DIR, "data")

INPUT_MASTER_LONG = os.path.join(REL_DATA_DIR, "pilot_comparison_master_long.xlsx")

OUT_SUMMARY_XLSX = os.path.join(REL_DATA_DIR, "pilot3_output_layer_selection_summary.xlsx")
OUT_SUMMARY_CSV = os.path.join(REL_DATA_DIR, "pilot3_output_layer_selection_summary.csv")

OUT_DUMBBELL_PNG = os.path.join(REL_DATA_DIR, "pilot3_dumbbell_plot.png")
OUT_DUMBBELL_SVG = os.path.join(REL_DATA_DIR, "pilot3_dumbbell_plot.svg")

OUT_BENCH_HEATMAP_PNG = os.path.join(REL_DATA_DIR, "pilot3_benchmark_heatmap.png")
OUT_BENCH_HEATMAP_SVG = os.path.join(REL_DATA_DIR, "pilot3_benchmark_heatmap.svg")

OUT_FULL_MATRIX_XLSX = os.path.join(REL_DATA_DIR, "pilot3_full_distance_matrix.xlsx")
OUT_FULL_HEATMAP_PNG = os.path.join(REL_DATA_DIR, "pilot3_full_distance_heatmap.png")
OUT_FULL_HEATMAP_SVG = os.path.join(REL_DATA_DIR, "pilot3_full_distance_heatmap.svg")

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

OUTPUT_LAYERS = [
    "local_pipeline_draft",
    "gemini_authoritative",
    "gemini_conservative",
    "local_authoritative",
    "local_conservative",
    "gemini_high_confidence",
    "local_high_confidence",
]

BENCHMARK_MODELS = [
    "GPT_5_2",
    "claude_haiku_4_5",
    "gemini_3_fast",
    "gemini_3_thinking",
    "grok_fast",
]

HUMAN_CODERS = [
    "zlaticko1",
    "zlaticko2",
    "zlaticko3",
    "zlaticko4",
    "zlaticko5",
]

CRITICAL_VARS = [
    "V08_Stance_Final",
    "V11_Legitimation_Final",
    "V19_Geopolitical_Rivalry_Final",
    "V21_Dominant_Discourse_Final",
]

WEIGHTED_VARS = {
    "V04_Relevance_Final": 2,
    "V08_Stance_Final": 2,
    "V11_Legitimation_Final": 2,
    "V19_Geopolitical_Rivalry_Final": 2,
    "V21_Dominant_Discourse_Final": 2,
}
DEFAULT_WEIGHT = 1

DISPLAY_NAMES = {
    "local_pipeline_draft": "Pre-LLM pipeline draft",
    "gemini_authoritative": "Gemini authoritative",
    "gemini_conservative": "Gemini conservative",
    "local_authoritative": "Local authoritative",
    "local_conservative": "Local conservative",
    "gemini_high_confidence": "Gemini high-confidence",
    "local_high_confidence": "Local high-confidence",

    "human_majority": "Human majority",

    "zlaticko1": "Human 1",
    "zlaticko2": "Human 2",
    "zlaticko3": "Human 3",
    "zlaticko4": "Human 4",
    "zlaticko5": "Human 5",

    "GPT_5_2": "GPT_5_2",
    "claude_haiku_4_5": "Claude Haiku",
    "gemini_3_fast": "Gemini 3 Fast",
    "gemini_3_thinking": "Gemini 3 Thinking",
    "grok_fast": "Grok Fast",
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

def compute_var_agreement(source_df, ref_df, var):
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
        return np.nan, 0

    a = merged[f"{var}_src"].tolist()
    b = merged[f"{var}_ref"].tolist()
    agree = float(np.mean(np.array(a) == np.array(b)))
    return agree, len(merged)

def compute_mean_agreement(source_df, ref_df, var_list=None, weights=None):
    if var_list is None:
        var_list = FINAL_VARS

    agreements = []
    ns = []
    weighted_scores = []
    weight_values = []

    for var in var_list:
        agree, n = compute_var_agreement(source_df, ref_df, var)
        if pd.isna(agree):
            continue

        agreements.append(agree)
        ns.append(n)

        w = weights.get(var, DEFAULT_WEIGHT) if weights else 1
        weighted_scores.append(agree * w)
        weight_values.append(w)

    if not agreements:
        return np.nan, np.nan, np.nan

    mean_agree = float(np.mean(agreements))
    mean_n = float(np.mean(ns)) if ns else np.nan
    weighted_mean = float(np.sum(weighted_scores) / np.sum(weight_values)) if weight_values else np.nan

    return mean_agree, 1.0 - mean_agree, mean_n, weighted_mean

def compute_mean_to_benchmark_models(source_df, benchmark_tables):
    agreements = []
    distances = []
    ns = []

    for model_id, model_df in benchmark_tables.items():
        if model_df.empty:
            continue
        a, d, n, _ = compute_mean_agreement(source_df, model_df)
        agreements.append(a)
        distances.append(d)
        ns.append(n)

    if not agreements:
        return np.nan, np.nan, np.nan

    return (
        float(np.nanmean(agreements)),
        float(np.nanmean(distances)),
        float(np.nanmean(ns))
    )

def balance_score(human_agreement, pipeline_agreement):
    if pd.isna(human_agreement) or pd.isna(pipeline_agreement):
        return np.nan
    return (human_agreement + pipeline_agreement) / 2

def recommendation_label(row):
    h = row["agreement_human_majority"]
    p = row["agreement_pipeline"]
    c = row["agreement_critical_subset"]
    b = row["balance_score"]

    if pd.isna(h) or pd.isna(p):
        return "insufficient_data"

    if "high-confidence" in row["display_name"].lower():
        return "supplementary_only"

    if h >= 0.78 and p >= 0.82 and c >= 0.65 and b >= 0.80:
        return "selected_main_output"

    if h >= 0.75 and p >= 0.90:
        return "strong_alternative"

    if p >= 0.94 and h < 0.77:
        return "pipeline_like_alternative"

    if h >= 0.80 and p < 0.78:
        return "human_like_but_less_stable"

    return "comparison_layer"

# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.exists(INPUT_MASTER_LONG):
        raise FileNotFoundError(f"Missing input: {INPUT_MASTER_LONG}")

    print("Loading merged pilot comparison dataset...")
    df = pd.read_excel(INPUT_MASTER_LONG, dtype={"article_id": str})
    df["article_id"] = df["article_id"].apply(normalize_article_id)

    for var in FINAL_VARS:
        if var in df.columns:
            df[var] = df[var].apply(to_int_or_nan)
        else:
            df[var] = np.nan

    print("Building human-majority reference...")
    human_majority = build_human_majority(df)
    print("Building Pre-LLM pipeline reference...")
    pipeline_df = prepare_source_table(df, "local_pipeline_draft")

    print("Preparing benchmark model reference tables...")
    benchmark_tables = {}
    for mid in tqdm(BENCHMARK_MODELS, desc="Benchmark models", ncols=90):
        benchmark_tables[mid] = prepare_source_table(df, mid)

    # ========================================================
    # SUMMARY TABLE
    # ========================================================
    print("Computing output-layer selection summary...")
    rows = []
    for out_id in tqdm(OUTPUT_LAYERS, desc="Output layers", ncols=90):
        src = prepare_source_table(df, out_id)
        if src.empty:
            continue

        ah, dh, nh, _ = compute_mean_agreement(src, human_majority)
        ap, dp, np_, _ = compute_mean_agreement(src, pipeline_df)
        am, dm, nm = compute_mean_to_benchmark_models(src, benchmark_tables)

        ac, dc, nc, _ = compute_mean_agreement(
            src,
            human_majority,
            var_list=CRITICAL_VARS,
            weights=WEIGHTED_VARS
        )

        _, _, _, ww = compute_mean_agreement(
            src,
            human_majority,
            var_list=FINAL_VARS,
            weights=WEIGHTED_VARS
        )

        row = {
            "output_layer": out_id,
            "display_name": DISPLAY_NAMES.get(out_id, out_id),

            "agreement_human_majority": ah,
            "distance_human_majority": dh,
            "mean_n_human_majority": nh,

            "agreement_benchmark_models": am,
            "distance_benchmark_models": dm,
            "mean_n_benchmark_models": nm,

            "agreement_pipeline": ap,
            "distance_pipeline": dp,
            "mean_n_pipeline": np_,

            "agreement_critical_subset": ac,
            "distance_critical_subset": dc,
            "mean_n_critical_subset": nc,

            "weighted_overall_score": ww,
            "balance_score": balance_score(ah, ap),
        }

        row["recommendation"] = recommendation_label(row)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_excel(OUT_SUMMARY_XLSX, index=False)
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # ========================================================
    # DUMBBELL PLOT
    # ========================================================
    print("Rendering dumbbell plot...")
    plot_df = df_summary.copy()
    plot_df = plot_df.sort_values("agreement_human_majority", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    def point_color(layer):
        if layer == "gemini_conservative":
            return "#1f77b4"
        elif "high-confidence" in layer:
            return "#7f7f7f"
        elif layer == "local_pipeline_draft":
            return "#000000"
        elif layer.startswith("gemini"):
            return "#4c4c4c"
        elif layer.startswith("local"):
            return "#9a9a9a"
        return "#555555"

    for i, row in plot_df.iterrows():
        ax.plot(
            [row["agreement_pipeline"], row["agreement_human_majority"]],
            [i, i],
            color="#808080",
            linewidth=1.6,
            zorder=1
        )

    ax.scatter(
        plot_df["agreement_pipeline"],
        y,
        s=95,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=3,
        label="Agreement with Pre-LLM pipeline draft"
    )

    colors = [point_color(c) for c in plot_df["output_layer"]]
    ax.scatter(
        plot_df["agreement_human_majority"],
        y,
        s=65,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label="Agreement with human majority"
    )

    for i, row in plot_df.iterrows():
        if row["output_layer"] == "gemini_conservative":
            ax.text(
                row["agreement_human_majority"] + 0.01,
                i,
                "selected",
                fontsize=9,
                va="center",
                ha="left",
                color="#1f77b4"
            )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"], fontsize=10)
    ax.set_xlabel("Mean agreement", fontsize=11)
    ax.set_xlim(0, 1.02)
    ax.set_title("Comparative alignment of candidate output layers", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(OUT_DUMBBELL_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DUMBBELL_SVG, bbox_inches="tight")
    plt.close()

    # ========================================================
    # HEATMAP A
    # ========================================================
    print("Rendering benchmark heatmap...")
    heat_df = df_summary[[
        "display_name",
        "distance_human_majority",
        "distance_benchmark_models",
        "distance_pipeline"
    ]].copy()
    heat_df = heat_df.rename(columns={
        "distance_human_majority": "Human majority",
        "distance_benchmark_models": "Benchmark LLM coders",
        "distance_pipeline": "Pre-LLM pipeline draft"
    }).set_index("display_name")

    plt.figure(figsize=(8.8, 5.8))
    sns.heatmap(
        heat_df,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Distance (1 - agreement)"}
    )
    plt.title("Distance of output layers from benchmark references", fontsize=12)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUT_BENCH_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_BENCH_HEATMAP_SVG, bbox_inches="tight")
    plt.close()

    # ========================================================
    # HEATMAP B
    # ========================================================
    print("Building full distance matrix...")
    matrix_ids = OUTPUT_LAYERS + ["human_majority"] + HUMAN_CODERS + BENCHMARK_MODELS
    matrix_tables = {"human_majority": human_majority}

    for cid in OUTPUT_LAYERS + HUMAN_CODERS + BENCHMARK_MODELS:
        matrix_tables[cid] = prepare_source_table(df, cid)

    matrix = pd.DataFrame(index=matrix_ids, columns=matrix_ids, dtype=float)

    total_cells = len(matrix_ids) * len(matrix_ids)
    with tqdm(total=total_cells, desc="Distance matrix", ncols=90) as pbar:
        for row_id in matrix_ids:
            for col_id in matrix_ids:
                df_a = matrix_tables.get(row_id, pd.DataFrame())
                df_b = matrix_tables.get(col_id, pd.DataFrame())

                if df_a.empty or df_b.empty:
                    matrix.loc[row_id, col_id] = np.nan
                    pbar.update(1)
                    continue

                a, d, n, w = compute_mean_agreement(df_a, df_b)
                matrix.loc[row_id, col_id] = d
                pbar.update(1)

    matrix_display = matrix.copy()
    matrix_display.index = [DISPLAY_NAMES.get(i, i) for i in matrix_display.index]
    matrix_display.columns = [DISPLAY_NAMES.get(i, i) for i in matrix_display.columns]
    matrix_display.to_excel(OUT_FULL_MATRIX_XLSX)

    print("Rendering full distance heatmap...")
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        matrix_display,
        cmap="YlOrRd",
        annot=False,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Distance (1 - agreement)"}
    )
    plt.title("Full distance matrix of pilot outputs and benchmark coders", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_FULL_HEATMAP_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_FULL_HEATMAP_SVG, bbox_inches="tight")
    plt.close()

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================
    print("\n=== PILOT 3 OUTPUT LAYER SELECTION SUMMARY ===\n")
    print(df_summary[[
        "display_name",
        "agreement_human_majority",
        "agreement_benchmark_models",
        "agreement_pipeline",
        "agreement_critical_subset",
        "weighted_overall_score",
        "balance_score",
        "recommendation"
    ]].to_string(index=False))

    print(f"\nSaved:")
    print(f"- {OUT_SUMMARY_XLSX}")
    print(f"- {OUT_SUMMARY_CSV}")
    print(f"- {OUT_DUMBBELL_PNG}")
    print(f"- {OUT_DUMBBELL_SVG}")
    print(f"- {OUT_BENCH_HEATMAP_PNG}")
    print(f"- {OUT_BENCH_HEATMAP_SVG}")
    print(f"- {OUT_FULL_MATRIX_XLSX}")
    print(f"- {OUT_FULL_HEATMAP_PNG}")
    print(f"- {OUT_FULL_HEATMAP_SVG}")

if __name__ == "__main__":
    main()