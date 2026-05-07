import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REL_DIR = os.path.join(BASE_DIR, "REL")
REL_DATA_DIR = os.path.join(REL_DIR, "data")

INPUT_MASTER_LONG = os.path.join(REL_DATA_DIR, "pilot_comparison_master_long.xlsx")

OUT_SUMMARY_XLSX = os.path.join(REL_DATA_DIR, "output_layers_vs_benchmarks_summary.xlsx")
OUT_SUMMARY_CSV = os.path.join(REL_DATA_DIR, "output_layers_vs_benchmarks_summary.csv")

OUT_DUMBBELL_PNG = os.path.join(REL_DATA_DIR, "output_layers_dumbbell_human_vs_models.png")
OUT_DUMBBELL_SVG = os.path.join(REL_DATA_DIR, "output_layers_dumbbell_human_vs_models.svg")

OUT_BENCH_HEATMAP_PNG = os.path.join(REL_DATA_DIR, "output_layers_vs_benchmarks_heatmap.png")
OUT_BENCH_HEATMAP_SVG = os.path.join(REL_DATA_DIR, "output_layers_vs_benchmarks_heatmap.svg")

OUT_FULL_MATRIX_XLSX = os.path.join(REL_DATA_DIR, "full_distance_matrix.xlsx")
OUT_FULL_HEATMAP_PNG = os.path.join(REL_DATA_DIR, "full_distance_matrix_heatmap.png")
OUT_FULL_HEATMAP_SVG = os.path.join(REL_DATA_DIR, "full_distance_matrix_heatmap.svg")

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
    "gemini_working",
    "gemini_authoritative",
    "gemini_conservative",
    "local_working",
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

def compute_agreement_kappa(source_df, ref_df):
    agreements = []
    kappas = []
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

        a = merged[f"{var}_src"].tolist()
        b = merged[f"{var}_ref"].tolist()

        agree = np.mean(np.array(a) == np.array(b))
        agreements.append(float(agree))
        ns.append(len(a))

        labels = sorted(set(a) | set(b))
        if len(labels) < 2:
            kappas.append(np.nan)
        else:
            try:
                from sklearn.metrics import cohen_kappa_score
                kappas.append(float(cohen_kappa_score(a, b, labels=labels)))
            except Exception:
                kappas.append(np.nan)

    if not agreements:
        return np.nan, np.nan, np.nan, np.nan

    mean_agreement = float(np.mean(agreements))
    mean_distance = 1.0 - mean_agreement
    mean_kappa = float(np.nanmean(kappas)) if len(kappas) > 0 else np.nan
    mean_n = float(np.mean(ns)) if len(ns) > 0 else np.nan

    return mean_agreement, mean_distance, mean_kappa, mean_n

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

    # references
    human_majority = build_human_majority(df)
    refs = {"human_majority": human_majority}

    for cid in HUMAN_CODERS + BENCHMARK_MODELS + ["local_pipeline_draft"]:
        refs[cid] = prepare_source_table(df, cid)

    # ========================================================
    # SUMMARY TABLE
    # ========================================================
    summary_rows = []

    for out_id in OUTPUT_LAYERS:
        src = prepare_source_table(df, out_id)
        if src.empty:
            continue

        # against human majority
        a_h, d_h, k_h, n_h = compute_agreement_kappa(src, refs["human_majority"])

        # against benchmark models (mean across all 5)
        agreements_models = []
        distances_models = []
        kappas_models = []
        ns_models = []

        for mid in BENCHMARK_MODELS:
            ref_df = refs[mid]
            if ref_df.empty:
                continue
            a_m, d_m, k_m, n_m = compute_agreement_kappa(src, ref_df)
            agreements_models.append(a_m)
            distances_models.append(d_m)
            kappas_models.append(k_m)
            ns_models.append(n_m)

        summary_rows.append({
            "output_layer": out_id,
            "display_name": DISPLAY_NAMES.get(out_id, out_id),
            "mean_agreement_human_majority": np.nanmean([a_h]),
            "mean_distance_human_majority": np.nanmean([d_h]),
            "mean_kappa_human_majority": np.nanmean([k_h]),
            "mean_n_human_majority": np.nanmean([n_h]),
            "mean_agreement_benchmark_models": np.nanmean(agreements_models),
            "mean_distance_benchmark_models": np.nanmean(distances_models),
            "mean_kappa_benchmark_models": np.nanmean(kappas_models),
            "mean_n_benchmark_models": np.nanmean(ns_models),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_excel(OUT_SUMMARY_XLSX, index=False)
    df_summary.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    # ========================================================
    # DUMBBELL PLOT
    # ========================================================
    plot_df = df_summary.copy()
    plot_df = plot_df.sort_values("mean_agreement_human_majority", ascending=True).reset_index(drop=True)

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    def point_color(layer):
        if layer == "gemini_conservative":
            return "#1f77b4"
        elif "high_confidence" in layer:
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
            [row["mean_agreement_benchmark_models"], row["mean_agreement_human_majority"]],
            [i, i],
            color="#808080",
            linewidth=1.6,
            zorder=1
        )

    # benchmark models
    ax.scatter(
        plot_df["mean_agreement_benchmark_models"],
        y,
        s=95,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=3,
        label="Agreement with benchmark models"
    )

    # human majority
    colors = [point_color(c) for c in plot_df["output_layer"]]
    ax.scatter(
        plot_df["mean_agreement_human_majority"],
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
                row["mean_agreement_human_majority"] + 0.01,
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
    # HEATMAP A: OUTPUTS VS BENCHMARKS
    # ========================================================
    heat_rows = []
    columns = ["human_majority"] + BENCHMARK_MODELS

    for out_id in OUTPUT_LAYERS:
        src = prepare_source_table(df, out_id)
        if src.empty:
            continue

        row = {"output_layer": DISPLAY_NAMES.get(out_id, out_id)}
        for ref_id in columns:
            ref_df = refs[ref_id]
            a, d, k, n = compute_agreement_kappa(src, ref_df)
            row[DISPLAY_NAMES.get(ref_id, ref_id)] = d
        heat_rows.append(row)

    df_heat = pd.DataFrame(heat_rows).set_index("output_layer")
    plt.figure(figsize=(10.5, 6.4))
    sns.heatmap(
        df_heat,
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
    # HEATMAP B: FULL DISTANCE MATRIX
    # ========================================================
    matrix_ids = OUTPUT_LAYERS + ["human_majority"] + HUMAN_CODERS + BENCHMARK_MODELS
    matrix_tables = {"human_majority": human_majority}

    for cid in OUTPUT_LAYERS + HUMAN_CODERS + BENCHMARK_MODELS:
        matrix_tables[cid] = prepare_source_table(df, cid)

    mat = pd.DataFrame(index=matrix_ids, columns=matrix_ids, dtype=float)

    for row_id in matrix_ids:
        for col_id in matrix_ids:
            df_a = matrix_tables.get(row_id, pd.DataFrame())
            df_b = matrix_tables.get(col_id, pd.DataFrame())
            if df_a.empty or df_b.empty:
                mat.loc[row_id, col_id] = np.nan
                continue
            a, d, k, n = compute_agreement_kappa(df_a, df_b)
            mat.loc[row_id, col_id] = d

    # relabel
    mat_display = mat.copy()
    mat_display.index = [DISPLAY_NAMES.get(i, i) for i in mat_display.index]
    mat_display.columns = [DISPLAY_NAMES.get(i, i) for i in mat_display.columns]

    mat_display.to_excel(OUT_FULL_MATRIX_XLSX)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        mat_display,
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

    # console
    print("\n=== OUTPUT LAYERS VS BENCHMARKS SUMMARY ===")
    print(df_summary.to_string(index=False))

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