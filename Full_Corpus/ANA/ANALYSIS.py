from pathlib import Path
from datetime import datetime
import argparse
import math
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from scipy.stats import chi2_contingency
except Exception:
    chi2_contingency = None

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

PILOT_DIR = ROOT_DIR / "pilot"
PILOT_DATA_DIR = PILOT_DIR / "data"
CORPUS_DIR = ROOT_DIR / "CORPUS"
CORPUS_DATA_DIR = CORPUS_DIR / "data"

OUTPUT_DIR = SCRIPT_DIR / "output"

TIMELINE_DIR = OUTPUT_DIR / "timeline"
CH52_DIR = OUTPUT_DIR / "chapter_5_2"
CH53_DIR = OUTPUT_DIR / "chapter_5_3"
DIAG_DIR = OUTPUT_DIR / "diagnostics"
CDA_DIR = OUTPUT_DIR / "cda_sample"

INPUT_REVIEW_MASTER = CORPUS_DATA_DIR / "review_master.csv"
INPUT_POST_CONSOLIDATED = PILOT_DATA_DIR / "postConsolidated.csv"
INPUT_STEPB = PILOT_DATA_DIR / "postStepB.csv"

START_YEAR = 2021
START_MONTH = 7
END_YEAR = 2025
END_MONTH = 12

OUTLET_MIN_R2PLUS = 50

RELEVANCE_LABELS = {
    1: "Not relevant",
    2: "Marginal mention only",
    3: "Substantively relevant",
    4: "Main topic",
}

ACTOR_MENTION_LABELS = {
    1: "Wagner Group",
    2: "Africa Corps",
    3: "Both explicitly",
    4: "Indirect Russian contractors/forces",
    5: "Cannot determine",
}

DOMINANT_LABEL_LABELS = {
    1: "Mercenaries",
    2: "Instructors/advisers",
    3: "Allies/partners",
    4: "Foreign/occupying forces",
    5: "Neutral designation",
    6: "Multiple/no clear dominance",
}

STANCE_LABELS = {
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Mixed/ambivalent",
    5: "Cannot determine",
}

LEGITIMATION_LABELS = {
    1: "Delegitimized",
    2: "Normalized / implicitly legitimized",
    3: "Explicitly legitimized",
    4: "Cannot be determined",
}

DOMINANT_LOCATION_LABELS = {
    1: "Mali",
    2: "Other African countries",
    3: "Ukraine",
    4: "Other location",
    5: "Mali and other location",
}

MAIN_ASSOCIATED_ACTOR_LABELS = {
    1: "Malian army / junta",
    2: "Russia / Russian state",
    3: "France",
    4: "UN / MINUSMA",
    5: "ECOWAS / regional actors",
    6: "Local civilians",
    7: "Jihadist / terrorist groups",
    8: "Western states broadly",
    9: "No clear dominant actor",
    10: "Other",
}

DISCOURSE_LABELS = {
    1: "Sovereignty and emancipation",
    2: "Security and stabilization",
    3: "Violence and abuse",
    4: "Geopolitical competition",
    5: "Technocratic / factual reporting",
    6: "Mixed / no clear dominance",
}

FRAME_COLS = [
    "Counterterrorism",
    "Sovereignty",
    "Human_Rights_Abuse",
    "Anti_or_Neocolonialism",
    "Western_Failure",
    "Security_Effectiveness",
    "Economic_Interests",
    "Geopolitical_Rivalry",
]

FRAME_SHORT_MAP = {
    "Counterterrorism": "CT",
    "Sovereignty": "SOV",
    "Human_Rights_Abuse": "HRA",
    "Anti_or_Neocolonialism": "ANEO",
    "Western_Failure": "WF",
    "Security_Effectiveness": "SE",
    "Economic_Interests": "EI",
    "Geopolitical_Rivalry": "GR",
}

PERIOD_ORDER = [
    "P1_Early_Wagner",
    "P2_Mutiny_Transition",
    "P3_AfricaCorps_Phase",
]

ACTOR_SUBSET_ORDER = [
    "Wagner_only",
    "AC_or_Both",
    "Other",
]

RELEVANCE_GROUP_ORDER = [
    "R2",
    "R3",
    "R4",
]

FINAL_FALLBACKS = {
    "Final_Relevance": ["Final_Relevance", "Adj_V04_Relevance", "Relevance"],
    "Final_Actor_Mention": ["Final_Actor_Mention", "Adj_V05_Actor_Mention", "Actor_Mention"],
    "Final_Successor_Frame": ["Final_Successor_Frame", "Adj_V06_Successor_Frame", "Successor_Frame"],
    "Final_Dominant_Label": ["Final_Dominant_Label", "Adj_V07_Dominant_Label", "Dominant_Label"],
    "Final_Stance": ["Final_Stance", "Adj_V08_Stance", "Stance_Support"],
    "Final_Dominant_Location": ["Final_Dominant_Location", "Adj_V09_Dominant_Location", "Dominant_Location"],
    "Final_Ambivalence": ["Final_Ambivalence", "Adj_V10_Ambivalence", "Ambivalence_Support"],
    "Final_Legitimation": ["Final_Legitimation", "Adj_V11_Legitimation", "Legitimation_Support"],
    "Final_Main_Associated_Actor": [
        "Final_Main_Associated_Actor",
        "Adj_V20_Main_Associated_Actor",
        "Main_Associated_Actor",
    ],
    "Final_Dominant_Discourse": [
        "Final_Dominant_Discourse",
        "Adj_V21_Dominant_Discourse",
        "Dominant_Discourse_Support",
    ],
    "Final_Counterterrorism": ["Final_Counterterrorism", "Adj_V12_Counterterrorism", "Counterterrorism"],
    "Final_Sovereignty": ["Final_Sovereignty", "Adj_V13_Sovereignty", "Sovereignty"],
    "Final_Human_Rights_Abuse": ["Final_Human_Rights_Abuse", "Adj_V14_Human_Rights_Abuse", "Human_Rights_Abuse"],
    "Final_Anti_or_Neocolonialism": [
        "Final_Anti_or_Neocolonialism",
        "Adj_V15_Anti_or_Neocolonialism",
        "Anti_or_Neocolonialism",
    ],
    "Final_Western_Failure": ["Final_Western_Failure", "Adj_V16_Western_Failure", "Western_Failure"],
    "Final_Security_Effectiveness": [
        "Final_Security_Effectiveness",
        "Adj_V17_Security_Effectiveness",
        "Security_Effectiveness",
    ],
    "Final_Economic_Interests": [
        "Final_Economic_Interests",
        "Adj_V18_Economic_Interests",
        "Economic_Interests",
    ],
    "Final_Geopolitical_Rivalry": [
        "Final_Geopolitical_Rivalry",
        "Adj_V19_Geopolitical_Rivalry",
        "Geopolitical_Rivalry",
    ],
}

EXPECTED_CODESETS = {
    "Final_Relevance": {1, 2, 3, 4},
    "Final_Actor_Mention": {1, 2, 3, 4, 5},
    "Final_Successor_Frame": {0, 1},
    "Final_Dominant_Label": {1, 2, 3, 4, 5, 6},
    "Final_Stance": {1, 2, 3, 4, 5},
    "Final_Dominant_Location": {1, 2, 3, 4, 5},
    "Final_Ambivalence": {0, 1},
    "Final_Legitimation": {1, 2, 3, 4},
    "Final_Main_Associated_Actor": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    "Final_Dominant_Discourse": {1, 2, 3, 4, 5, 6},
    "Final_Counterterrorism": {0, 1},
    "Final_Sovereignty": {0, 1},
    "Final_Human_Rights_Abuse": {0, 1},
    "Final_Anti_or_Neocolonialism": {0, 1},
    "Final_Western_Failure": {0, 1},
    "Final_Security_Effectiveness": {0, 1},
    "Final_Economic_Interests": {0, 1},
    "Final_Geopolitical_Rivalry": {0, 1},
}

CDA_SELECTED_ARTICLES = {
    "030073": "Wagner expands influence in Africa",
    "030194": "Cost of Wagner operation",
    "050036": "15 European countries and Canada condemn deployment",
    "070003": "French minister warning African governments",
    "040079": "Bankass Human Rights Watch",
    "020264": "Le Mali face aux contradictions françaises",
    "110010": "Hollande donne des leçons",
    "080002": "Wagner mission continue",
    "070259": "Kremlin strengthens ties after Wagner departure",
    "080027": "Africom vs Africa Corps",
}


# ============================================================
# HELPERS
# ============================================================

def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def safe_int(x, default=np.nan):
    if pd.isna(x) or x is None:
        return default
    try:
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def pct(n, d):
    return (n / d * 100) if d else 0.0


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs():
    for base in [TIMELINE_DIR, CH52_DIR, CH53_DIR, DIAG_DIR, CDA_DIR]:
        for sub in ["tables", "figures", "summary"]:
            ensure_dir(base / sub)


def save_csv(df, path):
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path, text):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_article_id(x):
    raw = safe_str(x)
    if raw == "":
        return None
    if re.fullmatch(r"\d+\.0", raw):
        raw = raw[:-2]
    digits = re.sub(r"\D", "", raw)
    if digits:
        return digits.zfill(6)
    return raw


def find_first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_column(df, col, fill_value=np.nan):
    if col not in df.columns:
        df[col] = fill_value
    return df


def relabel_columns(cols, label_map):
    out = []
    for c in cols:
        try:
            key = int(float(c))
            out.append(label_map.get(key, safe_str(c)))
        except Exception:
            out.append(label_map.get(c, safe_str(c)))
    return out


def get_plot_palette(n):
    if sns is not None:
        if n <= 10:
            return sns.color_palette("tab10", n)
        if n <= 20:
            return sns.color_palette("tab20", n)
        return sns.color_palette("husl", n)
    return plt.cm.tab20(np.linspace(0, 1, max(n, 1)))


def save_stacked_percent_plot(
    df,
    group_col,
    category_col,
    title,
    out_path,
    group_order=None,
    category_order=None,
    label_map=None,
):
    if df.empty or group_col not in df.columns or category_col not in df.columns:
        return None

    ct = pd.crosstab(df[group_col], df[category_col], dropna=False)

    if group_order is not None:
        ct = ct.reindex(group_order, fill_value=0)

    if category_order is not None:
        ordered = [c for c in category_order if c in ct.columns]
        rest = [c for c in ct.columns if c not in ordered]
        ct = ct[ordered + rest]

    rowperc = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
    rowperc = rowperc.fillna(0)

    if label_map:
        rowperc.columns = relabel_columns(rowperc.columns, label_map)

    colors = get_plot_palette(len(rowperc.columns))

    plt.figure(figsize=(11, 6))
    rowperc.plot(kind="bar", stacked=True, ax=plt.gca(), color=colors, edgecolor="white", linewidth=0.3)
    plt.title(title)
    plt.ylabel("Percent within group")
    plt.xlabel(group_col)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=category_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return out_path


def save_heatmap(df, index_col, value_cols, title, out_path):
    if df.empty or index_col not in df.columns:
        return None

    keep = [c for c in value_cols if c in df.columns]
    if not keep:
        return None

    temp = df.set_index(index_col)[keep].copy().fillna(0)

    plt.figure(figsize=(12, max(4.5, 0.55 * len(temp) + 2)))
    if sns is not None:
        ax = sns.heatmap(temp, annot=True, fmt=".1f", cmap="Blues", linewidths=0.4, linecolor="white")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    else:
        plt.imshow(temp.values, aspect="auto")
        plt.colorbar()
        plt.yticks(range(len(temp.index)), temp.index)
        plt.xticks(range(len(temp.columns)), temp.columns, rotation=45, ha="right")

    plt.title(title)
    plt.ylabel(index_col)
    plt.xlabel("Variable")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return out_path


# ============================================================
# DATA LOADING AND HARMONISATION
# ============================================================

def find_adjudicated_input():
    candidates = list(PILOT_DIR.rglob("final_conservative_adjudicated_table.csv"))
    return candidates[0] if candidates else None


def load_csv_optional(path, dtype=None):
    if path and Path(path).exists():
        return pd.read_csv(path, dtype=dtype, low_memory=False)
    return None


def load_master():
    if INPUT_REVIEW_MASTER.exists():
        df = pd.read_csv(INPUT_REVIEW_MASTER, dtype={"Article_ID": str}, low_memory=False)
        source = "review_master"
    else:
        post_con = load_csv_optional(INPUT_POST_CONSOLIDATED, dtype={"Article_ID": str})
        adjudicated_path = find_adjudicated_input()
        adjudicated = load_csv_optional(adjudicated_path, dtype={"Article_ID": str})
        stepb = load_csv_optional(INPUT_STEPB, dtype={"article_id": str})

        if post_con is None or adjudicated is None:
            raise FileNotFoundError(
                "Missing input. Expected CORPUS/data/review_master.csv or pilot/data/postConsolidated.csv "
                "with an adjudicated coding table."
            )

        post_con["Article_ID"] = post_con["Article_ID"].apply(normalize_article_id)
        adjudicated["Article_ID"] = adjudicated["Article_ID"].apply(normalize_article_id)

        df = post_con.merge(adjudicated, on="Article_ID", how="left", suffixes=("", "_coded"))

        if stepb is not None and "article_id" in stepb.columns:
            stepb["article_id"] = stepb["article_id"].apply(normalize_article_id)
            keep = [
                "article_id",
                "source_attributed_flag",
                "likely_republished_flag",
                "near_duplicate_flag",
                "near_duplicate_cross_outlet_flag",
                "republication_index",
                "republication_confidence",
                "likely_republished_basis",
            ]
            keep = [c for c in keep if c in stepb.columns]
            stepb_sub = stepb[keep].rename(columns={"article_id": "Article_ID"})
            df = df.merge(stepb_sub, on="Article_ID", how="left")

        source = "assembled_master"

    if "Article_ID" in df.columns:
        df["Article_ID"] = df["Article_ID"].apply(normalize_article_id)

    return df, source


def harmonise_final_variables(df):
    df = df.copy()

    for final_col, candidates in FINAL_FALLBACKS.items():
        df = ensure_column(df, final_col, np.nan)
        for c in candidates:
            if c in df.columns:
                df[final_col] = df[final_col].fillna(df[c])
        df[final_col] = pd.to_numeric(df[final_col], errors="coerce")

    for c in ["Article_ID", "Outlet", "Date", "Headline"]:
        df = ensure_column(df, c, "")

    for c in ["source_attributed_flag", "likely_republished_flag", "near_duplicate_flag"]:
        df = ensure_column(df, c, 0)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df


def assign_period_group(date_value):
    dt = pd.to_datetime(date_value, errors="coerce")
    if pd.isna(dt):
        return "Unknown"

    y = dt.year
    m = dt.month

    if (y > 2021 or (y == 2021 and m >= 7)) and (y < 2023 or (y == 2023 and m <= 5)):
        return "P1_Early_Wagner"
    if y == 2023 and 6 <= m <= 12:
        return "P2_Mutiny_Transition"
    if 2024 <= y <= 2025:
        return "P3_AfricaCorps_Phase"

    return "Outside_Main_Period"


def actor_subset_compact(code):
    code = safe_int(code, default=-1)
    if code == 1:
        return "Wagner_only"
    if code in [2, 3]:
        return "AC_or_Both"
    return "Other"


def relevance_group(v):
    v = safe_int(v, default=-1)
    if v in [1, 2, 3, 4]:
        return f"R{v}"
    return "Unknown"


def dominant_label_group(v):
    v = safe_int(v, default=-1)
    return {
        1: "Mercenaries",
        2: "Instructors",
        3: "Partners",
        4: "Foreign_or_Occupying",
        5: "Neutral",
        6: "Multiple",
    }.get(v, "Other")


def dominant_location_group(v):
    v = safe_int(v, default=-1)
    return {
        1: "Mali",
        2: "Other_Africa",
        3: "Ukraine",
        4: "Other",
        5: "Mali_plus_Other",
    }.get(v, "Other")


def frame_bundle_label(row):
    active = []
    for f in FRAME_COLS:
        if safe_int(row.get(f"Final_{f}"), 0) == 1:
            active.append(FRAME_SHORT_MAP[f])

    if not active:
        return "no_frame"
    if len(active) == 1:
        return f"{active[0]}_only"
    if len(active) == 2:
        return "_".join(sorted(active))
    return "mixed_multi_frame"


def derive_variables(df):
    df = df.copy()

    df["Period_Group"] = df["Date"].apply(assign_period_group)
    df["Actor_Subset_Compact"] = df["Final_Actor_Mention"].apply(actor_subset_compact)
    df["Relevance_Group"] = df["Final_Relevance"].apply(relevance_group)
    df["Source_Group"] = np.where(df["source_attributed_flag"] == 1, "source_attributed", "non_attributed")
    df["Republish_Group"] = np.where(df["likely_republished_flag"] == 1, "likely_republished", "not_likely_republished")
    df["Dominant_Label_Group"] = df["Final_Dominant_Label"].apply(dominant_label_group)
    df["Dominant_Location_Group"] = df["Final_Dominant_Location"].apply(dominant_location_group)

    for f in FRAME_COLS:
        col = f"Final_{f}"
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    frame_cols = [f"Final_{f}" for f in FRAME_COLS]
    df["Frame_Sum"] = df[frame_cols].sum(axis=1)
    df["Frame_Bundle_Label"] = df.apply(frame_bundle_label, axis=1)

    df["DV_Relevance4"] = (df["Final_Relevance"] == 4).astype(int)
    df["DV_Successor"] = (df["Final_Successor_Frame"] == 1).astype(int)
    df["DV_HRA"] = (df["Final_Human_Rights_Abuse"] == 1).astype(int)
    df["DV_SE"] = (df["Final_Security_Effectiveness"] == 1).astype(int)
    df["DV_GR"] = (df["Final_Geopolitical_Rivalry"] == 1).astype(int)
    df["DV_SourceAttributed"] = (df["source_attributed_flag"] == 1).astype(int)
    df["DV_NegativeStance"] = (df["Final_Stance"] == 1).astype(int)
    df["DV_PositiveStance"] = (df["Final_Stance"] == 3).astype(int)
    df["DV_Delegitimized"] = (df["Final_Legitimation"] == 1).astype(int)

    df["R2plus_Flag"] = df["Final_Relevance"].isin([2, 3, 4]).astype(int)

    outlet_counts = (
        df[df["R2plus_Flag"] == 1]
        .groupby("Outlet", dropna=False)
        .size()
        .to_dict()
    )
    df["Outlet_R2plus_Count"] = df["Outlet"].map(outlet_counts).fillna(0).astype(int)
    df["Outlet_Eligible_50plus"] = (df["Outlet_R2plus_Count"] >= OUTLET_MIN_R2PLUS).astype(int)

    return df


def apply_relevance_scope(df, mode):
    mode = mode.lower().replace("+", "plus")
    if mode == "2plus":
        return df[df["Final_Relevance"].isin([2, 3, 4])].copy()
    if mode == "3plus":
        return df[df["Final_Relevance"].isin([3, 4])].copy()
    if mode in ["4", "4only"]:
        return df[df["Final_Relevance"] == 4].copy()
    raise ValueError("Unsupported relevance scope. Use 2plus, 3plus or 4only.")


# ============================================================
# BASIC TABLES
# ============================================================

def frequency_table(df, col, label_map=None):
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=["code", "label", "count", "percent"])

    out = df[col].value_counts(dropna=False).reset_index()
    out.columns = ["code", "count"]
    out["percent"] = out["count"] / len(df) * 100 if len(df) else 0

    def label_value(v):
        try:
            iv = int(float(v))
            return label_map.get(iv, safe_str(v)) if label_map else safe_str(v)
        except Exception:
            return safe_str(v)

    out["label"] = out["code"].apply(label_value)
    return out[["code", "label", "count", "percent"]]


def save_crosstab(df, row_col, col_col, out_prefix, label_map=None):
    if df.empty or row_col not in df.columns or col_col not in df.columns:
        counts = pd.DataFrame()
        rowperc = pd.DataFrame()
    else:
        counts = pd.crosstab(df[row_col], df[col_col], dropna=False)
        rowperc = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100

    save_csv(counts.reset_index(), out_prefix.with_name(out_prefix.name + "_counts.csv"))
    save_csv(rowperc.reset_index(), out_prefix.with_name(out_prefix.name + "_rowperc.csv"))

    if label_map and not counts.empty:
        counts_l = counts.copy()
        rowperc_l = rowperc.copy()
        counts_l.columns = relabel_columns(counts_l.columns, label_map)
        rowperc_l.columns = relabel_columns(rowperc_l.columns, label_map)
        save_csv(counts_l.reset_index(), out_prefix.with_name(out_prefix.name + "_counts_labelled.csv"))
        save_csv(rowperc_l.reset_index(), out_prefix.with_name(out_prefix.name + "_rowperc_labelled.csv"))

    return counts, rowperc


# ============================================================
# TIMELINE
# ============================================================

def build_month_start(row):
    y = safe_int(row.get("Date_Year"), None)
    m = safe_int(row.get("Date_Month"), None)

    if y is not None and m is not None:
        try:
            return pd.Timestamp(year=y, month=m, day=1)
        except Exception:
            pass

    dt = pd.to_datetime(row.get("Date"), errors="coerce")
    if pd.isna(dt):
        return pd.NaT

    return pd.Timestamp(year=dt.year, month=dt.month, day=1)


def export_timeline(df, relevance_mode):
    base = TIMELINE_DIR
    tables = base / "tables"
    figures = base / "figures"
    summary = base / "summary"

    temp = df.copy()
    temp["Month_Start"] = temp.apply(build_month_start, axis=1)

    start = pd.Timestamp(year=START_YEAR, month=START_MONTH, day=1)
    end = pd.Timestamp(year=END_YEAR, month=END_MONTH, day=1)

    temp = temp[(temp["Month_Start"] >= start) & (temp["Month_Start"] <= end)].copy()
    temp = apply_relevance_scope(temp, relevance_mode)

    monthly = (
        temp.groupby(["Month_Start", "Final_Relevance"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    pivot = (
        monthly.pivot(index="Month_Start", columns="Final_Relevance", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
        .sort_values("Month_Start")
    )

    for rel in [2, 3, 4]:
        if rel not in pivot.columns:
            pivot[rel] = 0

    pivot = pivot[["Month_Start", 2, 3, 4]].copy()

    export = pivot.rename(columns={2: "Relevance_2", 3: "Relevance_3", 4: "Relevance_4"})
    export["Month_Label"] = export["Month_Start"].dt.strftime("%Y-%m")
    export["Total"] = export["Relevance_2"] + export["Relevance_3"] + export["Relevance_4"]
    export["Relevance_Scope"] = relevance_mode
    save_csv(export, tables / f"monthly_relevance_counts_{relevance_mode}.csv")

    plt.figure(figsize=(16, 7))
    bottom = np.zeros(len(pivot))
    colors = {2: "#9ecae1", 3: "#3182bd", 4: "#08519c"}

    for rel in [2, 3, 4]:
        vals = pivot[rel].values
        plt.bar(
            pivot["Month_Start"],
            vals,
            bottom=bottom,
            width=25,
            color=colors[rel],
            label=RELEVANCE_LABELS[rel],
        )
        bottom += vals

    plt.title(f"Monthly distribution of articles by relevance ({relevance_mode})")
    plt.xlabel("Month")
    plt.ylabel("Number of articles")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Relevance")
    plt.tight_layout()
    ensure_dir(figures)
    plt.savefig(figures / f"monthly_relevance_distribution_{relevance_mode}.png", dpi=220)
    plt.close()

    actor = temp.copy()
    actor["Wagner_only"] = (actor["Final_Actor_Mention"] == 1).astype(int)
    actor["AC_or_Both"] = actor["Final_Actor_Mention"].isin([2, 3]).astype(int)

    actor_monthly = (
        actor.groupby("Month_Start", dropna=False)
        .agg(Wagner_only=("Wagner_only", "sum"), AC_or_Both=("AC_or_Both", "sum"))
        .reset_index()
        .sort_values("Month_Start")
    )
    actor_monthly["Month_Label"] = actor_monthly["Month_Start"].dt.strftime("%Y-%m")
    actor_monthly["Relevance_Scope"] = relevance_mode
    save_csv(actor_monthly, tables / f"monthly_wagner_vs_ac_counts_{relevance_mode}.csv")

    plt.figure(figsize=(16, 7))
    plt.plot(actor_monthly["Month_Start"], actor_monthly["Wagner_only"], color="#111111", linewidth=2, label="Wagner-only")
    plt.plot(actor_monthly["Month_Start"], actor_monthly["AC_or_Both"], color="#d62728", linewidth=2, label="Africa Corps / both")
    plt.title(f"Monthly distribution of Wagner-only versus Africa Corps / both cases ({relevance_mode})")
    plt.xlabel("Month")
    plt.ylabel("Number of articles")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / f"monthly_wagner_vs_ac_lines_{relevance_mode}.png", dpi=220)
    plt.close()

    text = f"""
TIME-RELEVANCE SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Time window:
- {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}

Relevance scope:
- {relevance_mode}

Coverage after filtering:
- rows retained: {len(temp)}
- earliest included month: {export["Month_Label"].min() if not export.empty else "N/A"}
- latest included month: {export["Month_Label"].max() if not export.empty else "N/A"}

Relevance totals:
- Relevance 2: {int((temp["Final_Relevance"] == 2).sum())}
- Relevance 3: {int((temp["Final_Relevance"] == 3).sum())}
- Relevance 4: {int((temp["Final_Relevance"] == 4).sum())}

Actor mention totals:
- Wagner-only: {int((temp["Final_Actor_Mention"] == 1).sum())}
- Africa Corps / both: {int(temp["Final_Actor_Mention"].isin([2, 3]).sum())}

Outputs:
- tables/monthly_relevance_counts_{relevance_mode}.csv
- tables/monthly_wagner_vs_ac_counts_{relevance_mode}.csv
- figures/monthly_relevance_distribution_{relevance_mode}.png
- figures/monthly_wagner_vs_ac_lines_{relevance_mode}.png
""".strip()

    write_text(summary / f"time_relevance_summary_{relevance_mode}.txt", text)


# ============================================================
# CHAPTER 5.2
# ============================================================

def export_chapter52(df):
    tables = CH52_DIR / "tables"
    summary = CH52_DIR / "summary"

    total_n = len(df)

    basics = pd.DataFrame([
        {"metric": "retrieved_articles", "value": total_n},
        {"metric": "relevance_1", "value": int((df["Final_Relevance"] == 1).sum())},
        {"metric": "relevance_2", "value": int((df["Final_Relevance"] == 2).sum())},
        {"metric": "relevance_3", "value": int((df["Final_Relevance"] == 3).sum())},
        {"metric": "relevance_4", "value": int((df["Final_Relevance"] == 4).sum())},
        {"metric": "relevance_2plus", "value": int(df["Final_Relevance"].isin([2, 3, 4]).sum())},
        {"metric": "relevance_3plus", "value": int(df["Final_Relevance"].isin([3, 4]).sum())},
        {"metric": "wagner_only", "value": int((df["Final_Actor_Mention"] == 1).sum())},
        {"metric": "africa_corps_or_both", "value": int(df["Final_Actor_Mention"].isin([2, 3]).sum())},
        {"metric": "source_attributed", "value": int((df["source_attributed_flag"] == 1).sum())},
        {"metric": "likely_republished", "value": int((df["likely_republished_flag"] == 1).sum())},
        {"metric": "near_duplicate", "value": int((df["near_duplicate_flag"] == 1).sum())},
        {"metric": "outlets_n", "value": df["Outlet"].nunique() if "Outlet" in df.columns else np.nan},
    ])
    save_csv(basics, tables / "chapter52_corpus_basics.csv")

    outlets = df["Outlet"].value_counts(dropna=False).reset_index()
    outlets.columns = ["Outlet", "count"]
    outlets["percent"] = outlets["count"] / total_n * 100 if total_n else 0
    save_csv(outlets.head(15), tables / "chapter52_outlet_overview.csv")

    save_csv(frequency_table(df, "Final_Relevance", RELEVANCE_LABELS), tables / "chapter52_relevance_overview.csv")
    save_csv(frequency_table(df, "Final_Actor_Mention", ACTOR_MENTION_LABELS), tables / "chapter52_actor_overview.csv")
    save_csv(frequency_table(df, "Final_Dominant_Label", DOMINANT_LABEL_LABELS), tables / "chapter52_label_overview.csv")
    save_csv(frequency_table(df, "Final_Dominant_Location", DOMINANT_LOCATION_LABELS), tables / "chapter52_location_overview.csv")

    temp = df.copy()
    temp["Date_dt"] = pd.to_datetime(temp["Date"], errors="coerce")
    temp["year"] = temp["Date_dt"].dt.year
    temporal = temp.groupby("year", dropna=False).size().reset_index(name="count")
    temporal["percent"] = temporal["count"] / total_n * 100 if total_n else 0
    save_csv(temporal.sort_values("year"), tables / "chapter52_temporal_overview.csv")

    source_rep = pd.DataFrame([
        {
            "metric": "source_attributed_flag",
            "count": int((df["source_attributed_flag"] == 1).sum()),
            "percent": pct(int((df["source_attributed_flag"] == 1).sum()), total_n),
        },
        {
            "metric": "likely_republished_flag",
            "count": int((df["likely_republished_flag"] == 1).sum()),
            "percent": pct(int((df["likely_republished_flag"] == 1).sum()), total_n),
        },
        {
            "metric": "near_duplicate_flag",
            "count": int((df["near_duplicate_flag"] == 1).sum()),
            "percent": pct(int((df["near_duplicate_flag"] == 1).sum()), total_n),
        },
    ])
    save_csv(source_rep, tables / "chapter52_source_republication_overview.csv")

    top_outlet = outlets.iloc[0] if not outlets.empty else None

    text = f"""
CHAPTER 5.2 OVERVIEW PACK
Generated: {datetime.now().isoformat(timespec="seconds")}

Core corpus composition:
- Retrieved corpus: {total_n}
- Relevance 1: {int((df["Final_Relevance"] == 1).sum())}
- Relevance 2: {int((df["Final_Relevance"] == 2).sum())}
- Relevance 3: {int((df["Final_Relevance"] == 3).sum())}
- Relevance 4: {int((df["Final_Relevance"] == 4).sum())}
- Relevance 2+: {int(df["Final_Relevance"].isin([2, 3, 4]).sum())}
- Relevance 3+: {int(df["Final_Relevance"].isin([3, 4]).sum())}

Actor overview:
- Wagner-only cases: {int((df["Final_Actor_Mention"] == 1).sum())}
- Africa Corps or both: {int(df["Final_Actor_Mention"].isin([2, 3]).sum())}

Outlet overview:
- Number of outlets: {df["Outlet"].nunique() if "Outlet" in df.columns else "N/A"}
- Largest outlet: {safe_str(top_outlet["Outlet"]) if top_outlet is not None else "N/A"} ({int(top_outlet["count"]) if top_outlet is not None else 0} articles)

Source / republication overview:
- Source-attributed cases: {int((df["source_attributed_flag"] == 1).sum())}
- Likely republished cases: {int((df["likely_republished_flag"] == 1).sum())}
- Near-duplicate cases: {int((df["near_duplicate_flag"] == 1).sum())}

Recommended tables:
- chapter52_corpus_basics.csv
- chapter52_outlet_overview.csv
- chapter52_relevance_overview.csv
- chapter52_source_republication_overview.csv
""".strip()

    write_text(summary / "chapter52_overview_pack.txt", text)


# ============================================================
# CHAPTER 5.3
# ============================================================

def filter_main_periods(df):
    return df[df["Period_Group"].isin(PERIOD_ORDER)].copy()


def export_frame_profiles(sub, group_col):
    rows = []
    for grp, gdf in sub.groupby(group_col, dropna=False):
        row = {group_col: grp, "n_articles": len(gdf)}
        for f in FRAME_COLS:
            row[f"{f}_percent"] = pct((gdf[f"Final_{f}"] == 1).sum(), len(gdf))
        rows.append(row)
    return pd.DataFrame(rows)


def export_chapter53(df):
    tables = CH53_DIR / "tables"
    figures = CH53_DIR / "figures"
    summary = CH53_DIR / "summary"

    save_csv(df, tables / "ch53_analysis_master.csv")

    sub = df[df["Final_Relevance"].isin([2, 3, 4])].copy()
    sub_period = filter_main_periods(sub)

    # 5.3.1 Actor representation
    save_csv(frequency_table(sub, "Final_Actor_Mention", ACTOR_MENTION_LABELS), tables / "ch531_actor_representation_overview.csv")
    save_crosstab(sub, "Actor_Subset_Compact", "Relevance_Group", tables / "ch531_actor_by_relevance")
    save_crosstab(sub_period, "Period_Group", "Actor_Subset_Compact", tables / "ch531_actor_by_period")
    save_crosstab(sub_period, "Period_Group", "Final_Successor_Frame", tables / "ch531_successor_by_period", {0: "No", 1: "Yes"})

    save_stacked_percent_plot(
        sub,
        "Actor_Subset_Compact",
        "Relevance_Group",
        "Relevance profile by actor subset",
        figures / "ch531_actor_by_relevance.png",
        group_order=ACTOR_SUBSET_ORDER,
        category_order=RELEVANCE_GROUP_ORDER,
    )
    save_stacked_percent_plot(
        sub_period,
        "Period_Group",
        "Actor_Subset_Compact",
        "Actor subset by period group",
        figures / "ch531_actor_by_period.png",
        group_order=PERIOD_ORDER,
        category_order=ACTOR_SUBSET_ORDER,
    )

    # 5.3.2 Stance and legitimation
    save_csv(frequency_table(sub, "Final_Stance", STANCE_LABELS), tables / "ch532_stance_overall.csv")
    save_csv(frequency_table(sub, "Final_Legitimation", LEGITIMATION_LABELS), tables / "ch532_legitimation_overall.csv")
    save_crosstab(sub, "Actor_Subset_Compact", "Final_Stance", tables / "ch532_stance_by_actor_subset", STANCE_LABELS)
    save_crosstab(sub_period, "Period_Group", "Final_Stance", tables / "ch532_stance_by_period", STANCE_LABELS)
    save_crosstab(sub, "Actor_Subset_Compact", "Final_Legitimation", tables / "ch532_legitimation_by_actor_subset", LEGITIMATION_LABELS)
    save_crosstab(sub, "Actor_Subset_Compact", "Final_Ambivalence", tables / "ch532_ambivalence_by_actor_subset", {0: "No", 1: "Yes"})

    save_stacked_percent_plot(
        sub,
        "Actor_Subset_Compact",
        "Final_Stance",
        "Stance by actor subset",
        figures / "ch532_stance_by_actor_subset.png",
        group_order=ACTOR_SUBSET_ORDER,
        category_order=[1, 2, 3, 4, 5],
        label_map=STANCE_LABELS,
    )
    save_stacked_percent_plot(
        sub,
        "Actor_Subset_Compact",
        "Final_Legitimation",
        "Legitimation by actor subset",
        figures / "ch532_legitimation_by_actor_subset.png",
        group_order=ACTOR_SUBSET_ORDER,
        category_order=[1, 2, 3, 4],
        label_map=LEGITIMATION_LABELS,
    )

    # 5.3.3 Frames and discourse
    frame_rows = []
    for f in FRAME_COLS:
        count = int((sub[f"Final_{f}"] == 1).sum())
        frame_rows.append({"frame": f, "count": count, "percent": pct(count, len(sub))})
    frames_overall = pd.DataFrame(frame_rows).sort_values("percent", ascending=False)
    save_csv(frames_overall, tables / "ch533_frames_overall.csv")

    actor_frame = export_frame_profiles(sub, "Actor_Subset_Compact")
    period_frame = export_frame_profiles(sub_period, "Period_Group")
    save_csv(actor_frame, tables / "ch533_frames_by_actor_subset.csv")
    save_csv(period_frame, tables / "ch533_frames_by_period.csv")
    save_csv(frequency_table(sub, "Frame_Bundle_Label"), tables / "ch533_frame_bundles.csv")
    save_crosstab(sub, "Actor_Subset_Compact", "Final_Dominant_Discourse", tables / "ch533_discourse_by_actor_subset", DISCOURSE_LABELS)
    save_crosstab(sub_period, "Period_Group", "Final_Dominant_Discourse", tables / "ch533_discourse_by_period", DISCOURSE_LABELS)

    plt.figure(figsize=(10, 6))
    plt.bar(frames_overall["frame"], frames_overall["percent"], color=get_plot_palette(len(frames_overall)))
    plt.title("Frame prevalence overall (Relevance 2+)")
    plt.ylabel("Percent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ensure_dir(figures)
    plt.savefig(figures / "ch533_frames_overall.png", dpi=220, bbox_inches="tight")
    plt.close()

    frame_percent_cols = [f"{f}_percent" for f in FRAME_COLS]
    save_heatmap(actor_frame, "Actor_Subset_Compact", frame_percent_cols, "Frame prevalence by actor subset", figures / "ch533_frames_by_actor_subset_heatmap.png")
    save_heatmap(period_frame, "Period_Group", frame_percent_cols, "Frame prevalence by period group", figures / "ch533_frames_by_period_heatmap.png")

    # 5.3.4 Associated actors
    save_csv(frequency_table(sub, "Final_Main_Associated_Actor", MAIN_ASSOCIATED_ACTOR_LABELS), tables / "ch534_associated_actor_overall.csv")
    save_crosstab(sub, "Actor_Subset_Compact", "Final_Main_Associated_Actor", tables / "ch534_associated_actor_by_actor_subset", MAIN_ASSOCIATED_ACTOR_LABELS)
    save_crosstab(sub_period, "Period_Group", "Final_Main_Associated_Actor", tables / "ch534_associated_actor_by_period", MAIN_ASSOCIATED_ACTOR_LABELS)
    save_crosstab(sub, "Dominant_Location_Group", "Final_Main_Associated_Actor", tables / "ch534_associated_actor_by_location", MAIN_ASSOCIATED_ACTOR_LABELS)

    save_stacked_percent_plot(
        sub,
        "Actor_Subset_Compact",
        "Final_Main_Associated_Actor",
        "Main associated actor by actor subset",
        figures / "ch534_associated_actor_by_actor_subset.png",
        group_order=ACTOR_SUBSET_ORDER,
        category_order=list(MAIN_ASSOCIATED_ACTOR_LABELS.keys()),
        label_map=MAIN_ASSOCIATED_ACTOR_LABELS,
    )

    # 5.3.5 Source, republication and outlet profile
    source_rows = []
    for grp, gdf in sub.groupby("Source_Group", dropna=False):
        source_rows.append({
            "Source_Group": grp,
            "n_articles": len(gdf),
            "relevance4_percent": pct((gdf["Final_Relevance"] == 4).sum(), len(gdf)),
            "HRA_percent": pct((gdf["Final_Human_Rights_Abuse"] == 1).sum(), len(gdf)),
            "SE_percent": pct((gdf["Final_Security_Effectiveness"] == 1).sum(), len(gdf)),
            "GR_percent": pct((gdf["Final_Geopolitical_Rivalry"] == 1).sum(), len(gdf)),
            "AC_or_Both_percent": pct((gdf["Actor_Subset_Compact"] == "AC_or_Both").sum(), len(gdf)),
        })
    source_profile = pd.DataFrame(source_rows)
    save_csv(source_profile, tables / "ch535_source_vs_nonattributed.csv")

    repub_rows = []
    for grp, gdf in sub.groupby("Republish_Group", dropna=False):
        repub_rows.append({
            "Republish_Group": grp,
            "n_articles": len(gdf),
            "relevance4_percent": pct((gdf["Final_Relevance"] == 4).sum(), len(gdf)),
            "HRA_percent": pct((gdf["Final_Human_Rights_Abuse"] == 1).sum(), len(gdf)),
            "SE_percent": pct((gdf["Final_Security_Effectiveness"] == 1).sum(), len(gdf)),
            "GR_percent": pct((gdf["Final_Geopolitical_Rivalry"] == 1).sum(), len(gdf)),
        })
    repub_profile = pd.DataFrame(repub_rows)
    save_csv(repub_profile, tables / "ch535_republished_vs_nonrepublished.csv")

    eligible = sub[sub["Outlet_Eligible_50plus"] == 1].copy()
    outlet_rows = []
    for outlet, gdf in eligible.groupby("Outlet", dropna=False):
        outlet_rows.append({
            "Outlet": outlet,
            "n_articles": len(gdf),
            "relevance4_percent": pct((gdf["Final_Relevance"] == 4).sum(), len(gdf)),
            "source_attributed_percent": pct((gdf["source_attributed_flag"] == 1).sum(), len(gdf)),
            "likely_republished_percent": pct((gdf["likely_republished_flag"] == 1).sum(), len(gdf)),
            "HRA_percent": pct((gdf["Final_Human_Rights_Abuse"] == 1).sum(), len(gdf)),
            "SE_percent": pct((gdf["Final_Security_Effectiveness"] == 1).sum(), len(gdf)),
            "GR_percent": pct((gdf["Final_Geopolitical_Rivalry"] == 1).sum(), len(gdf)),
            "AC_or_Both_percent": pct((gdf["Actor_Subset_Compact"] == "AC_or_Both").sum(), len(gdf)),
            "successor_percent": pct((gdf["Final_Successor_Frame"] == 1).sum(), len(gdf)),
        })
    outlet_profile = pd.DataFrame(outlet_rows).sort_values("n_articles", ascending=False)
    save_csv(outlet_profile, tables / "ch535_outlet_profile_eligible_only.csv")

    save_heatmap(
        source_profile,
        "Source_Group",
        ["relevance4_percent", "HRA_percent", "SE_percent", "GR_percent", "AC_or_Both_percent"],
        "Source-attributed vs non-attributed profile",
        figures / "ch535_source_vs_nonattributed_heatmap.png",
    )
    save_heatmap(
        repub_profile,
        "Republish_Group",
        ["relevance4_percent", "HRA_percent", "SE_percent", "GR_percent"],
        "Likely republished vs non-republished profile",
        figures / "ch535_republished_vs_nonrepublished_heatmap.png",
    )
    save_heatmap(
        outlet_profile,
        "Outlet",
        [
            "relevance4_percent",
            "source_attributed_percent",
            "likely_republished_percent",
            "HRA_percent",
            "SE_percent",
            "GR_percent",
            "AC_or_Both_percent",
            "successor_percent",
        ],
        "Eligible outlet profile",
        figures / "ch535_outlet_profile_heatmap.png",
    )

    export_chapter53_stats(sub, sub_period, tables)

    write_chapter53_pack(df, sub, source_profile, repub_profile, outlet_profile, summary)


def cramers_v(ct):
    if chi2_contingency is None or ct.empty:
        return np.nan
    try:
        chi2, p, dof, expected = chi2_contingency(ct)
        n = expected.sum()
        r, k = ct.shape
        denom = min(k - 1, r - 1)
        if n == 0 or denom <= 0:
            return np.nan
        return math.sqrt((chi2 / n) / denom)
    except Exception:
        return np.nan


def export_chapter53_stats(sub, sub_period, tables):
    tests = [
        ("ActorSubset_vs_Stance", sub, "Actor_Subset_Compact", "Final_Stance"),
        ("ActorSubset_vs_Successor", sub, "Actor_Subset_Compact", "Final_Successor_Frame"),
        ("ActorSubset_vs_HRA", sub, "Actor_Subset_Compact", "Final_Human_Rights_Abuse"),
        ("ActorSubset_vs_SE", sub, "Actor_Subset_Compact", "Final_Security_Effectiveness"),
        ("ActorSubset_vs_GR", sub, "Actor_Subset_Compact", "Final_Geopolitical_Rivalry"),
        ("Period_vs_ActorSubset", sub_period, "Period_Group", "Actor_Subset_Compact"),
        ("Period_vs_Successor", sub_period, "Period_Group", "Final_Successor_Frame"),
        ("Source_vs_HRA", sub, "Source_Group", "Final_Human_Rights_Abuse"),
        ("Source_vs_GR", sub, "Source_Group", "Final_Geopolitical_Rivalry"),
    ]

    rows = []
    if chi2_contingency is not None:
        for label, use_df, row_col, col_col in tests:
            if use_df.empty or row_col not in use_df.columns or col_col not in use_df.columns:
                continue
            ct = pd.crosstab(use_df[row_col], use_df[col_col], dropna=False)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            try:
                chi2, p, dof, expected = chi2_contingency(ct)
                rows.append({
                    "test_label": label,
                    "row_col": row_col,
                    "col_col": col_col,
                    "chi2": chi2,
                    "p_value": p,
                    "dof": dof,
                    "cramers_v": cramers_v(ct),
                    "n_total": int(ct.values.sum()),
                })
            except Exception:
                pass

    chi_df = pd.DataFrame(rows)
    save_csv(chi_df, tables / "ch53_chisquare_summary.csv")

    if smf is None:
        save_csv(pd.DataFrame([{"status": "regression_skipped", "reason": "statsmodels_not_available"}]), tables / "ch53_regression_status.csv")
        return

    specs = [
        ("relevance4_model", "DV_Relevance4", ["Actor_Subset_Compact", "Period_Group", "Source_Group", "Dominant_Location_Group"]),
        ("successor_model", "DV_Successor", ["Actor_Subset_Compact", "Period_Group", "Source_Group", "Dominant_Location_Group"]),
        ("hra_model", "DV_HRA", ["Actor_Subset_Compact", "Period_Group", "Source_Group", "Dominant_Location_Group"]),
        ("security_effectiveness_model", "DV_SE", ["Actor_Subset_Compact", "Period_Group", "Source_Group", "Dominant_Location_Group"]),
        ("geopolitics_model", "DV_GR", ["Actor_Subset_Compact", "Period_Group", "Source_Group", "Dominant_Location_Group"]),
        ("negative_stance_model", "DV_NegativeStance", ["Actor_Subset_Compact", "Period_Group", "Source_Group", "Dominant_Label_Group"]),
        ("delegitimized_model", "DV_Delegitimized", ["Actor_Subset_Compact", "Period_Group", "Source_Group", "Dominant_Label_Group"]),
    ]

    coef_rows = []
    fit_rows = []
    status_rows = []

    for name, dv, predictors in specs:
        use_cols = [dv] + predictors
        temp = sub[use_cols].dropna().copy()

        if temp.empty or temp[dv].nunique() < 2:
            status_rows.append({"model_name": name, "status": "skipped_no_variation_or_empty"})
            continue

        formula = f"{dv} ~ " + " + ".join([f"C({p})" for p in predictors])

        try:
            model = smf.logit(formula=formula, data=temp).fit(disp=False, maxiter=200)
            fit_type = "logit"
        except Exception:
            try:
                model = smf.logit(formula=formula, data=temp).fit_regularized(disp=False, maxiter=200)
                fit_type = "regularized_logit"
            except Exception as e:
                status_rows.append({"model_name": name, "status": f"failed:{type(e).__name__}"})
                continue

        params = model.params
        pvals = getattr(model, "pvalues", pd.Series(index=params.index, dtype=float))

        try:
            conf = model.conf_int()
        except Exception:
            conf = pd.DataFrame(index=params.index, data={0: np.nan, 1: np.nan})

        for term in params.index:
            if term == "Intercept":
                continue

            coef = params[term]
            lo = conf.loc[term, 0] if term in conf.index else np.nan
            hi = conf.loc[term, 1] if term in conf.index else np.nan

            def safe_exp(v):
                if pd.isna(v):
                    return np.nan
                if v > 700:
                    return np.inf
                if v < -700:
                    return 0.0
                return math.exp(v)

            or_val = safe_exp(coef)

            coef_rows.append({
                "model_name": name,
                "dv": dv,
                "term": term,
                "coef_logodds": coef,
                "odds_ratio": or_val,
                "ci_low_or": safe_exp(lo),
                "ci_high_or": safe_exp(hi),
                "p_value": pvals.get(term, np.nan),
                "n_model": len(temp),
                "fit_type": fit_type,
                "extreme_or_flag": int(or_val == np.inf or or_val > 1000 or or_val < 0.001) if pd.notna(or_val) else 0,
            })

        fit_rows.append({
            "model_name": name,
            "dv": dv,
            "n_model": len(temp),
            "pseudo_r2": getattr(model, "prsquared", np.nan),
            "aic": getattr(model, "aic", np.nan),
            "bic": getattr(model, "bic", np.nan),
            "fit_type": fit_type,
        })
        status_rows.append({"model_name": name, "status": fit_type})

    coef_df = pd.DataFrame(coef_rows)
    fit_df = pd.DataFrame(fit_rows)
    status_df = pd.DataFrame(status_rows)

    save_csv(coef_df, tables / "ch53_regression_coefficients.csv")
    save_csv(fit_df, tables / "ch53_regression_fit_summary.csv")
    save_csv(status_df, tables / "ch53_regression_status.csv")

    if not coef_df.empty:
        interesting = coef_df[coef_df["p_value"].fillna(1) < 0.10].sort_values("p_value")
    else:
        interesting = pd.DataFrame()

    save_csv(interesting, tables / "ch53_regression_interesting_results.csv")


def write_chapter53_pack(df, sub, source_profile, repub_profile, outlet_profile, summary):
    frame_stats = []
    for f in FRAME_COLS:
        count = int((sub[f"Final_{f}"] == 1).sum())
        frame_stats.append((f, count, pct(count, len(sub))))
    frame_stats = sorted(frame_stats, key=lambda x: x[1], reverse=True)
    frame_text = "\n".join([f"- {f}: {c} ({p:.1f}%)" for f, c, p in frame_stats])

    text = f"""
CHAPTER 5.3 WRITING PACK
Generated: {datetime.now().isoformat(timespec="seconds")}

Corpus basis:
- Retrieved corpus: {len(df)}
- Relevance 2+ corpus: {len(sub)}
- Relevance 3+ corpus: {len(df[df["Final_Relevance"].isin([3, 4])])}
- Relevance 4 corpus: {len(df[df["Final_Relevance"] == 4])}

Actor structure within relevance 2+:
- Wagner-only: {int((sub["Actor_Subset_Compact"] == "Wagner_only").sum())}
- Africa Corps / both: {int((sub["Actor_Subset_Compact"] == "AC_or_Both").sum())}
- Other / indirect / unclear: {int((sub["Actor_Subset_Compact"] == "Other").sum())}

Source and republication:
- Source-attributed: {int((sub["source_attributed_flag"] == 1).sum())} ({pct((sub["source_attributed_flag"] == 1).sum(), len(sub)):.1f}%)
- Likely republished: {int((sub["likely_republished_flag"] == 1).sum())} ({pct((sub["likely_republished_flag"] == 1).sum(), len(sub)):.1f}%)

Overall frame prevalence:
{frame_text}

Key output groups:
- 5.3.1 actor representation tables and figures
- 5.3.2 stance and legitimation tables and figures
- 5.3.3 frame and discourse tables and figures
- 5.3.4 associated actor tables and figures
- 5.3.5 source, republication and outlet-level tables and figures

Analytical cautions:
- Statistical outputs describe within-corpus associations.
- Outlet comparisons are restricted to outlets with at least {OUTLET_MIN_R2PLUS} relevance 2+ articles.
- Small categories should be interpreted cautiously.
- Extreme regression odds ratios should be treated as unstable.
""".strip()

    write_text(summary / "chapter53_writing_pack.txt", text)


# ============================================================
# DIAGNOSTICS
# ============================================================

def export_diagnostics(df, source_used):
    tables = DIAG_DIR / "tables"
    summary = DIAG_DIR / "summary"

    total = len(df)
    rows = []

    date_dt = pd.to_datetime(df["Date"], errors="coerce")
    missing_date = df["Date"].isna() | (df["Date"].astype(str).str.strip() == "")
    unparseable = (~missing_date) & date_dt.isna()

    rows.append({"problem_type": "missing_date", "affected_rows": int(missing_date.sum()), "percent": pct(missing_date.sum(), total)})
    rows.append({"problem_type": "unparseable_date", "affected_rows": int(unparseable.sum()), "percent": pct(unparseable.sum(), total)})
    rows.append({"problem_type": "period_unknown", "affected_rows": int((df["Period_Group"] == "Unknown").sum()), "percent": pct((df["Period_Group"] == "Unknown").sum(), total)})
    rows.append({"problem_type": "period_outside_main_period", "affected_rows": int((df["Period_Group"] == "Outside_Main_Period").sum()), "percent": pct((df["Period_Group"] == "Outside_Main_Period").sum(), total)})

    missing_outlet = df["Outlet"].isna() | (df["Outlet"].astype(str).str.strip() == "")
    rows.append({"problem_type": "missing_outlet", "affected_rows": int(missing_outlet.sum()), "percent": pct(missing_outlet.sum(), total)})

    invalid_rows = []
    missing_rows = []

    for col, allowed in EXPECTED_CODESETS.items():
        if col not in df.columns:
            continue

        missing = df[col].isna()
        invalid = (~df[col].isna()) & (~df[col].isin(allowed))

        rows.append({"problem_type": f"missing_{col}", "affected_rows": int(missing.sum()), "percent": pct(missing.sum(), total)})
        rows.append({"problem_type": f"invalid_{col}", "affected_rows": int(invalid.sum()), "percent": pct(invalid.sum(), total)})

        if invalid.any():
            tmp = df.loc[invalid, ["Article_ID", "Outlet", "Date", "Headline"]].copy()
            tmp["problem_column"] = col
            tmp["invalid_value"] = df.loc[invalid, col].values
            invalid_rows.append(tmp)

        if missing.any():
            tmp = df.loc[missing, ["Article_ID", "Outlet", "Date", "Headline"]].copy()
            tmp["problem_column"] = col
            missing_rows.append(tmp)

    summary_df = pd.DataFrame(rows)
    save_csv(summary_df, tables / "diagnostic_summary.csv")

    invalid_df = pd.concat(invalid_rows, ignore_index=True) if invalid_rows else pd.DataFrame()
    missing_df = pd.concat(missing_rows, ignore_index=True) if missing_rows else pd.DataFrame()
    save_csv(invalid_df, tables / "diagnostic_invalid_values.csv")
    save_csv(missing_df, tables / "diagnostic_missing_values.csv")

    period_issues = df[df["Period_Group"].isin(["Unknown", "Outside_Main_Period"])][
        ["Article_ID", "Outlet", "Date", "Period_Group", "Headline"]
    ].copy()
    save_csv(period_issues, tables / "diagnostic_period_issues.csv")

    outlet_counts = (
        df[df["R2plus_Flag"] == 1]
        .groupby("Outlet", dropna=False)
        .size()
        .reset_index(name="r2plus_count")
        .sort_values("r2plus_count", ascending=False)
    )
    outlet_counts["eligible_50plus"] = (outlet_counts["r2plus_count"] >= OUTLET_MIN_R2PLUS).astype(int)
    save_csv(outlet_counts, tables / "diagnostic_outlet_counts.csv")

    text = f"""
DATA DIAGNOSTIC REPORT
Generated: {datetime.now().isoformat(timespec="seconds")}

Input source:
- {source_used}

Corpus size:
- rows inspected: {total}

Date and period:
- Missing dates: {int(missing_date.sum())}
- Unparseable dates: {int(unparseable.sum())}
- Unknown period: {int((df["Period_Group"] == "Unknown").sum())}
- Outside main period: {int((df["Period_Group"] == "Outside_Main_Period").sum())}

Coding diagnostics:
- Rows with invalid coded values: {len(invalid_df["Article_ID"].unique()) if not invalid_df.empty else 0}
- Rows with missing coded values: {len(missing_df["Article_ID"].unique()) if not missing_df.empty else 0}

Generated files:
- diagnostic_summary.csv
- diagnostic_invalid_values.csv
- diagnostic_missing_values.csv
- diagnostic_period_issues.csv
- diagnostic_outlet_counts.csv
""".strip()

    write_text(summary / "diagnostic_report.txt", text)


# ============================================================
# CDA CLEAN SAMPLE EXPORT
# ============================================================

def safe_filename(value, max_len=100):
    value = safe_str(value) or "untitled"
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^\w\-.]+", "", value, flags=re.UNICODE)
    value = value.strip("._-")
    if not value:
        value = "untitled"
    return value[:max_len]


def find_text_column(df):
    candidates = [
        "Body_Postclean",
        "body_postclean",
        "Body",
        "body",
        "Full_Text",
        "full_text",
        "Text",
        "text",
        "Content",
        "content",
    ]
    return find_first_existing_column(df, candidates)


def clean_article_bundle(row, text_col):
    article_id = safe_str(row.get("Article_ID"))
    title = safe_str(row.get("Headline"))
    outlet = safe_str(row.get("Outlet"))
    date = safe_str(row.get("Date"))
    url = safe_str(row.get("URL")) if "URL" in row.index else safe_str(row.get("url"))
    body = safe_str(row.get(text_col)) if text_col else ""

    relevance = safe_int(row.get("Final_Relevance"), "")
    actor = safe_int(row.get("Final_Actor_Mention"), "")

    return f"""
ARTICLE ID: {article_id}
TITLE: {title}
OUTLET: {outlet}
DATE: {date}
URL: {url}

CODING SNAPSHOT
Relevance: {relevance}
Actor mention: {actor}
Source-attributed: {safe_int(row.get("source_attributed_flag"), 0)}
Likely republished: {safe_int(row.get("likely_republished_flag"), 0)}

ARTICLE TEXT
================================================================================

HEADLINE
{title}

BODY
{body if body else "[No body text available]"}
""".strip()


def export_cda_sample(df):
    tables = CDA_DIR / "tables"
    texts = CDA_DIR / "texts"
    summary = CDA_DIR / "summary"
    ensure_dir(texts)

    text_col = find_text_column(df)

    selected = df[df["Article_ID"].isin(CDA_SELECTED_ARTICLES.keys())].copy()

    rows = []
    all_parts = []

    for article_id, short_title in CDA_SELECTED_ARTICLES.items():
        match = selected[selected["Article_ID"] == article_id]

        if match.empty:
            rows.append({
                "Article_ID": article_id,
                "Found": False,
                "Short_Title": short_title,
                "Outlet": "",
                "Date": "",
                "Headline": "",
            })
            continue

        row = match.iloc[0]
        headline = safe_str(row.get("Headline")) or short_title
        base_name = f"{article_id}__{safe_filename(headline)}"

        bundle = clean_article_bundle(row, text_col)
        write_text(texts / f"{base_name}.txt", bundle)

        all_parts.append("\n" + "=" * 100 + "\n")
        all_parts.append(bundle)

        rows.append({
            "Article_ID": article_id,
            "Found": True,
            "Short_Title": short_title,
            "Outlet": safe_str(row.get("Outlet")),
            "Date": safe_str(row.get("Date")),
            "Headline": headline,
            "Relevance": safe_int(row.get("Final_Relevance"), ""),
            "Actor_Mention": safe_int(row.get("Final_Actor_Mention"), ""),
            "Source_Attributed": safe_int(row.get("source_attributed_flag"), 0),
            "Likely_Republished": safe_int(row.get("likely_republished_flag"), 0),
        })

    sample_table = pd.DataFrame(rows)
    save_csv(sample_table, tables / "cda_sample_table.csv")

    write_text(
        summary / "ALL_SELECTED_CDA_ARTICLES.txt",
        "SELECTED CDA ARTICLES\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Selected IDs: {len(CDA_SELECTED_ARTICLES)}\n"
        f"Found IDs: {int(sample_table['Found'].sum()) if not sample_table.empty else 0}\n\n"
        + "".join(all_parts),
    )

    write_text(
        summary / "cda_export_readme.txt",
        """
CDA SAMPLE EXPORT

Contents:
- tables/cda_sample_table.csv
- texts/*.txt
- summary/ALL_SELECTED_CDA_ARTICLES.txt

Each article text bundle contains metadata, coding snapshot and article text where available.
""".strip(),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified analysis pipeline.")
    parser.add_argument("--all", action="store_true", help="Run all exports.")
    parser.add_argument("--timeline", action="store_true", help="Export timeline outputs.")
    parser.add_argument("--chapter52", action="store_true", help="Export Chapter 5.2 outputs.")
    parser.add_argument("--chapter53", action="store_true", help="Export Chapter 5.3 outputs.")
    parser.add_argument("--diagnostics", action="store_true", help="Export diagnostics.")
    parser.add_argument("--cda", action="store_true", help="Export CDA sample text bundles.")
    parser.add_argument("--relevance", default="2plus", choices=["2plus", "3plus", "4only"], help="Relevance scope for timeline.")
    args = parser.parse_args()

    if not any([args.all, args.timeline, args.chapter52, args.chapter53, args.diagnostics, args.cda]):
        args.all = True

    ensure_output_dirs()

    if sns is not None:
        sns.set_theme(style="whitegrid")

    print("\nLoading and preparing master dataset...")
    df_raw, source_used = load_master()
    df = harmonise_final_variables(df_raw)
    df = derive_variables(df)

    save_csv(df, OUTPUT_DIR / "analysis_master_clean.csv")

    if args.all or args.timeline:
        print("Exporting timeline outputs...")
        export_timeline(df, args.relevance)

    if args.all or args.chapter52:
        print("Exporting Chapter 5.2 outputs...")
        export_chapter52(df)

    if args.all or args.chapter53:
        print("Exporting Chapter 5.3 outputs...")
        export_chapter53(df)

    if args.all or args.diagnostics:
        print("Exporting diagnostics...")
        export_diagnostics(df, source_used)

    if args.all or args.cda:
        print("Exporting CDA sample...")
        export_cda_sample(df)

    print("\nDONE.")
    print(f"Output root: {OUTPUT_DIR.resolve()}")
    print(f"Rows in analysis master: {len(df)}")
    print(f"Relevance 2+ rows: {len(df[df['Final_Relevance'].isin([2, 3, 4])])}")
    print(f"Relevance 3+ rows: {len(df[df['Final_Relevance'].isin([3, 4])])}")
    print(f"Relevance 4 rows: {len(df[df['Final_Relevance'] == 4])}")


if __name__ == "__main__":
    main()