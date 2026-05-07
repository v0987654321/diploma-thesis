import os
import re
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
BACKEND_OPTIONS = {
    "1": "GEMINI",
    "2": "LOCAL",
    "gemini": "GEMINI",
    "local": "LOCAL",
}

ROOT_DATA_DIR = "data"
GEMINI_DATA_DIR = os.path.join("GEMINI", "data")
LOCAL_DATA_DIR = os.path.join("LOCAL", "data")

SHARED_CONSOLIDATED_PATH = os.path.join(ROOT_DATA_DIR, "postConsolidated.csv")
SHARED_STEPB_PATH = os.path.join(ROOT_DATA_DIR, "postStepB.csv")

GEMINI_ADJ_PATH = os.path.join(GEMINI_DATA_DIR, "final_conservative_adjudicated_table.csv")
LOCAL_ADJ_PATH = os.path.join(LOCAL_DATA_DIR, "final_local_conservative_adjudicated_table.csv")

OUT_CANDIDATES_CSV = "postStepC_candidates.csv"
OUT_REVIEW_CSV = "postStepC_review.csv"
OUT_SUMMARY_TXT = "postStepC_summary.txt"

TARGET_TOTAL = 24
TARGET_REL4 = 20
MAX_REL3 = 4

CATEGORY_TARGETS = {
    "dominant": 6,
    "strong": 5,
    "anomalous": 5,
    "transitional": 4,
    "outlet_contrastive": 4,
}

MAX_PER_OUTLET = 3
MIN_ARTICLE_SENT_COUNT = 8

DUPLICATE_PENALTY = 2.0
REPUBLISH_PENALTY = 1.0

# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def safe_int(x, default=None):
    if pd.isna(x) or x is None:
        return default
    try:
        return int(float(x))
    except Exception:
        return default

def safe_float(x, default=0.0):
    if pd.isna(x) or x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default

def normalize_article_id(x):
    if pd.isna(x) or x is None:
        return None
    digits = re.sub(r"\D", "", str(x))
    if not digits:
        return None
    return digits.zfill(6)

def ensure_cols(df, cols, fill_value=None):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df

def choose_backend():
    print("\nChoose backend for StepC:")
    print("  1 = GEMINI")
    print("  2 = LOCAL")
    while True:
        choice = input("Enter choice [1/2]: ").strip().lower()
        if choice in BACKEND_OPTIONS:
            return BACKEND_OPTIONS[choice]
        print("Invalid choice. Please enter 1 or 2.")

def backend_paths(backend):
    if backend == "GEMINI":
        return {
            "backend_data_dir": GEMINI_DATA_DIR,
            "adj_path": GEMINI_ADJ_PATH
        }
    elif backend == "LOCAL":
        return {
            "backend_data_dir": LOCAL_DATA_DIR,
            "adj_path": LOCAL_ADJ_PATH
        }
    else:
        raise ValueError(f"Unsupported backend: {backend}")

def split_sentences(text):
    text = safe_str(text)
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

# =========================
# LOADERS
# =========================
def load_shared_consolidated(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required shared consolidated file: {path}")

    df = pd.read_csv(path, dtype={"Article_ID": str}, low_memory=False)
    df["Article_ID"] = df["Article_ID"].apply(normalize_article_id)

    expected = [
        "Article_ID", "Outlet", "Date", "Headline", "Lead", "Body_Postclean", "URL",
        "Relevance", "Relevance_Label", "Relevance_Note",
        "Actor_Mention", "Successor_Frame", "Dominant_Label", "Dominant_Location",
        "Main_Associated_Actor",
        "Counterterrorism", "Sovereignty", "Human_Rights_Abuse", "Anti_or_Neocolonialism",
        "Western_Failure", "Security_Effectiveness", "Economic_Interests", "Geopolitical_Rivalry",
        "Stance_Support", "Ambivalence_Support", "Legitimation_Support",
        "Dominant_Discourse_Support",
        "Review_Flag_Count", "Any_Review_Flag", "Review_Sources",
        "target_hits_total", "mali_hits_total", "non_mali_hits_total",
        "strong_mali_focus_hits", "mali_specific_linkage_hits", "generic_linkage_hits",
    ]
    df = ensure_cols(df, expected)
    return df

def load_stepb_optional(path):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, dtype={"article_id": str}, low_memory=False)
    df["article_id"] = df["article_id"].apply(normalize_article_id)

    keep = [
        "article_id",
        "near_duplicate_flag",
        "near_duplicate_cluster_id",
        "near_duplicate_match_count",
        "near_duplicate_cross_outlet_flag",
        "likely_republished_flag",
        "republication_index",
        "republication_confidence",
        "source_attributed_flag",
        "explicit_external_source_flag",
        "explicit_malian_media_reference_flag",
        "likely_republished_basis",
    ]
    df = ensure_cols(df, keep)
    return df[keep].copy()

def load_adjudicated(backend, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing adjudicated file for backend {backend}: {path}")

    df = pd.read_csv(path, dtype={"Article_ID": str}, low_memory=False)
    df["Article_ID"] = df["Article_ID"].apply(normalize_article_id)

    expected = [
        "Article_ID", "Outlet", "Date",
        "Adj_V04_Relevance", "Adj_V05_Actor_Mention", "Adj_V06_Successor_Frame",
        "Adj_V07_Dominant_Label", "Adj_V08_Stance", "Adj_V09_Dominant_Location",
        "Adj_V10_Ambivalence", "Adj_V11_Legitimation",
        "Adj_V12_Counterterrorism", "Adj_V13_Sovereignty", "Adj_V14_Human_Rights_Abuse",
        "Adj_V15_Anti_or_Neocolonialism", "Adj_V16_Western_Failure",
        "Adj_V17_Security_Effectiveness", "Adj_V18_Economic_Interests",
        "Adj_V19_Geopolitical_Rivalry", "Adj_V20_Main_Associated_Actor",
        "Adj_V21_Dominant_Discourse",
        "Manual_Check_Required", "Manual_Check_Reasons",
        "LLM_Review_Note", "Pro_Review_Candidate", "Pro_Review_Reason"
    ]
    df = ensure_cols(df, expected)
    return df

# =========================
# MERGE AND HARMONIZATION
# =========================
def merge_inputs(df_cons, df_adj, df_stepb=None):
    df = df_cons.merge(df_adj, on="Article_ID", how="left", suffixes=("", "_adj"))

    if df_stepb is not None:
        df = df.merge(df_stepb, left_on="Article_ID", right_on="article_id", how="left")
    else:
        df["near_duplicate_flag"] = 0
        df["near_duplicate_cross_outlet_flag"] = 0
        df["likely_republished_flag"] = 0
        df["republication_index"] = 0
        df["republication_confidence"] = ""
        df["source_attributed_flag"] = 0
        df["explicit_external_source_flag"] = 0
        df["explicit_malian_media_reference_flag"] = 0
        df["likely_republished_basis"] = ""

    # IMPORTANT FIX: row-wise fallback, not df.get() inside Series apply
    df["Final_Relevance"] = df.apply(
        lambda row: safe_int(row.get("Adj_V04_Relevance"), safe_int(row.get("Relevance"), 0)),
        axis=1
    )

    df["Final_Actor_Mention"] = df.apply(
        lambda row: safe_int(row.get("Adj_V05_Actor_Mention"), safe_int(row.get("Actor_Mention"), None)),
        axis=1
    )
    df["Final_Successor_Frame"] = df.apply(
        lambda row: safe_int(row.get("Adj_V06_Successor_Frame"), safe_int(row.get("Successor_Frame"), None)),
        axis=1
    )
    df["Final_Dominant_Label"] = df.apply(
        lambda row: safe_int(row.get("Adj_V07_Dominant_Label"), safe_int(row.get("Dominant_Label"), None)),
        axis=1
    )
    df["Final_Stance"] = df.apply(
        lambda row: safe_int(row.get("Adj_V08_Stance"), safe_int(row.get("Stance_Support"), None)),
        axis=1
    )
    df["Final_Dominant_Location"] = df.apply(
        lambda row: safe_int(row.get("Adj_V09_Dominant_Location"), safe_int(row.get("Dominant_Location"), None)),
        axis=1
    )
    df["Final_Ambivalence"] = df.apply(
        lambda row: safe_int(row.get("Adj_V10_Ambivalence"), safe_int(row.get("Ambivalence_Support"), 0)),
        axis=1
    )
    df["Final_Legitimation"] = df.apply(
        lambda row: safe_int(row.get("Adj_V11_Legitimation"), safe_int(row.get("Legitimation_Support"), None)),
        axis=1
    )

    df["Final_Counterterrorism"] = df.apply(
        lambda row: safe_int(row.get("Adj_V12_Counterterrorism"), safe_int(row.get("Counterterrorism"), 0)),
        axis=1
    )
    df["Final_Sovereignty"] = df.apply(
        lambda row: safe_int(row.get("Adj_V13_Sovereignty"), safe_int(row.get("Sovereignty"), 0)),
        axis=1
    )
    df["Final_Human_Rights_Abuse"] = df.apply(
        lambda row: safe_int(row.get("Adj_V14_Human_Rights_Abuse"), safe_int(row.get("Human_Rights_Abuse"), 0)),
        axis=1
    )
    df["Final_Anti_or_Neocolonialism"] = df.apply(
        lambda row: safe_int(row.get("Adj_V15_Anti_or_Neocolonialism"), safe_int(row.get("Anti_or_Neocolonialism"), 0)),
        axis=1
    )
    df["Final_Western_Failure"] = df.apply(
        lambda row: safe_int(row.get("Adj_V16_Western_Failure"), safe_int(row.get("Western_Failure"), 0)),
        axis=1
    )
    df["Final_Security_Effectiveness"] = df.apply(
        lambda row: safe_int(row.get("Adj_V17_Security_Effectiveness"), safe_int(row.get("Security_Effectiveness"), 0)),
        axis=1
    )
    df["Final_Economic_Interests"] = df.apply(
        lambda row: safe_int(row.get("Adj_V18_Economic_Interests"), safe_int(row.get("Economic_Interests"), 0)),
        axis=1
    )
    df["Final_Geopolitical_Rivalry"] = df.apply(
        lambda row: safe_int(row.get("Adj_V19_Geopolitical_Rivalry"), safe_int(row.get("Geopolitical_Rivalry"), 0)),
        axis=1
    )
    df["Final_Main_Associated_Actor"] = df.apply(
        lambda row: safe_int(row.get("Adj_V20_Main_Associated_Actor"), safe_int(row.get("Main_Associated_Actor"), None)),
        axis=1
    )
    df["Final_Dominant_Discourse"] = df.apply(
        lambda row: safe_int(row.get("Adj_V21_Dominant_Discourse"), safe_int(row.get("Dominant_Discourse_Support"), None)),
        axis=1
    )

    df["article_sentence_count_proxy"] = (
        df["Headline"].apply(safe_str) + " " +
        df["Lead"].apply(safe_str) + " " +
        df["Body_Postclean"].apply(safe_str)
    ).apply(lambda x: len(split_sentences(x)))

    df["is_bulletin_proxy"] = df["Relevance_Note"].apply(lambda x: 1 if "bulletin-style" in safe_str(x).lower() else 0)

    return df

# =========================
# ELIGIBILITY
# =========================
def is_nontrivial_text(row):
    body = safe_str(row.get("Body_Postclean", ""))
    headline = safe_str(row.get("Headline", ""))
    return bool(body and headline and len(body) > 150)

def enough_sentences(row):
    return safe_int(row.get("article_sentence_count_proxy"), 0) >= MIN_ARTICLE_SENT_COUNT

def enough_target_development(row):
    note = safe_str(row.get("Relevance_Note", "")).lower()

    if "target sentence count=1" in note:
        return False

    if safe_int(row.get("target_hits_total"), 0) >= 2:
        return True

    return safe_int(row.get("mali_specific_linkage_hits"), 0) >= 1

def compute_eligibility(row):
    reasons = []
    rel = safe_int(row.get("Final_Relevance"), safe_int(row.get("Relevance"), 0))

    if rel not in [3, 4]:
        reasons.append("relevance_not_3_or_4")
    if safe_int(row.get("is_bulletin_proxy"), 0) == 1:
        reasons.append("bulletin_style")
    if not is_nontrivial_text(row):
        reasons.append("nontrivial_text_fail")
    if not enough_sentences(row):
        reasons.append("too_short")
    if not enough_target_development(row):
        reasons.append("weak_target_development")

    eligible = 1 if len(reasons) == 0 else 0
    return eligible, "; ".join(reasons) if reasons else "eligible"

# =========================
# FEATURE ENGINEERING
# =========================
def assign_period_group(row):
    date_str = safe_str(row.get("Date", ""))
    year = None

    m = re.match(r"(\d{4})-(\d{2})", date_str)
    if m:
        year = int(m.group(1))

    if year is None:
        return "unknown"

    if year <= 2022:
        return "wagner_period"
    elif year == 2023:
        return "transition_period"
    else:
        return "africa_corps_period"

def discursive_density_score(row):
    score = 0.0
    sent_count = safe_int(row.get("article_sentence_count_proxy"), 0)

    if sent_count >= 25:
        score += 3
    elif sent_count >= 15:
        score += 2
    elif sent_count >= 8:
        score += 1

    target_hits = safe_int(row.get("target_hits_total"), 0)
    if target_hits >= 4:
        score += 2
    elif target_hits >= 2:
        score += 1

    frame_sum = sum([
        safe_int(row.get("Final_Counterterrorism"), 0),
        safe_int(row.get("Final_Sovereignty"), 0),
        safe_int(row.get("Final_Human_Rights_Abuse"), 0),
        safe_int(row.get("Final_Anti_or_Neocolonialism"), 0),
        safe_int(row.get("Final_Western_Failure"), 0),
        safe_int(row.get("Final_Security_Effectiveness"), 0),
        safe_int(row.get("Final_Economic_Interests"), 0),
        safe_int(row.get("Final_Geopolitical_Rivalry"), 0),
    ])

    if frame_sum >= 3:
        score += 2
    elif frame_sum >= 1:
        score += 1

    if safe_int(row.get("Final_Dominant_Discourse"), None) in [1, 2, 3, 4]:
        score += 1

    if safe_int(row.get("Final_Main_Associated_Actor"), None) not in [None, 9, 10]:
        score += 1

    return score

def build_pattern_key(row):
    parts = [
        f"stance={safe_int(row.get('Final_Stance'), -1)}",
        f"disc={safe_int(row.get('Final_Dominant_Discourse'), -1)}",
        f"actor={safe_int(row.get('Final_Main_Associated_Actor'), -1)}",
        f"succ={safe_int(row.get('Final_Successor_Frame'), -1)}"
    ]
    return "|".join(parts)

# =========================
# CATEGORY SCORING
# =========================
def compute_pattern_frequencies(df):
    df["pattern_key"] = df.apply(build_pattern_key, axis=1)
    pattern_counts = df["pattern_key"].value_counts(dropna=False).to_dict()
    outlet_pattern_counts = (
        df.groupby(["Outlet", "pattern_key"]).size().reset_index(name="n")
    )
    return pattern_counts, outlet_pattern_counts

def add_outlet_modes(df):
    outlet_disc_mode = (
        df.groupby("Outlet")["Final_Dominant_Discourse"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .to_dict()
    )
    outlet_stance_mode = (
        df.groupby("Outlet")["Final_Stance"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        .to_dict()
    )

    df["outlet_disc_mode"] = df["Outlet"].map(outlet_disc_mode)
    df["outlet_stance_mode"] = df["Outlet"].map(outlet_stance_mode)
    return df

def dominant_case_score(row, pattern_counts):
    key = safe_str(row.get("pattern_key", ""))
    freq = pattern_counts.get(key, 0)
    score = min(freq, 10) * 0.5
    score += discursive_density_score(row) * 0.5
    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 1.0
    return score

def strong_case_score(row):
    score = 0.0
    stance = safe_int(row.get("Final_Stance"), None)
    legit = safe_int(row.get("Final_Legitimation"), None)
    amb = safe_int(row.get("Final_Ambivalence"), 0)

    if stance in [1, 3]:
        score += 2.5
    if legit in [1, 3]:
        score += 2.0
    if amb == 0:
        score += 0.5

    frame_sum = sum([
        safe_int(row.get("Final_Counterterrorism"), 0),
        safe_int(row.get("Final_Sovereignty"), 0),
        safe_int(row.get("Final_Human_Rights_Abuse"), 0),
        safe_int(row.get("Final_Anti_or_Neocolonialism"), 0),
        safe_int(row.get("Final_Western_Failure"), 0),
        safe_int(row.get("Final_Security_Effectiveness"), 0),
        safe_int(row.get("Final_Economic_Interests"), 0),
        safe_int(row.get("Final_Geopolitical_Rivalry"), 0),
    ])

    if frame_sum >= 2:
        score += 1.0

    score += discursive_density_score(row) * 0.4
    return score

def anomaly_score(row):
    score = 0.0
    stance = safe_int(row.get("Final_Stance"), None)
    disc = safe_int(row.get("Final_Dominant_Discourse"), None)
    outlet_stance_mode = safe_int(row.get("outlet_stance_mode"), None)
    outlet_disc_mode = safe_int(row.get("outlet_disc_mode"), None)

    if stance is not None and outlet_stance_mode is not None and stance != outlet_stance_mode:
        score += 2.0
    if disc is not None and outlet_disc_mode is not None and disc != outlet_disc_mode:
        score += 2.0

    if safe_str(row.get("pattern_key", "")):
        score += 0.5

    if safe_int(row.get("Final_Relevance"), 0) == 3:
        score += 0.5

    score += discursive_density_score(row) * 0.3
    return score

def transition_score(row):
    score = 0.0
    actor_mention = safe_int(row.get("Final_Actor_Mention"), None)
    successor = safe_int(row.get("Final_Successor_Frame"), 0)
    period = safe_str(row.get("period_group", ""))

    if successor == 1:
        score += 3.0
    if actor_mention == 3:
        score += 2.0
    if period == "transition_period":
        score += 1.5
    if actor_mention == 2:
        score += 1.0

    score += discursive_density_score(row) * 0.3
    return score

def outlet_contrast_score(row):
    score = 0.0
    disc = safe_int(row.get("Final_Dominant_Discourse"), None)
    stance = safe_int(row.get("Final_Stance"), None)
    outlet_disc_mode = safe_int(row.get("outlet_disc_mode"), None)
    outlet_stance_mode = safe_int(row.get("outlet_stance_mode"), None)

    if disc is not None and outlet_disc_mode is not None and disc != outlet_disc_mode:
        score += 1.5
    if stance is not None and outlet_stance_mode is not None and stance != outlet_stance_mode:
        score += 1.5

    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 1.0

    score += discursive_density_score(row) * 0.3
    return score

# =========================
# RELEVANCE 3 JUSTIFICATION
# =========================
def relevance3_justification(row):
    rel = safe_int(row.get("Final_Relevance"), 0)
    if rel != 3:
        return 0, ""

    reasons = []

    if safe_float(row.get("transition_score"), 0.0) >= 3.0:
        reasons.append("transitional_value")
    if safe_float(row.get("anomaly_score"), 0.0) >= 3.0:
        reasons.append("anomalous_value")
    if safe_float(row.get("outlet_contrast_score"), 0.0) >= 2.5:
        reasons.append("outlet_contrastive_value")
    if safe_float(row.get("discursive_density_score"), 0.0) >= 4.0:
        reasons.append("discursive_density")

    justified = 1 if len(reasons) > 0 else 0
    return justified, "; ".join(reasons)

def overall_priority_score(row):
    score = 0.0
    rel = safe_int(row.get("Final_Relevance"), 0)

    if rel == 4:
        score += 4.0
    elif rel == 3:
        if safe_int(row.get("relevance3_justified"), 0) == 1:
            score += 2.0
        else:
            score -= 3.0

    score += safe_float(row.get("discursive_density_score"), 0.0)
    score += max(
        safe_float(row.get("dominant_case_score"), 0.0),
        safe_float(row.get("strong_case_score"), 0.0),
        safe_float(row.get("anomaly_score"), 0.0),
        safe_float(row.get("transition_score"), 0.0),
        safe_float(row.get("outlet_contrast_score"), 0.0),
    )

    if safe_int(row.get("near_duplicate_flag"), 0) == 1:
        score -= DUPLICATE_PENALTY
    if safe_int(row.get("likely_republished_flag"), 0) == 1:
        score -= REPUBLISH_PENALTY

    return score

# =========================
# CATEGORY LABELS
# =========================
def assign_primary_secondary_categories(row):
    scores = {
        "dominant": safe_float(row.get("dominant_case_score"), 0.0),
        "strong": safe_float(row.get("strong_case_score"), 0.0),
        "anomalous": safe_float(row.get("anomaly_score"), 0.0),
        "transitional": safe_float(row.get("transition_score"), 0.0),
        "outlet_contrastive": safe_float(row.get("outlet_contrast_score"), 0.0),
    }

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = ordered[0][0] if ordered else ""
    secondary = ordered[1][0] if len(ordered) > 1 else ""
    return primary, secondary

# =========================
# BALANCED ASSEMBLY
# =========================
def add_candidates_from_category(df_pool, chosen_ids, outlet_counts, category_name, n_target):
    score_col = {
        "dominant": "dominant_case_score",
        "strong": "strong_case_score",
        "anomalous": "anomaly_score",
        "transitional": "transition_score",
        "outlet_contrastive": "outlet_contrast_score",
    }[category_name]

    df_cat = df_pool.copy()
    df_cat = df_cat.sort_values(
        by=[score_col, "overall_priority_score", "discursive_density_score"],
        ascending=[False, False, False]
    )

    added = []

    for _, row in df_cat.iterrows():
        aid = safe_str(row["Article_ID"])
        outlet = safe_str(row["Outlet"])
        rel = safe_int(row.get("Final_Relevance"), 0)

        if aid in chosen_ids:
            continue
        if outlet_counts.get(outlet, 0) >= MAX_PER_OUTLET:
            continue

        if rel == 3 and safe_int(row.get("relevance3_justified"), 0) != 1:
            continue

        chosen_ids.add(aid)
        outlet_counts[outlet] = outlet_counts.get(outlet, 0) + 1
        added.append(aid)

        if len(added) >= n_target:
            break

    return chosen_ids, outlet_counts, added

def rebalance_relevance(df_selected):
    rel3 = df_selected[df_selected["Final_Relevance"] == 3].copy()

    if len(rel3) <= MAX_REL3:
        return df_selected

    rel3 = rel3.sort_values(
        by=["overall_priority_score", "discursive_density_score"],
        ascending=[True, True]
    )
    to_drop = rel3.head(len(rel3) - MAX_REL3)["Article_ID"].tolist()
    df_selected = df_selected[~df_selected["Article_ID"].isin(to_drop)].copy()

    return df_selected

def fill_to_target(df_pool, df_selected):
    chosen_ids = set(df_selected["Article_ID"].tolist())
    outlet_counts = df_selected["Outlet"].value_counts().to_dict()
    current_rel3 = int((df_selected["Final_Relevance"] == 3).sum())

    df_pool_sorted = df_pool.sort_values(
        by=["overall_priority_score", "discursive_density_score"],
        ascending=[False, False]
    )

    for _, row in df_pool_sorted.iterrows():
        if len(df_selected) >= TARGET_TOTAL:
            break

        aid = safe_str(row["Article_ID"])
        outlet = safe_str(row["Outlet"])
        rel = safe_int(row.get("Final_Relevance"), 0)

        if aid in chosen_ids:
            continue
        if outlet_counts.get(outlet, 0) >= MAX_PER_OUTLET:
            continue

        if rel == 4:
            pass
        elif rel == 3:
            if current_rel3 >= MAX_REL3:
                continue
            if safe_int(row.get("relevance3_justified"), 0) != 1:
                continue
        else:
            continue

        df_selected = pd.concat([df_selected, row.to_frame().T], ignore_index=True)
        chosen_ids.add(aid)
        outlet_counts[outlet] = outlet_counts.get(outlet, 0) + 1

        if rel == 3:
            current_rel3 += 1

    return df_selected

def assemble_candidate_pool(df):
    df_pool = df[df["stepC_eligible"] == 1].copy()
    df_pool = df_pool[df_pool["Final_Relevance"].isin([3, 4])].copy()

    chosen_ids = set()
    outlet_counts = {}
    category_additions = {}

    for cat, target_n in CATEGORY_TARGETS.items():
        chosen_ids, outlet_counts, added = add_candidates_from_category(
            df_pool, chosen_ids, outlet_counts, cat, target_n
        )
        category_additions[cat] = added

    df_selected = df_pool[df_pool["Article_ID"].isin(chosen_ids)].copy()
    df_selected = rebalance_relevance(df_selected)
    df_selected = fill_to_target(df_pool, df_selected)

    df_selected = df_selected.sort_values(
        by=["overall_priority_score", "discursive_density_score"],
        ascending=[False, False]
    ).head(TARGET_TOTAL).copy()

    return df_selected, category_additions

# =========================
# EXPORT HELPERS
# =========================
def build_stepc_notes(row):
    notes = []
    notes.append(f"primary={safe_str(row.get('primary_stepC_category'))}")
    notes.append(f"secondary={safe_str(row.get('secondary_stepC_category'))}")
    notes.append(f"rel={safe_int(row.get('Final_Relevance'), -1)}")
    notes.append(f"density={safe_float(row.get('discursive_density_score'), 0):.2f}")

    if safe_int(row.get("relevance3_justified"), 0) == 1:
        notes.append(f"rel3_justified={safe_str(row.get('relevance3_justification_note'))}")
    if safe_int(row.get("near_duplicate_flag"), 0) == 1:
        notes.append("near_duplicate_flag=1")
    if safe_int(row.get("likely_republished_flag"), 0) == 1:
        notes.append("likely_republished_flag=1")

    return "; ".join(notes)

def export_outputs(df_selected, backend_data_dir, backend_name, category_additions, df_all):
    os.makedirs(backend_data_dir, exist_ok=True)

    df_selected = df_selected.copy()
    df_selected["stepC_candidate_rank"] = range(1, len(df_selected) + 1)
    df_selected["stepC_candidate_note"] = df_selected.apply(build_stepc_notes, axis=1)

    df_selected["researcher_keep"] = ""
    df_selected["researcher_final_category"] = ""
    df_selected["researcher_note"] = ""

    candidate_cols = [
        "Article_ID", "Outlet", "Date", "Headline", "URL",
        "Final_Relevance",
        "discursive_density_score",
        "dominant_case_score",
        "strong_case_score",
        "anomaly_score",
        "transition_score",
        "outlet_contrast_score",
        "primary_stepC_category",
        "secondary_stepC_category",
        "relevance3_justified",
        "relevance3_justification_note",
        "stepC_candidate_rank",
        "stepC_candidate_note",
        "researcher_keep",
        "researcher_final_category",
        "researcher_note"
    ]
    candidate_cols = [c for c in candidate_cols if c in df_selected.columns]

    review_cols = [
        "Article_ID", "Outlet", "Date", "Headline", "Lead", "Body_Postclean", "URL",
        "Final_Relevance", "Final_Stance", "Final_Legitimation", "Final_Dominant_Discourse",
        "Final_Main_Associated_Actor", "Final_Successor_Frame", "Final_Dominant_Label",
        "Final_Dominant_Location",
        "Final_Counterterrorism", "Final_Sovereignty", "Final_Human_Rights_Abuse",
        "Final_Anti_or_Neocolonialism", "Final_Western_Failure", "Final_Security_Effectiveness",
        "Final_Economic_Interests", "Final_Geopolitical_Rivalry",
        "discursive_density_score",
        "dominant_case_score", "strong_case_score", "anomaly_score",
        "transition_score", "outlet_contrast_score",
        "primary_stepC_category", "secondary_stepC_category",
        "relevance3_justified", "relevance3_justification_note",
        "near_duplicate_flag", "likely_republished_flag",
        "stepC_candidate_rank", "stepC_candidate_note",
        "researcher_keep", "researcher_final_category", "researcher_note"
    ]
    review_cols = [c for c in review_cols if c in df_selected.columns]

    out_candidates_csv = os.path.join(backend_data_dir, OUT_CANDIDATES_CSV)
    out_review_csv = os.path.join(backend_data_dir, OUT_REVIEW_CSV)
    out_summary_txt = os.path.join(backend_data_dir, OUT_SUMMARY_TXT)

    df_selected[candidate_cols].to_csv(out_candidates_csv, index=False, encoding="utf-8-sig")
    df_selected[review_cols].to_csv(out_review_csv, index=False, encoding="utf-8-sig")

    rel4_n = int((df_selected["Final_Relevance"] == 4).sum())
    rel3_n = int((df_selected["Final_Relevance"] == 3).sum())
    outlet_counts = df_selected["Outlet"].value_counts(dropna=False).to_dict()
    period_counts = df_selected["period_group"].value_counts(dropna=False).to_dict()
    primary_counts = df_selected["primary_stepC_category"].value_counts(dropna=False).to_dict()

    eligible_total = int((df_all["stepC_eligible"] == 1).sum())
    eligible_rel4 = int(((df_all["stepC_eligible"] == 1) & (df_all["Final_Relevance"] == 4)).sum())
    eligible_rel3 = int(((df_all["stepC_eligible"] == 1) & (df_all["Final_Relevance"] == 3)).sum())
    justified_rel3 = int(((df_all["stepC_eligible"] == 1) & (df_all["Final_Relevance"] == 3) & (df_all["relevance3_justified"] == 1)).sum())

    bulletin_excluded = int(df_all["stepC_eligibility_note"].fillna("").astype(str).str.contains("bulletin_style").sum())
    short_excluded = int(df_all["stepC_eligibility_note"].fillna("").astype(str).str.contains("too_short").sum())
    weak_target_excluded = int(df_all["stepC_eligibility_note"].fillna("").astype(str).str.contains("weak_target_development").sum())
    duplicate_penalized = int((df_all["near_duplicate_flag"].fillna(0).astype(int) == 1).sum()) if "near_duplicate_flag" in df_all.columns else 0
    republished_penalized = int((df_all["likely_republished_flag"].fillna(0).astype(int) == 1).sum()) if "likely_republished_flag" in df_all.columns else 0

    lines = []
    lines.append(f"StepC summary for backend: {backend_name}")
    lines.append("=" * 60)
    lines.append(f"Total candidates selected: {len(df_selected)}")
    lines.append(f"Relevance 4 selected: {rel4_n}")
    lines.append(f"Relevance 3 selected: {rel3_n}")
    lines.append("")
    lines.append("Eligibility summary:")
    lines.append(f"  - Eligible total: {eligible_total}")
    lines.append(f"  - Eligible relevance 4: {eligible_rel4}")
    lines.append(f"  - Eligible relevance 3: {eligible_rel3}")
    lines.append(f"  - Justified relevance 3: {justified_rel3}")
    lines.append("")
    lines.append("Exclusion / caution indicators:")
    lines.append(f"  - Bulletin-style exclusions flagged: {bulletin_excluded}")
    lines.append(f"  - Too-short exclusions flagged: {short_excluded}")
    lines.append(f"  - Weak target development exclusions flagged: {weak_target_excluded}")
    lines.append(f"  - Near-duplicate penalized cases available: {duplicate_penalized}")
    lines.append(f"  - Likely republished penalized cases available: {republished_penalized}")
    lines.append("")
    lines.append("Primary category distribution:")
    for k, v in primary_counts.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Outlet distribution:")
    for k, v in outlet_counts.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Period distribution:")
    for k, v in period_counts.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Category assembly additions:")
    for k, v in category_additions.items():
        lines.append(f"  - {k}: {len(v)} articles initially added")

    with open(out_summary_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSaved candidates to: {out_candidates_csv}")
    print(f"Saved review file to: {out_review_csv}")
    print(f"Saved summary to: {out_summary_txt}")

# =========================
# MAIN
# =========================
def main():
    backend = choose_backend()
    paths = backend_paths(backend)

    print(f"\nRunning StepC for backend: {backend}")
    print(f"Shared consolidated: {SHARED_CONSOLIDATED_PATH}")
    print(f"Shared StepB: {SHARED_STEPB_PATH}")
    print(f"Backend adjudicated: {paths['adj_path']}")

    df_cons = load_shared_consolidated(SHARED_CONSOLIDATED_PATH)
    df_adj = load_adjudicated(backend, paths["adj_path"])
    df_stepb = load_stepb_optional(SHARED_STEPB_PATH)

    df = merge_inputs(df_cons, df_adj, df_stepb)

    # Eligibility
    elig = df.apply(compute_eligibility, axis=1)
    df["stepC_eligible"] = elig.apply(lambda x: x[0])
    df["stepC_eligibility_note"] = elig.apply(lambda x: x[1])

    # Features
    df["period_group"] = df.apply(assign_period_group, axis=1)
    df["discursive_density_score"] = df.apply(discursive_density_score, axis=1)
    df = add_outlet_modes(df)

    pattern_counts, _ = compute_pattern_frequencies(df)
    df["dominant_case_score"] = df.apply(lambda row: dominant_case_score(row, pattern_counts), axis=1)
    df["strong_case_score"] = df.apply(strong_case_score, axis=1)
    df["anomaly_score"] = df.apply(anomaly_score, axis=1)
    df["transition_score"] = df.apply(transition_score, axis=1)
    df["outlet_contrast_score"] = df.apply(outlet_contrast_score, axis=1)

    # Relevance 3 justification
    rel3 = df.apply(relevance3_justification, axis=1)
    df["relevance3_justified"] = rel3.apply(lambda x: x[0])
    df["relevance3_justification_note"] = rel3.apply(lambda x: x[1])

    # Overall score
    df["overall_priority_score"] = df.apply(overall_priority_score, axis=1)

    # Category labels
    cats = df.apply(assign_primary_secondary_categories, axis=1)
    df["primary_stepC_category"] = cats.apply(lambda x: x[0])
    df["secondary_stepC_category"] = cats.apply(lambda x: x[1])

    # Assemble final 24-candidate pool
    df_selected, category_additions = assemble_candidate_pool(df)

    # Final sort for export
    df_selected = df_selected.sort_values(
        by=["overall_priority_score", "discursive_density_score"],
        ascending=[False, False]
    ).copy()

    export_outputs(df_selected, paths["backend_data_dir"], backend, category_additions, df)

    print("\nPreview:")
    preview_cols = [
        "Article_ID", "Outlet", "Headline", "Final_Relevance",
        "primary_stepC_category", "secondary_stepC_category",
        "discursive_density_score", "overall_priority_score"
    ]
    preview_cols = [c for c in preview_cols if c in df_selected.columns]
    print(df_selected[preview_cols].head(24))

if __name__ == "__main__":
    main()