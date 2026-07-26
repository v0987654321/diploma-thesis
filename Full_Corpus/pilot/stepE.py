# =============================================================================
# stepE.py
# Qualitative Review Corpus Selection for Article-Level Discourse Analysis
#
# Purpose:
# - Select a transparent, balanced qualitative review corpus for the article:
#   "Normalizing Without Praising? Wagner Group, Africa Corps, and the Limits
#   of Pro-Russian Framing in Malian Online News"
# - Combine selected thesis CDA cases with newly identified article candidates.
# - Export both CSV files and a human-readable TXT review corpus.
#
# The resulting TXT file is formatted for close reading and qualitative discourse
# analysis. It contains article metadata, lead, full text, coding context, source
# indicators, and selection rationale.
#
# Inputs:
#   pilot/data/postConsolidated.csv
#   pilot/GEMINI/data/final_conservative_adjudicated_table.csv
#   pilot/data/postStepB.csv                  [optional but recommended]
#   pilot/data/stepBA/discussion/
#       stepBA_russian_sources_with_final_coding.csv   [optional]
#
# Outputs:
#   pilot/data/stepEarticle/
#
# Important:
# - This script produces a candidate review corpus, not a final automatic
#   qualitative interpretation.
# - Final inclusion and discourse interpretation remain researcher decisions.
# =============================================================================

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / "data"
GEMINI_DATA_DIR = SCRIPT_DIR / "GEMINI" / "data"

INPUT_CONSOLIDATED = DATA_DIR / "postConsolidated.csv"
INPUT_ADJUDICATED = GEMINI_DATA_DIR / "final_conservative_adjudicated_table.csv"
INPUT_STEPB = DATA_DIR / "postStepB.csv"

# Optional Russian-source enrichment output from StepB.A
INPUT_STEPBA = (
    DATA_DIR
    / "stepBA"
    / "discussion"
    / "stepBA_russian_sources_with_final_coding.csv"
)

OUTPUT_DIR = DATA_DIR / "stepEarticle"

OUTPUT_ALL = OUTPUT_DIR / "stepEarticle_candidates_all.csv"
OUTPUT_RANKED = OUTPUT_DIR / "stepEarticle_ranked_candidates.csv"
OUTPUT_SELECTED_CSV = OUTPUT_DIR / "stepEarticle_selected_review_corpus.csv"
OUTPUT_SELECTED_TXT = OUTPUT_DIR / "stepEarticle_selected_review_corpus.txt"
OUTPUT_THESIS_CASES = OUTPUT_DIR / "stepEarticle_existing_thesis_cases.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "stepEarticle_summary.txt"


# =============================================================================
# SELECTION CONFIGURATION
# =============================================================================

# Only relevance 3/4 texts are eligible for the qualitative review corpus.
ELIGIBLE_RELEVANCE = {3, 4}

# Text-richness requirements. These exclude thin or bulletin-style items.
MIN_BODY_CHARS = 500
MIN_SENTENCES = 8

# Avoid allowing one outlet to dominate the selected corpus.
MAX_PER_OUTLET = 2

# Number of selected NEW candidates, in addition to selected thesis cases.
TARGET_NEW_CASES = 10

# Existing thesis CDA texts retained as established reference cases.
# The selection corpus will contain these where technically eligible.
THESIS_CDA_CASES = {
    "030194": "Wagner as a costly and risky security choice",
    "050036": "International condemnation and mercenary framing",
    "040079": "Contested human-rights discourse and criticism of HRW/France",
    "020264": "Anti-French displacement and French contradictions",
    "080002": "Wagner continuity and Africa Corps reclassification",
    "080027": "Foreign military tutelage critique involving Africa Corps",
}

# Desired composition of the new candidate group.
CATEGORY_TARGETS = {
    "hard_negative_wagner": 2,
    "mercenary_problematisation": 1,
    "source_attributed_critical": 2,
    "anti_french_displacement": 2,
    "russia_source_not_positive": 1,
    "africa_corps_reclassification": 2,
}


# =============================================================================
# LABEL MAPS
# =============================================================================

ACTOR_LABELS = {
    1: "Wagner Group",
    2: "Africa Corps",
    3: "Wagner Group and Africa Corps",
    4: "Indirect Russian contractors/forces",
    5: "Cannot determine",
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
    4: "Cannot determine",
}

DOMINANT_LABELS = {
    1: "Mercenaries",
    2: "Instructors/advisers",
    3: "Allies/partners",
    4: "Foreign/occupying forces",
    5: "Neutral designation",
    6: "Multiple/no clear dominance",
}

ASSOCIATED_ACTOR_LABELS = {
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


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def safe_str(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()


def safe_int(value, default=None):
    if pd.isna(value) or value is None:
        return default

    try:
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if np.isnan(value):
                return default
            return int(value) if float(value).is_integer() else default

        numeric_value = float(str(value).strip())
        return int(numeric_value) if numeric_value.is_integer() else default

    except Exception:
        return default


def safe_float(value, default=0.0):
    if pd.isna(value) or value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def normalize_article_id(value):
    raw = safe_str(value)
    if not raw:
        return None

    digits = re.sub(r"\D", "", raw)
    return digits.zfill(6) if digits else None


def ensure_columns(df, columns, fill_value=None):
    for column in columns:
        if column not in df.columns:
            df[column] = fill_value
    return df


def coalesce_int(row, preferred_col, fallback_col, default=None):
    preferred = safe_int(row.get(preferred_col), None)
    if preferred is not None:
        return preferred
    return safe_int(row.get(fallback_col), default)


def split_sentences(text):
    text = safe_str(text)
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[\.\!\?])\s+", text)
        if sentence.strip()
    ]


def build_full_text(row):
    parts = [
        safe_str(row.get("Headline")),
        safe_str(row.get("Lead")),
        safe_str(row.get("Body_Postclean")),
    ]
    return "\n\n".join([part for part in parts if part]).strip()


def article_sentence_count(row):
    return len(split_sentences(build_full_text(row)))


def article_period(date_value):
    date = pd.to_datetime(date_value, errors="coerce")

    if pd.isna(date):
        return "Unknown period"

    if date < pd.Timestamp("2023-06-01"):
        return "Early Wagner period"

    if date < pd.Timestamp("2024-01-01"):
        return "Mutiny transition period"

    return "Africa Corps period"


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =============================================================================
# LOAD INPUTS
# =============================================================================

def load_required_inputs():
    if not INPUT_CONSOLIDATED.exists():
        raise FileNotFoundError(
            f"Missing consolidated corpus file:\n{INPUT_CONSOLIDATED}"
        )

    if not INPUT_ADJUDICATED.exists():
        raise FileNotFoundError(
            f"Missing conservative adjudicated coding file:\n{INPUT_ADJUDICATED}"
        )

    consolidated = pd.read_csv(
        INPUT_CONSOLIDATED,
        dtype={"Article_ID": str},
        low_memory=False
    )

    adjudicated = pd.read_csv(
        INPUT_ADJUDICATED,
        dtype={"Article_ID": str},
        low_memory=False
    )

    consolidated["Article_ID"] = consolidated["Article_ID"].apply(normalize_article_id)
    adjudicated["Article_ID"] = adjudicated["Article_ID"].apply(normalize_article_id)

    return consolidated, adjudicated


def load_stepb_optional():
    if not INPUT_STEPB.exists():
        print("Warning: StepB output not found. Source/republication fields will default to zero.")
        return None

    stepb = pd.read_csv(
        INPUT_STEPB,
        dtype={"article_id": str},
        low_memory=False
    )

    stepb["article_id"] = stepb["article_id"].apply(normalize_article_id)

    keep = [
        "article_id",
        "source_attributed_flag",
        "likely_republished_flag",
        "near_duplicate_flag",
        "near_duplicate_cluster_id",
        "near_duplicate_match_count",
        "near_duplicate_cross_outlet_flag",
        "near_duplicate_top_match_id",
        "near_duplicate_top_match_outlet",
        "near_duplicate_top_match_score",
        "republication_index",
        "republication_confidence",
        "likely_republished_basis",
        "explicit_external_source_flag",
        "explicit_external_source_type",
        "explicit_external_source_name",
        "author_external_source_flag",
        "author_external_source_name",
        "explicit_malian_media_reference_flag",
        "explicit_malian_media_reference_names",
    ]

    stepb = ensure_columns(stepb, keep)
    return stepb[keep].copy()


def load_stepba_optional():
    if not INPUT_STEPBA.exists():
        print("Note: StepB.A Russian-source enrichment not found. Russian-source categories will be inactive.")
        return None

    stepba = pd.read_csv(
        INPUT_STEPBA,
        dtype={"article_id": str},
        low_memory=False
    )

    stepba["article_id"] = stepba["article_id"].apply(normalize_article_id)

    keep = [
        "article_id",
        "Russian_Source_Present",
        "Russian_Source_Dominant",
        "Russian_State_Source_Present",
        "Russian_State_Media_Source_Present",
        "Russian_Official_Source_Present",
        "Russian_Embassy_Source_Present",
        "ProRussian_Source_Present",
        "Russia_Attributed_Claim_Present",
        "Russian_Source_Type",
        "Russian_Source_Names",
        "Russian_Source_Evidence",
        "Russian_Source_Score",
        "Russian_Attribution_Score",
        "NonRussian_Source_Score",
        "Russian_Source_Confidence",
        "Russian_Source_Review_Flag",
        "Russian_Source_Review_Reason",
    ]

    stepba = ensure_columns(stepba, keep)
    return stepba[keep].copy()


# =============================================================================
# DATA HARMONIZATION
# =============================================================================

def prepare_master(consolidated, adjudicated, stepb=None, stepba=None):
    adj_keep = [
        "Article_ID",
        "Adj_V04_Relevance",
        "Adj_V05_Actor_Mention",
        "Adj_V06_Successor_Frame",
        "Adj_V07_Dominant_Label",
        "Adj_V08_Stance",
        "Adj_V09_Dominant_Location",
        "Adj_V10_Ambivalence",
        "Adj_V11_Legitimation",
        "Adj_V12_Counterterrorism",
        "Adj_V13_Sovereignty",
        "Adj_V14_Human_Rights_Abuse",
        "Adj_V15_Anti_or_Neocolonialism",
        "Adj_V16_Western_Failure",
        "Adj_V17_Security_Effectiveness",
        "Adj_V18_Economic_Interests",
        "Adj_V19_Geopolitical_Rivalry",
        "Adj_V20_Main_Associated_Actor",
        "Adj_V21_Dominant_Discourse",
        "Manual_Check_Required",
        "Manual_Check_Reasons",
        "LLM_Review_Note",
        "Pro_Review_Candidate",
        "Pro_Review_Reason",
    ]

    adjudicated = ensure_columns(adjudicated, adj_keep)
    master = consolidated.merge(adjudicated[adj_keep], on="Article_ID", how="left")

    if stepb is not None:
        master = master.merge(
            stepb,
            left_on="Article_ID",
            right_on="article_id",
            how="left"
        )
    else:
        default_stepb = {
            "source_attributed_flag": 0,
            "likely_republished_flag": 0,
            "near_duplicate_flag": 0,
            "near_duplicate_cluster_id": "",
            "near_duplicate_match_count": 0,
            "near_duplicate_cross_outlet_flag": 0,
            "near_duplicate_top_match_id": "",
            "near_duplicate_top_match_outlet": "",
            "near_duplicate_top_match_score": 0,
            "republication_index": 0,
            "republication_confidence": "",
            "likely_republished_basis": "",
            "explicit_external_source_flag": 0,
            "explicit_external_source_name": "",
        }

        for column, default_value in default_stepb.items():
            master[column] = default_value

    if stepba is not None:
        stepba_merge = stepba.copy()
        source_cols = [column for column in stepba_merge.columns if column != "article_id"]

        master = master.merge(
            stepba_merge[["article_id"] + source_cols],
            left_on="Article_ID",
            right_on="article_id",
            how="left",
            suffixes=("", "_stepBA")
        )
    else:
        default_stepba = {
            "Russian_Source_Present": 0,
            "Russian_Source_Dominant": 0,
            "Russian_State_Source_Present": 0,
            "Russian_State_Media_Source_Present": 0,
            "Russian_Official_Source_Present": 0,
            "Russian_Embassy_Source_Present": 0,
            "ProRussian_Source_Present": 0,
            "Russia_Attributed_Claim_Present": 0,
            "Russian_Source_Type": "none",
            "Russian_Source_Names": "",
            "Russian_Source_Evidence": "",
            "Russian_Source_Score": 0,
            "Russian_Attribution_Score": 0,
            "NonRussian_Source_Score": 0,
            "Russian_Source_Confidence": "none",
            "Russian_Source_Review_Flag": 0,
            "Russian_Source_Review_Reason": "",
        }

        for column, default_value in default_stepba.items():
            master[column] = default_value

    original_cols = [
        "Relevance",
        "Actor_Mention",
        "Successor_Frame",
        "Dominant_Label",
        "Stance_Support",
        "Dominant_Location",
        "Ambivalence_Support",
        "Legitimation_Support",
        "Counterterrorism",
        "Sovereignty",
        "Human_Rights_Abuse",
        "Anti_or_Neocolonialism",
        "Western_Failure",
        "Security_Effectiveness",
        "Economic_Interests",
        "Geopolitical_Rivalry",
        "Main_Associated_Actor",
        "Dominant_Discourse_Support",
        "Headline",
        "Lead",
        "Body_Postclean",
        "Outlet",
        "Date",
        "URL",
        "Relevance_Note",
    ]

    master = ensure_columns(master, original_cols)

    final_map = {
        "Final_Relevance": ("Adj_V04_Relevance", "Relevance"),
        "Final_Actor_Mention": ("Adj_V05_Actor_Mention", "Actor_Mention"),
        "Final_Successor_Frame": ("Adj_V06_Successor_Frame", "Successor_Frame"),
        "Final_Dominant_Label": ("Adj_V07_Dominant_Label", "Dominant_Label"),
        "Final_Stance": ("Adj_V08_Stance", "Stance_Support"),
        "Final_Dominant_Location": ("Adj_V09_Dominant_Location", "Dominant_Location"),
        "Final_Ambivalence": ("Adj_V10_Ambivalence", "Ambivalence_Support"),
        "Final_Legitimation": ("Adj_V11_Legitimation", "Legitimation_Support"),
        "Final_Counterterrorism": ("Adj_V12_Counterterrorism", "Counterterrorism"),
        "Final_Sovereignty": ("Adj_V13_Sovereignty", "Sovereignty"),
        "Final_Human_Rights_Abuse": ("Adj_V14_Human_Rights_Abuse", "Human_Rights_Abuse"),
        "Final_Anti_or_Neocolonialism": ("Adj_V15_Anti_or_Neocolonialism", "Anti_or_Neocolonialism"),
        "Final_Western_Failure": ("Adj_V16_Western_Failure", "Western_Failure"),
        "Final_Security_Effectiveness": ("Adj_V17_Security_Effectiveness", "Security_Effectiveness"),
        "Final_Economic_Interests": ("Adj_V18_Economic_Interests", "Economic_Interests"),
        "Final_Geopolitical_Rivalry": ("Adj_V19_Geopolitical_Rivalry", "Geopolitical_Rivalry"),
        "Final_Main_Associated_Actor": ("Adj_V20_Main_Associated_Actor", "Main_Associated_Actor"),
        "Final_Dominant_Discourse": ("Adj_V21_Dominant_Discourse", "Dominant_Discourse_Support"),
    }

    for final_col, (adj_col, fallback_col) in final_map.items():
        master[final_col] = master.apply(
            lambda row: coalesce_int(row, adj_col, fallback_col, default=None),
            axis=1
        )

    master["Body_Char_Count"] = master["Body_Postclean"].apply(
        lambda value: len(safe_str(value))
    )

    master["Sentence_Count"] = master.apply(article_sentence_count, axis=1)
    master["Period_Group"] = master["Date"].apply(article_period)

    frame_cols = [
        "Final_Counterterrorism",
        "Final_Sovereignty",
        "Final_Human_Rights_Abuse",
        "Final_Anti_or_Neocolonialism",
        "Final_Western_Failure",
        "Final_Security_Effectiveness",
        "Final_Economic_Interests",
        "Final_Geopolitical_Rivalry",
    ]

    master["Frame_Sum"] = master[frame_cols].apply(
        lambda row: sum(safe_int(value, 0) for value in row),
        axis=1
    )

    def active_frames(row):
        mapping = {
            "Final_Counterterrorism": "Counterterrorism",
            "Final_Sovereignty": "Sovereignty",
            "Final_Human_Rights_Abuse": "Human_Rights_Abuse",
            "Final_Anti_or_Neocolonialism": "Anti_or_Neocolonialism",
            "Final_Western_Failure": "Western_Failure",
            "Final_Security_Effectiveness": "Security_Effectiveness",
            "Final_Economic_Interests": "Economic_Interests",
            "Final_Geopolitical_Rivalry": "Geopolitical_Rivalry",
        }

        active = [
            label
            for column, label in mapping.items()
            if safe_int(row.get(column), 0) == 1
        ]

        return "; ".join(active)

    master["Active_Frames"] = master.apply(active_frames, axis=1)

    master["Used_In_Thesis_CDA"] = master["Article_ID"].isin(
        THESIS_CDA_CASES.keys()
    ).astype(int)

    master["Thesis_CDA_Description"] = master["Article_ID"].map(
        THESIS_CDA_CASES
    ).fillna("")

    return master


# =============================================================================
# ELIGIBILITY
# =============================================================================

def eligibility_note(row):
    reasons = []

    relevance = safe_int(row.get("Final_Relevance"), 0)
    body_chars = safe_int(row.get("Body_Char_Count"), 0)
    sentence_count = safe_int(row.get("Sentence_Count"), 0)

    relevance_note = safe_str(row.get("Relevance_Note")).lower()
    headline = safe_str(row.get("Headline")).lower()
    body = safe_str(row.get("Body_Postclean")).lower()

    if relevance not in ELIGIBLE_RELEVANCE:
        reasons.append("relevance_not_3_or_4")

    if not safe_str(row.get("Headline")):
        reasons.append("missing_headline")

    if not safe_str(row.get("Body_Postclean")):
        reasons.append("missing_body")

    if body_chars < MIN_BODY_CHARS:
        reasons.append("short_body")

    if sentence_count < MIN_SENTENCES:
        reasons.append("short_text")

    if "bulletin-style" in relevance_note or "les titres du" in headline:
        reasons.append("bulletin_style")

    if body.count("👉") >= 3:
        reasons.append("bulletin_markers")

    if safe_int(row.get("Final_Actor_Mention"), 5) == 5:
        reasons.append("actor_unclear")

    return "; ".join(reasons) if reasons else "eligible"


def apply_eligibility(df):
    df = df.copy()
    df["Eligibility_Note"] = df.apply(eligibility_note, axis=1)
    df["Eligible_For_Qualitative_Review"] = (
        df["Eligibility_Note"] == "eligible"
    ).astype(int)
    return df


# =============================================================================
# SELECTION SCORING
# =============================================================================

def score_hard_negative_wagner(row):
    score = 0.0

    if safe_int(row.get("Final_Actor_Mention"), 0) in [1, 3]:
        score += 1.0

    if safe_int(row.get("Final_Human_Rights_Abuse"), 0) == 1:
        score += 4.0

    if safe_int(row.get("Final_Stance"), 0) == 1:
        score += 3.0

    if safe_int(row.get("Final_Legitimation"), 0) == 1:
        score += 3.0

    if safe_int(row.get("Final_Dominant_Discourse"), 0) == 3:
        score += 2.0

    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 2.0

    return score


def score_mercenary_problematisation(row):
    score = 0.0

    if safe_int(row.get("Final_Dominant_Label"), 0) == 1:
        score += 4.0

    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 2.0

    if safe_int(row.get("Final_Stance"), 0) in [1, 2, 5]:
        score += 1.0

    if safe_int(row.get("Final_Human_Rights_Abuse"), 0) == 1:
        score += 1.5

    return score


def score_source_attributed_critical(row):
    score = 0.0

    if safe_int(row.get("source_attributed_flag"), 0) == 1:
        score += 3.0

    if safe_int(row.get("likely_republished_flag"), 0) == 1:
        score += 1.0

    if safe_int(row.get("explicit_external_source_flag"), 0) == 1:
        score += 1.0

    if safe_int(row.get("Final_Human_Rights_Abuse"), 0) == 1:
        score += 2.0

    if safe_int(row.get("Final_Stance"), 0) == 1:
        score += 2.0

    if safe_int(row.get("Final_Legitimation"), 0) == 1:
        score += 2.0

    if safe_int(row.get("Final_Dominant_Label"), 0) == 1:
        score += 1.0

    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 1.0

    return score


def score_anti_french_displacement(row):
    score = 0.0

    if safe_int(row.get("Final_Main_Associated_Actor"), 0) == 3:
        score += 3.0

    if safe_int(row.get("Final_Western_Failure"), 0) == 1:
        score += 2.5

    if safe_int(row.get("Final_Anti_or_Neocolonialism"), 0) == 1:
        score += 2.5

    if safe_int(row.get("Final_Sovereignty"), 0) == 1:
        score += 1.0

    if safe_int(row.get("Final_Dominant_Discourse"), 0) in [1, 4]:
        score += 1.5

    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 1.0

    return score


def score_russia_source_not_positive(row):
    score = 0.0

    if safe_int(row.get("Russian_Source_Present"), 0) == 1:
        score += 3.0

    if safe_int(row.get("Russia_Attributed_Claim_Present"), 0) == 1:
        score += 1.5

    if safe_int(row.get("Final_Stance"), 0) in [1, 2, 5]:
        score += 3.0
    else:
        score -= 4.0

    if safe_int(row.get("Final_Human_Rights_Abuse"), 0) == 1:
        score += 1.5

    if safe_int(row.get("Final_Dominant_Label"), 0) == 1:
        score += 1.0

    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 1.0

    return score


def score_africa_corps_reclassification(row):
    score = 0.0

    if safe_int(row.get("Final_Actor_Mention"), 0) in [2, 3]:
        score += 2.0

    if safe_int(row.get("Final_Successor_Frame"), 0) == 1:
        score += 4.0

    if safe_int(row.get("Final_Security_Effectiveness"), 0) == 1:
        score += 1.5

    if safe_int(row.get("Final_Counterterrorism"), 0) == 1:
        score += 1.5

    if safe_int(row.get("Final_Main_Associated_Actor"), 0) == 2:
        score += 1.5

    if safe_int(row.get("Final_Relevance"), 0) == 4:
        score += 1.0

    return score


def score_discursive_richness(row):
    score = 0.0

    sentence_count = safe_int(row.get("Sentence_Count"), 0)
    body_chars = safe_int(row.get("Body_Char_Count"), 0)
    frame_count = safe_int(row.get("Frame_Sum"), 0)

    if sentence_count >= 30:
        score += 3.0
    elif sentence_count >= 20:
        score += 2.0
    elif sentence_count >= MIN_SENTENCES:
        score += 1.0

    if body_chars >= 5000:
        score += 2.0
    elif body_chars >= 2000:
        score += 1.0

    if frame_count >= 3:
        score += 2.0
    elif frame_count >= 2:
        score += 1.0

    if safe_int(row.get("Final_Main_Associated_Actor"), 9) not in [9, 10]:
        score += 0.5

    if safe_int(row.get("Final_Dominant_Discourse"), 5) in [1, 2, 3, 4]:
        score += 0.5

    return score


def redundancy_penalty(row):
    penalty = 0.0

    # Republishing is analytically meaningful and therefore only mildly penalized.
    if safe_int(row.get("near_duplicate_flag"), 0) == 1:
        penalty += 1.5

    if safe_int(row.get("near_duplicate_cross_outlet_flag"), 0) == 1:
        penalty += 0.5

    if safe_int(row.get("likely_republished_flag"), 0) == 1:
        penalty += 0.5

    return penalty


def categorise_row(row):
    categories = []

    if row["Score_Hard_Negative_Wagner"] >= 5:
        categories.append("hard_negative_wagner")

    if row["Score_Mercenary_Problematisation"] >= 4:
        categories.append("mercenary_problematisation")

    if row["Score_Source_Attributed_Critical"] >= 5:
        categories.append("source_attributed_critical")

    if row["Score_Anti_French_Displacement"] >= 4:
        categories.append("anti_french_displacement")

    if row["Score_Russia_Source_Not_Positive"] >= 5:
        categories.append("russia_source_not_positive")

    if row["Score_Africa_Corps_Reclassification"] >= 4:
        categories.append("africa_corps_reclassification")

    return "; ".join(categories)


def primary_category(row):
    category_scores = {
        "hard_negative_wagner": row["Score_Hard_Negative_Wagner"],
        "mercenary_problematisation": row["Score_Mercenary_Problematisation"],
        "source_attributed_critical": row["Score_Source_Attributed_Critical"],
        "anti_french_displacement": row["Score_Anti_French_Displacement"],
        "russia_source_not_positive": row["Score_Russia_Source_Not_Positive"],
        "africa_corps_reclassification": row["Score_Africa_Corps_Reclassification"],
    }

    best_category, best_score = max(
        category_scores.items(),
        key=lambda item: item[1]
    )

    if best_score < 4:
        return "general_discursive_case"

    return best_category


def score_candidates(df):
    df = df.copy()

    df["Score_Hard_Negative_Wagner"] = df.apply(
        score_hard_negative_wagner,
        axis=1
    )

    df["Score_Mercenary_Problematisation"] = df.apply(
        score_mercenary_problematisation,
        axis=1
    )

    df["Score_Source_Attributed_Critical"] = df.apply(
        score_source_attributed_critical,
        axis=1
    )

    df["Score_Anti_French_Displacement"] = df.apply(
        score_anti_french_displacement,
        axis=1
    )

    df["Score_Russia_Source_Not_Positive"] = df.apply(
        score_russia_source_not_positive,
        axis=1
    )

    df["Score_Africa_Corps_Reclassification"] = df.apply(
        score_africa_corps_reclassification,
        axis=1
    )

    df["Discursive_Richness_Score"] = df.apply(
        score_discursive_richness,
        axis=1
    )

    df["Redundancy_Penalty"] = df.apply(
        redundancy_penalty,
        axis=1
    )

    category_score_cols = [
        "Score_Hard_Negative_Wagner",
        "Score_Mercenary_Problematisation",
        "Score_Source_Attributed_Critical",
        "Score_Anti_French_Displacement",
        "Score_Russia_Source_Not_Positive",
        "Score_Africa_Corps_Reclassification",
    ]

    df["Max_Category_Score"] = df[category_score_cols].max(axis=1)

    df["Relevance_Bonus"] = df["Final_Relevance"].apply(
        lambda value: 2.0 if safe_int(value, 0) == 4 else 0.75
    )

    df["Candidate_Categories"] = df.apply(categorise_row, axis=1)
    df["Primary_Category"] = df.apply(primary_category, axis=1)

    df["Qualitative_Review_Score"] = (
        df["Max_Category_Score"]
        + df["Discursive_Richness_Score"]
        + df["Relevance_Bonus"]
        - df["Redundancy_Penalty"]
    ).round(3)

    return df


# =============================================================================
# HUMAN-READABLE LABELS
# =============================================================================

def add_labels(df):
    df = df.copy()

    df["Actor_Label"] = df["Final_Actor_Mention"].apply(
        lambda value: ACTOR_LABELS.get(safe_int(value, 0), "Missing")
    )

    df["Stance_Label"] = df["Final_Stance"].apply(
        lambda value: STANCE_LABELS.get(safe_int(value, 0), "Missing")
    )

    df["Legitimation_Label"] = df["Final_Legitimation"].apply(
        lambda value: LEGITIMATION_LABELS.get(safe_int(value, 0), "Missing")
    )

    df["Dominant_Label_Text"] = df["Final_Dominant_Label"].apply(
        lambda value: DOMINANT_LABELS.get(safe_int(value, 0), "Missing")
    )

    df["Associated_Actor_Label"] = df["Final_Main_Associated_Actor"].apply(
        lambda value: ASSOCIATED_ACTOR_LABELS.get(safe_int(value, 0), "Missing")
    )

    df["Dominant_Discourse_Label"] = df["Final_Dominant_Discourse"].apply(
        lambda value: DISCOURSE_LABELS.get(safe_int(value, 0), "Missing")
    )

    return df


# =============================================================================
# SAMPLE ASSEMBLY
# =============================================================================

def choose_category_cases(pool, category, target_count, selected_ids, outlet_counts):
    category_score_map = {
        "hard_negative_wagner": "Score_Hard_Negative_Wagner",
        "mercenary_problematisation": "Score_Mercenary_Problematisation",
        "source_attributed_critical": "Score_Source_Attributed_Critical",
        "anti_french_displacement": "Score_Anti_French_Displacement",
        "russia_source_not_positive": "Score_Russia_Source_Not_Positive",
        "africa_corps_reclassification": "Score_Africa_Corps_Reclassification",
    }

    score_col = category_score_map[category]

    candidates = pool[
        pool["Candidate_Categories"].fillna("").str.contains(
            category,
            regex=False
        )
    ].copy()

    candidates = candidates.sort_values(
        [score_col, "Qualitative_Review_Score", "Discursive_Richness_Score"],
        ascending=[False, False, False]
    )

    selected_rows = []

    for _, row in candidates.iterrows():
        article_id = safe_str(row["Article_ID"])
        outlet = safe_str(row.get("Outlet"))

        if article_id in selected_ids:
            continue

        if outlet_counts.get(outlet, 0) >= MAX_PER_OUTLET:
            continue

        selected_ids.add(article_id)
        outlet_counts[outlet] = outlet_counts.get(outlet, 0) + 1

        chosen = row.copy()
        chosen["Selection_Rationale"] = category
        selected_rows.append(chosen)

        if len(selected_rows) >= target_count:
            break

    return selected_rows


def select_new_candidates(df_ranked):
    pool = df_ranked[
        (df_ranked["Eligible_For_Qualitative_Review"] == 1)
        & (df_ranked["Used_In_Thesis_CDA"] == 0)
    ].copy()

    selected_ids = set()
    outlet_counts = {}
    selected_rows = []

    for category, target_count in CATEGORY_TARGETS.items():
        category_rows = choose_category_cases(
            pool=pool,
            category=category,
            target_count=target_count,
            selected_ids=selected_ids,
            outlet_counts=outlet_counts
        )

        selected_rows.extend(category_rows)

    selected = (
        pd.DataFrame(selected_rows)
        if selected_rows
        else pd.DataFrame(columns=pool.columns.tolist() + ["Selection_Rationale"])
    )

    # Fill remaining slots using high-scoring cases not yet selected.
    if len(selected) < TARGET_NEW_CASES:
        current_ids = set(selected["Article_ID"].tolist()) if not selected.empty else set()
        current_outlets = selected["Outlet"].value_counts().to_dict() if not selected.empty else {}

        remaining = pool.sort_values(
            ["Qualitative_Review_Score", "Discursive_Richness_Score"],
            ascending=[False, False]
        )

        filler_rows = []

        for _, row in remaining.iterrows():
            if len(selected) + len(filler_rows) >= TARGET_NEW_CASES:
                break

            article_id = safe_str(row["Article_ID"])
            outlet = safe_str(row.get("Outlet"))

            if article_id in current_ids:
                continue

            if current_outlets.get(outlet, 0) >= MAX_PER_OUTLET:
                continue

            chosen = row.copy()
            chosen["Selection_Rationale"] = "general_high_value_case"

            filler_rows.append(chosen)
            current_ids.add(article_id)
            current_outlets[outlet] = current_outlets.get(outlet, 0) + 1

        if filler_rows:
            selected = pd.concat(
                [selected, pd.DataFrame(filler_rows)],
                ignore_index=True
            )

    if selected.empty:
        return selected

    selected = selected.drop_duplicates(subset=["Article_ID"]).copy()

    selected = selected.sort_values(
        ["Qualitative_Review_Score", "Discursive_Richness_Score"],
        ascending=[False, False]
    ).head(TARGET_NEW_CASES).copy()

    selected["Review_Corpus_Rank"] = range(1, len(selected) + 1)

    return selected


def build_selected_review_corpus(df_ranked, df_new):
    thesis_cases = df_ranked[
        (df_ranked["Eligible_For_Qualitative_Review"] == 1)
        & (df_ranked["Used_In_Thesis_CDA"] == 1)
    ].copy()

    if not thesis_cases.empty:
        thesis_cases["Selection_Rationale"] = "existing_thesis_case"
        thesis_cases["Review_Corpus_Rank"] = np.nan
        thesis_cases["Review_Corpus_Group"] = "Existing thesis CDA case"

    if not df_new.empty:
        df_new = df_new.copy()
        df_new["Review_Corpus_Group"] = "New article review candidate"

    selected = pd.concat(
        [thesis_cases, df_new],
        ignore_index=True
    )

    if selected.empty:
        return selected

    # Existing thesis cases first; then new cases ordered by rank.
    selected["_sort_group"] = np.where(
        selected["Used_In_Thesis_CDA"] == 1,
        0,
        1
    )

    selected = selected.sort_values(
        ["_sort_group", "Review_Corpus_Rank", "Qualitative_Review_Score"],
        ascending=[True, True, False]
    ).drop(columns="_sort_group")

    selected["Researcher_Keep"] = ""
    selected["Researcher_Final_Category"] = ""
    selected["Researcher_Notes"] = ""

    return selected


# =============================================================================
# TEXT CORPUS EXPORT
# =============================================================================

def write_article_block(row, order_number):
    article_id = safe_str(row.get("Article_ID"))
    title = safe_str(row.get("Headline"))
    outlet = safe_str(row.get("Outlet"))
    date = safe_str(row.get("Date"))
    url = safe_str(row.get("URL"))
    lead = safe_str(row.get("Lead"))
    body = safe_str(row.get("Body_Postclean"))

    selection_group = safe_str(row.get("Review_Corpus_Group"))
    selection_rationale = safe_str(row.get("Selection_Rationale"))

    metadata = [
        f"REVIEW CORPUS ITEM: {order_number}",
        f"ARTICLE ID: {article_id}",
        f"OUTLET: {outlet}",
        f"PUBLICATION DATE: {date}",
        f"URL: {url}",
        f"CORPUS GROUP: {selection_group}",
        f"SELECTION RATIONALE: {selection_rationale}",
        "",
        "CODING CONTEXT",
        f"Relevance: {safe_int(row.get('Final_Relevance'), '')}",
        f"Actor mention: {safe_str(row.get('Actor_Label'))}",
        f"Successor framing: {'Yes' if safe_int(row.get('Final_Successor_Frame'), 0) == 1 else 'No'}",
        f"Dominant label: {safe_str(row.get('Dominant_Label_Text'))}",
        f"Stance: {safe_str(row.get('Stance_Label'))}",
        f"Legitimation: {safe_str(row.get('Legitimation_Label'))}",
        f"Main associated actor: {safe_str(row.get('Associated_Actor_Label'))}",
        f"Dominant discourse: {safe_str(row.get('Dominant_Discourse_Label'))}",
        f"Active frames: {safe_str(row.get('Active_Frames')) or 'None'}",
        f"Period group: {safe_str(row.get('Period_Group'))}",
        "",
        "SOURCE AND CIRCULATION CONTEXT",
        f"Source attributed: {'Yes' if safe_int(row.get('source_attributed_flag'), 0) == 1 else 'No'}",
        f"Likely republished: {'Yes' if safe_int(row.get('likely_republished_flag'), 0) == 1 else 'No'}",
        f"Near duplicate: {'Yes' if safe_int(row.get('near_duplicate_flag'), 0) == 1 else 'No'}",
        f"Russian-source signal: {'Yes' if safe_int(row.get('Russian_Source_Present'), 0) == 1 else 'No'}",
        f"Russian-source type: {safe_str(row.get('Russian_Source_Type')) or 'None'}",
        f"Russian-source names: {safe_str(row.get('Russian_Source_Names')) or 'None'}",
        "",
        "ARTICLE TITLE",
        title,
        "",
        "LEAD / PEREX",
        lead if lead else "[No separately recovered lead available.]",
        "",
        "ARTICLE TEXT",
        body if body else "[No recovered body text available.]",
    ]

    return "\n".join(metadata)


def write_review_corpus_txt(df_selected):
    lines = [
        "QUALITATIVE REVIEW CORPUS",
        "=" * 100,
        "Article-level material selected for close reading and discourse interpretation.",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Number of included texts: {len(df_selected)}",
        "",
        "The corpus contains existing reference cases from the thesis and newly selected",
        "article candidates. Metadata and coding context are included to support transparent",
        "comparative reading. Final analytical interpretation remains subject to researcher review.",
        "",
    ]

    for index, (_, row) in enumerate(df_selected.iterrows(), start=1):
        lines.append("=" * 100)
        lines.append(write_article_block(row, index))
        lines.append("")

    with open(OUTPUT_SELECTED_TXT, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# =============================================================================
# SUMMARY EXPORT
# =============================================================================

def write_summary(df_all, df_ranked, df_selected_new, df_selected_all):
    eligible = df_ranked[
        df_ranked["Eligible_For_Qualitative_Review"] == 1
    ].copy()

    thesis_eligible = eligible[
        eligible["Used_In_Thesis_CDA"] == 1
    ].copy()

    lines = [
        "STEPE ARTICLE QUALITATIVE REVIEW CORPUS SUMMARY",
        "=" * 78,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Purpose:",
        "To assemble a transparent qualitative review corpus for close reading",
        "and discourse interpretation in the article on Wagner Group, Africa Corps,",
        "and the limits of straightforward pro-Russian framing in Malian online news.",
        "",
        "Corpus overview:",
        f"- Total merged articles: {len(df_all)}",
        f"- Eligible relevance 3/4 texts: {len(eligible)}",
        f"- Existing eligible thesis CDA cases: {len(thesis_eligible)}",
        f"- Newly selected review candidates: {len(df_selected_new)}",
        f"- Total review corpus texts: {len(df_selected_all)}",
        "",
        "New candidate categories:",
    ]

    if not df_selected_new.empty:
        category_counts = (
            df_selected_new["Selection_Rationale"]
            .value_counts(dropna=False)
            .to_dict()
        )

        for category, count in category_counts.items():
            lines.append(f"- {category}: {count}")

        lines.extend([
            "",
            "New candidate overview:",
        ])

        for _, row in df_selected_new.iterrows():
            lines.append(
                f"- {safe_str(row.get('Article_ID'))} | "
                f"{safe_str(row.get('Outlet'))} | "
                f"{safe_str(row.get('Date'))} | "
                f"{safe_str(row.get('Headline'))} | "
                f"{safe_str(row.get('Selection_Rationale'))} | "
                f"score={safe_float(row.get('Qualitative_Review_Score')):.2f}"
            )
    else:
        lines.append("- No new candidates selected.")

    lines.extend([
        "",
        "Interpretive caution:",
        "- The selected corpus is a purposive review set, not a statistically representative sample.",
        "- Source attribution and republication are retained because they are substantively relevant.",
        "- Russian-source presence is not treated as equivalent to editorial endorsement.",
        "- Final inclusion in the article and final discourse interpretation remain researcher decisions.",
        "",
        "Outputs:",
        f"- {OUTPUT_ALL.name}",
        f"- {OUTPUT_RANKED.name}",
        f"- {OUTPUT_SELECTED_CSV.name}",
        f"- {OUTPUT_SELECTED_TXT.name}",
        f"- {OUTPUT_THESIS_CASES.name}",
    ])

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# =============================================================================
# MAIN EXPORT
# =============================================================================

def export_outputs(df_all, df_ranked, df_selected_new, df_selected_all):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_columns = [
        "Article_ID", "Outlet", "Date", "Headline", "URL",
        "Final_Relevance", "Actor_Label", "Final_Successor_Frame",
        "Dominant_Label_Text", "Stance_Label", "Legitimation_Label",
        "Associated_Actor_Label", "Dominant_Discourse_Label",
        "Period_Group", "Active_Frames", "Frame_Sum",
        "Body_Char_Count", "Sentence_Count",
        "Eligible_For_Qualitative_Review", "Eligibility_Note",
        "Used_In_Thesis_CDA", "Thesis_CDA_Description",
        "source_attributed_flag", "likely_republished_flag",
        "near_duplicate_flag", "near_duplicate_cross_outlet_flag",
        "Russian_Source_Present", "Russian_Source_Type",
        "Russia_Attributed_Claim_Present",
        "Candidate_Categories", "Primary_Category",
        "Score_Hard_Negative_Wagner",
        "Score_Mercenary_Problematisation",
        "Score_Source_Attributed_Critical",
        "Score_Anti_French_Displacement",
        "Score_Russia_Source_Not_Positive",
        "Score_Africa_Corps_Reclassification",
        "Discursive_Richness_Score",
        "Redundancy_Penalty",
        "Qualitative_Review_Score",
    ]

    all_columns = [column for column in all_columns if column in df_all.columns]
    save_csv(df_all[all_columns], OUTPUT_ALL)

    ranked_columns = [
        "Article_ID", "Outlet", "Date", "Headline", "URL",
        "Final_Relevance", "Actor_Label", "Final_Successor_Frame",
        "Dominant_Label_Text", "Stance_Label", "Legitimation_Label",
        "Associated_Actor_Label", "Dominant_Discourse_Label",
        "Period_Group", "Active_Frames",
        "Candidate_Categories", "Primary_Category",
        "Used_In_Thesis_CDA", "Thesis_CDA_Description",
        "source_attributed_flag", "likely_republished_flag",
        "near_duplicate_flag",
        "Russian_Source_Present", "Russian_Source_Type",
        "Russia_Attributed_Claim_Present",
        "Score_Hard_Negative_Wagner",
        "Score_Mercenary_Problematisation",
        "Score_Source_Attributed_Critical",
        "Score_Anti_French_Displacement",
        "Score_Russia_Source_Not_Positive",
        "Score_Africa_Corps_Reclassification",
        "Discursive_Richness_Score",
        "Redundancy_Penalty",
        "Qualitative_Review_Score",
    ]

    ranked_columns = [column for column in ranked_columns if column in df_ranked.columns]
    save_csv(df_ranked[ranked_columns], OUTPUT_RANKED)

    # Existing thesis cases reference file
    thesis_cases = df_ranked[
        df_ranked["Used_In_Thesis_CDA"] == 1
    ].copy()

    thesis_columns = [
        "Article_ID", "Outlet", "Date", "Headline", "URL",
        "Thesis_CDA_Description", "Final_Relevance", "Actor_Label",
        "Candidate_Categories", "Primary_Category",
        "Qualitative_Review_Score",
    ]

    thesis_columns = [column for column in thesis_columns if column in thesis_cases.columns]
    save_csv(thesis_cases[thesis_columns], OUTPUT_THESIS_CASES)

    # Full selected review corpus CSV
    selected_columns = [
        "Review_Corpus_Group", "Review_Corpus_Rank", "Selection_Rationale",
        "Article_ID", "Outlet", "Date", "Headline", "Lead", "Body_Postclean", "URL",
        "Final_Relevance", "Actor_Label", "Final_Successor_Frame",
        "Dominant_Label_Text", "Stance_Label", "Legitimation_Label",
        "Associated_Actor_Label", "Dominant_Discourse_Label",
        "Period_Group", "Active_Frames",
        "Used_In_Thesis_CDA", "Thesis_CDA_Description",
        "source_attributed_flag", "likely_republished_flag",
        "likely_republished_basis", "republication_confidence",
        "near_duplicate_flag", "near_duplicate_cluster_id",
        "near_duplicate_top_match_id", "near_duplicate_top_match_outlet",
        "Russian_Source_Present", "Russian_Source_Type",
        "Russian_Source_Names", "Russian_Source_Evidence",
        "Russia_Attributed_Claim_Present",
        "Candidate_Categories", "Primary_Category",
        "Qualitative_Review_Score", "Discursive_Richness_Score",
        "Researcher_Keep", "Researcher_Final_Category", "Researcher_Notes",
    ]

    selected_columns = [
        column
        for column in selected_columns
        if column in df_selected_all.columns
    ]

    save_csv(df_selected_all[selected_columns], OUTPUT_SELECTED_CSV)
    write_review_corpus_txt(df_selected_all)

    write_summary(
        df_all=df_all,
        df_ranked=df_ranked,
        df_selected_new=df_selected_new,
        df_selected_all=df_selected_all
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("STEPE: ARTICLE QUALITATIVE REVIEW CORPUS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading corpus and coding inputs...")
    consolidated, adjudicated = load_required_inputs()
    stepb = load_stepb_optional()
    stepba = load_stepba_optional()

    print(f"Consolidated corpus rows: {len(consolidated)}")
    print(f"Adjudicated coding rows: {len(adjudicated)}")
    print(f"StepB source layer available: {'yes' if stepb is not None else 'no'}")
    print(f"StepB.A source layer available: {'yes' if stepba is not None else 'no'}")

    print("\nPreparing integrated review dataset...")
    df_all = prepare_master(consolidated, adjudicated, stepb, stepba)
    df_all = apply_eligibility(df_all)
    df_all = score_candidates(df_all)
    df_all = add_labels(df_all)

    df_ranked = df_all[
        df_all["Eligible_For_Qualitative_Review"] == 1
    ].copy()

    df_ranked = df_ranked.sort_values(
        ["Qualitative_Review_Score", "Discursive_Richness_Score"],
        ascending=[False, False]
    ).copy()

    print("Selecting new review candidates...")
    df_selected_new = select_new_candidates(df_ranked)

    print("Combining thesis reference cases and new review candidates...")
    df_selected_all = build_selected_review_corpus(
        df_ranked,
        df_selected_new
    )

    print("Exporting CSV and TXT review corpus...")
    export_outputs(
        df_all=df_all,
        df_ranked=df_ranked,
        df_selected_new=df_selected_new,
        df_selected_all=df_selected_all
    )

    eligible_count = int(df_all["Eligible_For_Qualitative_Review"].sum())
    thesis_count = int(
        (
            (df_all["Eligible_For_Qualitative_Review"] == 1)
            & (df_all["Used_In_Thesis_CDA"] == 1)
        ).sum()
    )

    print("\n=== DIAGNOSTICS ===")
    print(f"Total merged articles: {len(df_all)}")
    print(f"Eligible relevance 3/4 review texts: {eligible_count}")
    print(f"Eligible existing thesis CDA cases: {thesis_count}")
    print(f"New review candidates selected: {len(df_selected_new)}")
    print(f"Total qualitative review corpus: {len(df_selected_all)}")

    if not df_selected_new.empty:
        print("\nNew review candidates:")
        preview_columns = [
            "Review_Corpus_Rank",
            "Selection_Rationale",
            "Article_ID",
            "Outlet",
            "Date",
            "Headline",
            "Qualitative_Review_Score",
        ]

        preview_columns = [
            column
            for column in preview_columns
            if column in df_selected_new.columns
        ]

        print(df_selected_new[preview_columns].to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {OUTPUT_ALL}")
    print(f"- {OUTPUT_RANKED}")
    print(f"- {OUTPUT_SELECTED_CSV}")
    print(f"- {OUTPUT_SELECTED_TXT}")
    print(f"- {OUTPUT_THESIS_CASES}")
    print(f"- {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()