import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    SEABORN_AVAILABLE = False


# =============================================================================
# StepB.A — Russian / Russia-attributed source environment analysis
# =============================================================================
#
# Purpose:
#   Post-process existing StepB output and identify Russian or Russia-attributed
#   source signals. Then connect those signals to final frame coding and generate
#   discussion-ready tables + heatmaps.
#
# Inputs:
#   pilot/data/postStepB.csv
#   pilot/data/postConsolidated.csv
#   pilot/GEMINI/data/final_conservative_adjudicated_table.csv  [preferred if present]
#
# Outputs:
#   pilot/data/stepBA/
#
# Methodological status:
#   Supplementary source-environment enrichment layer.
#   Not a direct measure of editorial alignment or endorsement.
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Inputs
INPUT_STEPB = SCRIPT_DIR / "data" / "postStepB.csv"
INPUT_CONSOLIDATED = SCRIPT_DIR / "data" / "postConsolidated.csv"
INPUT_GEMINI_CONSERVATIVE = (
    SCRIPT_DIR / "GEMINI" / "data" / "final_conservative_adjudicated_table.csv"
)

# Dedicated StepB.A output root
STEPBA_OUTPUT_DIR = SCRIPT_DIR / "data" / "stepBA"
STEPBA_DISCUSSION_DIR = STEPBA_OUTPUT_DIR / "discussion"
STEPBA_FIGURES_DIR = STEPBA_OUTPUT_DIR / "figures"

# Main outputs
OUTPUT_CSV = STEPBA_OUTPUT_DIR / "postStepBA_russian_sources.csv"
OUTPUT_REVIEW_CSV = STEPBA_OUTPUT_DIR / "postStepBA_russian_sources_review.csv"
OUTPUT_SUMMARY_TXT = STEPBA_OUTPUT_DIR / "postStepBA_russian_sources_summary.txt"

# Discussion / figure output dirs
OUTPUT_DISCUSSION_DIR = STEPBA_DISCUSSION_DIR
OUTPUT_FIGURES_DIR = STEPBA_FIGURES_DIR

# Main analytical comparison threshold.
# Use 2 for Relevance 2+, or 3 for Relevance 3+.
RELEVANCE_THRESHOLD = 2

# Optional: number of illustrative examples in export.
TOP_EXAMPLES_N = 100

FRAME_LABELS = {
    "Final_Counterterrorism": "Counterterrorism",
    "Final_Sovereignty": "Sovereignty",
    "Final_Human_Rights_Abuse": "Human rights abuse",
    "Final_Anti_or_Neocolonialism": "Anti-/neocolonialism",
    "Final_Western_Failure": "Western failure",
    "Final_Security_Effectiveness": "Security effectiveness",
    "Final_Economic_Interests": "Economic interests",
    "Final_Geopolitical_Rivalry": "Geopolitical rivalry",
}

FRAME_COLS_FINAL = list(FRAME_LABELS.keys())


# =============================================================================
# HELPERS
# =============================================================================

def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def safe_int(x, default=0):
    if pd.isna(x) or x is None:
        return default
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            if pd.isna(x):
                return default
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
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
    raw = safe_str(x)
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return digits.zfill(6)


def ensure_columns(df, cols, fill_value=None):
    for col in cols:
        if col not in df.columns:
            df[col] = fill_value
    return df


def ensure_dirs():
    STEPBA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STEPBA_DISCUSSION_DIR.mkdir(parents=True, exist_ok=True)
    STEPBA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def percent(n, d):
    return (n / d * 100) if d else 0.0


def count_matches(text, patterns):
    text = safe_str(text)
    total = 0
    matched = []

    if not text:
        return 0, []

    for pat in patterns:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            total += len(found)
            matched.append(pat)

    return total, sorted(set(matched))


def has_any(text, patterns):
    n, _ = count_matches(text, patterns)
    return n > 0


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


# =============================================================================
# TEXT ACCESS
# =============================================================================

def text_for_source_analysis(row):
    """
    Combines original article text + StepB source-related fields.

    Important:
    This does not treat every mention of Russia as source evidence.
    Source evidence is produced through source fields, source names,
    attribution structures, or source-like reporting patterns.
    """
    parts = [
        safe_str(row.get("headline_clean")),
        safe_str(row.get("lead_clean")),
        safe_str(row.get("body_postclean")),
        safe_str(row.get("author_clean")),
        safe_str(row.get("rubrique_clean")),
        safe_str(row.get("url")),
        safe_str(row.get("explicit_external_source_name")),
        safe_str(row.get("author_external_source_name")),
        safe_str(row.get("likely_republished_basis")),
        safe_str(row.get("republication_phrase_note")),
        safe_str(row.get("attribution_phrase_note")),
    ]
    return "\n".join([p for p in parts if p]).strip()


def source_field_text(row):
    parts = [
        safe_str(row.get("explicit_external_source_name")),
        safe_str(row.get("author_external_source_name")),
        safe_str(row.get("likely_republished_basis")),
        safe_str(row.get("explicit_external_source_type")),
        safe_str(row.get("author_external_source_type")),
    ]
    return " ".join([p for p in parts if p]).strip()


# =============================================================================
# RUSSIAN SOURCE LEXICONS — CONSERVATIVE
# =============================================================================

RUSSIAN_STATE_MEDIA_PATTERNS = [
    r"\brt\b",
    r"\brt\s+france\b",
    r"\brussia\s+today\b",
    r"\bsputnik\b",
    r"\bsputnik\s+afrique\b",
    r"\bsputnik\s+africa\b",
    r"\brussia\s+24\b",
    r"\brossiya\s+24\b",
]

RUSSIAN_NEWS_AGENCY_PATTERNS = [
    r"\btass\b",
    r"\bitar[-\s]?tass\b",
    r"\bria\s+novosti\b",
    r"\brussian\s+news\s+agency\b",
    r"\bagence\s+russe\s+tass\b",
    r"\bagence\s+tass\b",
]

RUSSIAN_EMBASSY_PATTERNS = [
    r"\bambassade\s+de\s+russie\b",
    r"\bambassade\s+russe\b",
    r"\bambassadeur\s+russe\b",
    r"\bambassadeur\s+de\s+russie\b",
    r"\bdiplomate\s+russe\b",
    r"\brepr[ée]sentation\s+diplomatique\s+russe\b",
]

RUSSIAN_DEFENSE_PATTERNS = [
    r"\bminist[èe]re\s+russe\s+de\s+la\s+d[ée]fense\b",
    r"\bminist[èe]re\s+de\s+la\s+d[ée]fense\s+russe\b",
    r"\bd[ée]fense\s+russe\b",
    r"\barmée\s+russe\b",
    r"\bforces?\s+arm[ée]es?\s+russes?\b",
]

RUSSIAN_MFA_PATTERNS = [
    r"\bminist[èe]re\s+russe\s+des?\s+affaires?\s+[ée]trang[èe]res\b",
    r"\bminist[èe]re\s+des?\s+affaires?\s+[ée]trang[èe]res\s+russe\b",
    r"\bdiplomatie\s+russe\b",
]

# Official-person source structures.
# Raw mentions of Putin/Lavrov/Peskov are NOT enough; these patterns require reporting verbs.
RUSSIAN_OFFICIAL_ATTRIBUTION_PATTERNS = [
    r"\b(?:vladimir\s+)?poutine\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
    r"\b(?:sergue[iï]|sergey)\s+lavrov\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
    r"\bdmitri\s+peskov\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
    r"\ble\s+kremlin\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
    r"\bselon\s+le\s+kremlin\b",
    r"\bd['’]apr[eè]s\s+le\s+kremlin\b",
]

# Russia-attributed claims.
# These indicate claims attributed to Russia/Moscow, not editorial endorsement.
RUSSIA_ATTRIBUTION_PATTERNS = [
    r"\bselon\s+(?:la\s+)?russie\b",
    r"\bselon\s+moscou\b",
    r"\bd['’]apr[eè]s\s+(?:la\s+)?russie\b",
    r"\bd['’]apr[eè]s\s+moscou\b",
    r"\bla\s+russie\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
    r"\bmoscou\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
    r"\bles\s+autorités?\s+russes?\s+ont\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
    r"\ble\s+gouvernement\s+russe\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
]

PRO_RUSSIAN_PROXY_PATTERNS = [
    r"\bm[ée]dias?\s+pro[-\s]?russes?\b",
    r"\bcanaux?\s+pro[-\s]?russes?\b",
    r"\bcomptes?\s+pro[-\s]?russes?\b",
    r"\bpropagande\s+russe\b",
    r"\brelais\s+pro[-\s]?russes?\b",
    r"\br[ée]seaux?\s+pro[-\s]?russes?\b",
]

SOURCE_CUE_PATTERNS = [
    r"\bselon\b",
    r"\bd['’]apr[eè]s\b",
    r"\bsource\s*:",
    r"\bcit[ée]?\s+par\b",
    r"\ba\s+d[ée]clar[ée]\b",
    r"\ba\s+indiqu[ée]\b",
    r"\ba\s+affirm[ée]\b",
    r"\ba\s+annonc[ée]\b",
    r"\ba\s+rapport[ée]\b",
    r"\brapporte\b",
    r"\bcommuniqu[ée]\b",
]

# Non-Russian / Western / international source signals for dominance comparison.
WESTERN_OR_NONRUSSIAN_SOURCE_PATTERNS = [
    r"\bafp\b",
    r"\breuters\b",
    r"\bassociated\s+press\b",
    r"\banadolu\b",
    r"\brfi\b",
    r"\bfrance\s*24\b",
    r"\bbbc\b",
    r"\ble\s+monde\b",
    r"\bjeune\s+afrique\b",
    r"\btv5\s*monde\b",
    r"\bonu\b",
    r"\bminusma\b",
    r"\bhrw\b",
    r"\bhuman\s+rights\s+watch\b",
    r"\bamnesty\b",
    r"\bunion\s+europ[ée]enne\b",
    r"\bue\b",
    r"\b[ée]tats?-unis\b",
    r"\betats?-unis\b",
    r"\bfrance\b",
]

ALL_RUSSIAN_SOURCE_PATTERNS = (
    RUSSIAN_STATE_MEDIA_PATTERNS
    + RUSSIAN_NEWS_AGENCY_PATTERNS
    + RUSSIAN_EMBASSY_PATTERNS
    + RUSSIAN_DEFENSE_PATTERNS
    + RUSSIAN_MFA_PATTERNS
    + RUSSIAN_OFFICIAL_ATTRIBUTION_PATTERNS
    + RUSSIA_ATTRIBUTION_PATTERNS
    + PRO_RUSSIAN_PROXY_PATTERNS
)


# =============================================================================
# CLASSIFICATION HELPERS
# =============================================================================

def classify_russian_source_type(row, text):
    types = []

    if has_any(text, RUSSIAN_STATE_MEDIA_PATTERNS):
        types.append("russian_state_media")

    if has_any(text, RUSSIAN_NEWS_AGENCY_PATTERNS):
        types.append("russian_news_agency")

    if has_any(text, RUSSIAN_EMBASSY_PATTERNS):
        types.append("russian_embassy")

    if has_any(text, RUSSIAN_DEFENSE_PATTERNS):
        types.append("russian_military_defense")

    if has_any(text, RUSSIAN_MFA_PATTERNS):
        types.append("russian_mfa")

    if has_any(text, RUSSIAN_OFFICIAL_ATTRIBUTION_PATTERNS):
        types.append("russian_state_official")

    if has_any(text, RUSSIA_ATTRIBUTION_PATTERNS):
        types.append("russia_attributed_claim")

    if has_any(text, PRO_RUSSIAN_PROXY_PATTERNS):
        types.append("pro_russian_proxy")

    # Strong StepB source-field override.
    src_fields = source_field_text(row)

    if has_any(src_fields, RUSSIAN_STATE_MEDIA_PATTERNS):
        types.append("russian_state_media")

    if has_any(src_fields, RUSSIAN_NEWS_AGENCY_PATTERNS):
        types.append("russian_news_agency")

    if has_any(src_fields, RUSSIAN_EMBASSY_PATTERNS):
        types.append("russian_embassy")

    if has_any(src_fields, RUSSIAN_DEFENSE_PATTERNS):
        types.append("russian_military_defense")

    if has_any(src_fields, RUSSIAN_MFA_PATTERNS):
        types.append("russian_mfa")

    types = sorted(set(types))

    if not types:
        return "none"

    if len(types) == 1:
        return types[0]

    return "mixed_russian_source"


def extract_russian_source_names(row, text):
    found = []

    src_fields = source_field_text(row)
    combined = f"{src_fields}\n{text}"

    name_patterns = {
        "RT": [
            r"\brt\b",
            r"\brt\s+france\b",
            r"\brussia\s+today\b",
        ],
        "Sputnik": [
            r"\bsputnik\b",
            r"\bsputnik\s+afrique\b",
            r"\bsputnik\s+africa\b",
        ],
        "TASS": [
            r"\btass\b",
            r"\bitar[-\s]?tass\b",
        ],
        "RIA Novosti": [
            r"\bria\s+novosti\b",
        ],
        "Kremlin": [
            r"\bkremlin\b",
        ],
        "Russian MFA": [
            r"\bminist[èe]re\s+russe\s+des?\s+affaires?\s+[ée]trang[èe]res\b",
            r"\bdiplomatie\s+russe\b",
            r"\blavrov\b",
        ],
        "Russian Ministry of Defence": [
            r"\bminist[èe]re\s+russe\s+de\s+la\s+d[ée]fense\b",
            r"\bminist[èe]re\s+de\s+la\s+d[ée]fense\s+russe\b",
            r"\bd[ée]fense\s+russe\b",
        ],
        "Russian Embassy": [
            r"\bambassade\s+de\s+russie\b",
            r"\bambassade\s+russe\b",
        ],
        "Russia / Moscow attribution": [
            r"\bselon\s+(?:la\s+)?russie\b",
            r"\bselon\s+moscou\b",
            r"\bd['’]apr[eè]s\s+(?:la\s+)?russie\b",
            r"\bd['’]apr[eè]s\s+moscou\b",
            r"\bmoscou\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
            r"\bla\s+russie\s+a\s+(?:affirm[ée]|d[ée]clar[ée]|indiqu[ée]|annonc[ée]|accus[ée]|d[ée]menti|soutenu|estim[ée])\b",
        ],
        "Pro-Russian proxy": [
            r"\bm[ée]dias?\s+pro[-\s]?russes?\b",
            r"\bcanaux?\s+pro[-\s]?russes?\b",
            r"\bcomptes?\s+pro[-\s]?russes?\b",
            r"\brelais\s+pro[-\s]?russes?\b",
        ],
    }

    for name, patterns in name_patterns.items():
        if has_any(combined, patterns):
            found.append(name)

    return "; ".join(sorted(set(found)))


# =============================================================================
# SCORING
# =============================================================================

def compute_russian_source_scores(row):
    text = text_for_source_analysis(row)
    src_fields = source_field_text(row)

    score = 0.0
    attribution_score = 0.0
    evidence = []

    # -------------------------------------------------------------------------
    # Strong StepB source-field evidence
    # -------------------------------------------------------------------------
    if has_any(src_fields, RUSSIAN_STATE_MEDIA_PATTERNS):
        score += 6
        evidence.append("Russian state media signal in StepB source fields")

    if has_any(src_fields, RUSSIAN_NEWS_AGENCY_PATTERNS):
        score += 6
        evidence.append("Russian news agency signal in StepB source fields")

    if has_any(src_fields, RUSSIAN_EMBASSY_PATTERNS):
        score += 6
        evidence.append("Russian embassy signal in StepB source fields")

    if has_any(src_fields, RUSSIAN_DEFENSE_PATTERNS):
        score += 5
        evidence.append("Russian defense source signal in StepB source fields")

    if has_any(src_fields, RUSSIAN_MFA_PATTERNS):
        score += 5
        evidence.append("Russian MFA source signal in StepB source fields")

    # -------------------------------------------------------------------------
    # Textual explicit source evidence
    # -------------------------------------------------------------------------
    n_media, _ = count_matches(text, RUSSIAN_STATE_MEDIA_PATTERNS)
    n_agency, _ = count_matches(text, RUSSIAN_NEWS_AGENCY_PATTERNS)
    n_embassy, _ = count_matches(text, RUSSIAN_EMBASSY_PATTERNS)
    n_defense, _ = count_matches(text, RUSSIAN_DEFENSE_PATTERNS)
    n_mfa, _ = count_matches(text, RUSSIAN_MFA_PATTERNS)
    n_official_attr, _ = count_matches(text, RUSSIAN_OFFICIAL_ATTRIBUTION_PATTERNS)
    n_russia_attr, _ = count_matches(text, RUSSIA_ATTRIBUTION_PATTERNS)
    n_proxy, _ = count_matches(text, PRO_RUSSIAN_PROXY_PATTERNS)

    if n_media > 0:
        score += min(n_media, 3) * 3.0
        evidence.append(f"Russian state media textual signal n={n_media}")

    if n_agency > 0:
        score += min(n_agency, 3) * 3.0
        evidence.append(f"Russian news agency textual signal n={n_agency}")

    if n_embassy > 0:
        score += min(n_embassy, 3) * 3.0
        evidence.append(f"Russian embassy textual signal n={n_embassy}")

    if n_defense > 0:
        score += min(n_defense, 3) * 2.5
        evidence.append(f"Russian defense textual signal n={n_defense}")

    if n_mfa > 0:
        score += min(n_mfa, 3) * 2.5
        evidence.append(f"Russian MFA textual signal n={n_mfa}")

    if n_official_attr > 0:
        score += min(n_official_attr, 4) * 2.5
        attribution_score += min(n_official_attr, 4) * 2.0
        evidence.append(f"Russian official attribution pattern n={n_official_attr}")

    if n_russia_attr > 0:
        score += min(n_russia_attr, 5) * 2.0
        attribution_score += min(n_russia_attr, 5) * 2.0
        evidence.append(f"Russia/Moscow-attributed claim pattern n={n_russia_attr}")

    if n_proxy > 0:
        score += min(n_proxy, 3) * 2.0
        evidence.append(f"Pro-Russian proxy signal n={n_proxy}")

    # -------------------------------------------------------------------------
    # StepB prior source-attribution boost, only if there is already RU signal
    # -------------------------------------------------------------------------
    if score > 0 and safe_int(row.get("source_attributed_flag")) == 1:
        score += 1
        evidence.append("StepB source_attributed_flag=1")

    if score > 0 and safe_int(row.get("likely_republished_flag")) == 1:
        score += 1
        evidence.append("StepB likely_republished_flag=1")

    return {
        "Russian_Source_Score": round(score, 3),
        "Russian_Attribution_Score": round(attribution_score, 3),
        "Russian_Source_Evidence": "; ".join(evidence),
    }


def compute_nonrussian_source_score(row):
    text = text_for_source_analysis(row)
    src_fields = source_field_text(row)

    score = 0.0

    if has_any(src_fields, WESTERN_OR_NONRUSSIAN_SOURCE_PATTERNS):
        score += 4.0

    n, _ = count_matches(text, WESTERN_OR_NONRUSSIAN_SOURCE_PATTERNS)
    score += min(n, 6) * 1.0

    if safe_int(row.get("source_attributed_flag")) == 1 and score > 0:
        score += 1.0

    return round(score, 3)


# =============================================================================
# MAIN ROW CODING
# =============================================================================

def code_russian_source(row):
    text = text_for_source_analysis(row)
    src_fields = source_field_text(row)

    scores = compute_russian_source_scores(row)
    russian_score = scores["Russian_Source_Score"]
    attribution_score = scores["Russian_Attribution_Score"]
    nonrus_score = compute_nonrussian_source_score(row)

    source_type = classify_russian_source_type(row, text)
    source_names = extract_russian_source_names(row, text)

    state_media_present = 1 if (
        has_any(text, RUSSIAN_STATE_MEDIA_PATTERNS + RUSSIAN_NEWS_AGENCY_PATTERNS) or
        has_any(src_fields, RUSSIAN_STATE_MEDIA_PATTERNS + RUSSIAN_NEWS_AGENCY_PATTERNS)
    ) else 0

    embassy_present = 1 if (
        has_any(text, RUSSIAN_EMBASSY_PATTERNS) or
        has_any(src_fields, RUSSIAN_EMBASSY_PATTERNS)
    ) else 0

    official_present = 1 if (
        has_any(text, RUSSIAN_DEFENSE_PATTERNS + RUSSIAN_MFA_PATTERNS + RUSSIAN_OFFICIAL_ATTRIBUTION_PATTERNS) or
        has_any(src_fields, RUSSIAN_DEFENSE_PATTERNS + RUSSIAN_MFA_PATTERNS)
    ) else 0

    state_source_present = 1 if (
        state_media_present == 1 or
        embassy_present == 1 or
        official_present == 1
    ) else 0

    proxy_present = 1 if has_any(text, PRO_RUSSIAN_PROXY_PATTERNS) else 0
    attr_present = 1 if attribution_score > 0 else 0

    # Conservative presence rule.
    russian_present = 1 if (
        russian_score >= 3 or
        source_type in [
            "russian_state_media",
            "russian_news_agency",
            "russian_embassy",
            "russian_military_defense",
            "russian_mfa",
            "russian_state_official",
            "mixed_russian_source",
        ]
    ) else 0

    # Dominance logic.
    russian_dominant = 0
    if russian_present == 1:
        if russian_score >= 7 and russian_score >= nonrus_score:
            russian_dominant = 1
        elif has_any(src_fields, RUSSIAN_STATE_MEDIA_PATTERNS + RUSSIAN_NEWS_AGENCY_PATTERNS):
            russian_dominant = 1
        elif has_any(src_fields, RUSSIAN_EMBASSY_PATTERNS + RUSSIAN_DEFENSE_PATTERNS + RUSSIAN_MFA_PATTERNS):
            russian_dominant = 1

    # Confidence logic.
    if russian_present == 0:
        confidence = "none"
    elif russian_score >= 7:
        confidence = "high"
    elif russian_score >= 4:
        confidence = "medium"
    else:
        confidence = "low"

    review_flag = 0
    review_reasons = []

    if russian_present == 1 and confidence == "low":
        review_flag = 1
        review_reasons.append("low_confidence_russian_signal")

    if russian_present == 1 and nonrus_score > russian_score:
        review_flag = 1
        review_reasons.append("nonrussian_source_score_higher_than_russian_score")

    if attr_present == 1 and russian_score < 4:
        review_flag = 1
        review_reasons.append("attribution_without_strong_source_basis")

    if source_type == "mixed_russian_source":
        review_flag = 1
        review_reasons.append("mixed_russian_source_type")

    if russian_present == 1 and safe_int(row.get("source_attributed_flag")) == 0:
        review_flag = 1
        review_reasons.append("russian_signal_without_stepB_source_attribution")

    return {
        "Russian_Source_Present": russian_present,
        "Russian_Source_Dominant": russian_dominant,
        "Russian_State_Source_Present": state_source_present,
        "Russian_State_Media_Source_Present": state_media_present,
        "Russian_Official_Source_Present": official_present,
        "Russian_Embassy_Source_Present": embassy_present,
        "ProRussian_Source_Present": proxy_present,
        "Russia_Attributed_Claim_Present": attr_present,
        "Russian_Source_Type": source_type,
        "Russian_Source_Names": source_names,
        "Russian_Source_Evidence": scores["Russian_Source_Evidence"],
        "Russian_Source_Score": russian_score,
        "Russian_Attribution_Score": attribution_score,
        "NonRussian_Source_Score": nonrus_score,
        "Russian_Source_Confidence": confidence,
        "Russian_Source_Review_Flag": review_flag,
        "Russian_Source_Review_Reason": "; ".join(review_reasons),
    }


# =============================================================================
# FINAL CODING LAYER MERGE
# =============================================================================

def load_final_coding_layer():
    """
    Preferred source:
      GEMINI/data/final_conservative_adjudicated_table.csv

    Fallback:
      data/postConsolidated.csv

    Returns normalized final columns:
      Final_Relevance
      Final_Counterterrorism
      ...
      Final_Dominant_Discourse
      Final_Actor_Mention
    """

    if not INPUT_CONSOLIDATED.exists():
        raise FileNotFoundError(f"Missing required file: {INPUT_CONSOLIDATED}")

    df_cons = pd.read_csv(INPUT_CONSOLIDATED, dtype={"Article_ID": str}, low_memory=False)
    df_cons["Article_ID"] = df_cons["Article_ID"].apply(normalize_article_id)

    needed_cons = [
        "Article_ID",
        "Relevance",
        "Counterterrorism",
        "Sovereignty",
        "Human_Rights_Abuse",
        "Anti_or_Neocolonialism",
        "Western_Failure",
        "Security_Effectiveness",
        "Economic_Interests",
        "Geopolitical_Rivalry",
        "Dominant_Discourse_Support",
        "Actor_Mention",
    ]
    df_cons = ensure_columns(df_cons, needed_cons)

    df_final = pd.DataFrame({
        "Article_ID": df_cons["Article_ID"],
        "Final_Relevance": df_cons["Relevance"].apply(safe_int),
        "Final_Counterterrorism": df_cons["Counterterrorism"].apply(safe_int),
        "Final_Sovereignty": df_cons["Sovereignty"].apply(safe_int),
        "Final_Human_Rights_Abuse": df_cons["Human_Rights_Abuse"].apply(safe_int),
        "Final_Anti_or_Neocolonialism": df_cons["Anti_or_Neocolonialism"].apply(safe_int),
        "Final_Western_Failure": df_cons["Western_Failure"].apply(safe_int),
        "Final_Security_Effectiveness": df_cons["Security_Effectiveness"].apply(safe_int),
        "Final_Economic_Interests": df_cons["Economic_Interests"].apply(safe_int),
        "Final_Geopolitical_Rivalry": df_cons["Geopolitical_Rivalry"].apply(safe_int),
        "Final_Dominant_Discourse": df_cons["Dominant_Discourse_Support"].apply(safe_int),
        "Final_Actor_Mention": df_cons["Actor_Mention"].apply(safe_int),
    })

    df_final["Final_Coding_Source"] = "postConsolidated_fallback_only"

    if INPUT_GEMINI_CONSERVATIVE.exists():
        df_adj = pd.read_csv(INPUT_GEMINI_CONSERVATIVE, dtype={"Article_ID": str}, low_memory=False)
        df_adj["Article_ID"] = df_adj["Article_ID"].apply(normalize_article_id)

        adj_needed = [
            "Article_ID",
            "Adj_V04_Relevance",
            "Adj_V12_Counterterrorism",
            "Adj_V13_Sovereignty",
            "Adj_V14_Human_Rights_Abuse",
            "Adj_V15_Anti_or_Neocolonialism",
            "Adj_V16_Western_Failure",
            "Adj_V17_Security_Effectiveness",
            "Adj_V18_Economic_Interests",
            "Adj_V19_Geopolitical_Rivalry",
            "Adj_V21_Dominant_Discourse",
            "Adj_V05_Actor_Mention",
        ]
        df_adj = ensure_columns(df_adj, adj_needed)
        df_adj = df_adj[adj_needed].drop_duplicates(subset=["Article_ID"], keep="last").copy()

        df_final = df_final.merge(df_adj, on="Article_ID", how="left")

        override_map = {
            "Final_Relevance": "Adj_V04_Relevance",
            "Final_Counterterrorism": "Adj_V12_Counterterrorism",
            "Final_Sovereignty": "Adj_V13_Sovereignty",
            "Final_Human_Rights_Abuse": "Adj_V14_Human_Rights_Abuse",
            "Final_Anti_or_Neocolonialism": "Adj_V15_Anti_or_Neocolonialism",
            "Final_Western_Failure": "Adj_V16_Western_Failure",
            "Final_Security_Effectiveness": "Adj_V17_Security_Effectiveness",
            "Final_Economic_Interests": "Adj_V18_Economic_Interests",
            "Final_Geopolitical_Rivalry": "Adj_V19_Geopolitical_Rivalry",
            "Final_Dominant_Discourse": "Adj_V21_Dominant_Discourse",
            "Final_Actor_Mention": "Adj_V05_Actor_Mention",
        }

        for final_col, adj_col in override_map.items():
            df_final[final_col] = df_final.apply(
                lambda r: safe_int(r.get(adj_col), safe_int(r.get(final_col))),
                axis=1
            )

        drop_cols = [c for c in df_final.columns if c.startswith("Adj_")]
        df_final = df_final.drop(columns=drop_cols)

        df_final["Final_Coding_Source"] = "gemini_conservative_with_consolidated_fallback"

    return df_final


def attach_final_coding_to_stepba(df_stepba):
    df_coding = load_final_coding_layer()

    if "article_id" not in df_stepba.columns:
        raise ValueError("StepB.A output must contain article_id")

    df_stepba = df_stepba.copy()
    df_stepba["article_id"] = df_stepba["article_id"].apply(normalize_article_id)

    df = df_stepba.merge(
        df_coding,
        left_on="article_id",
        right_on="Article_ID",
        how="left"
    )

    return df


# =============================================================================
# DISCUSSION TABLES
# =============================================================================

def build_frame_comparison(df):
    """
    Compares frame prevalence in:
      A. all articles with Final_Relevance >= threshold
      B. Russian-source articles within that threshold
      C. non-Russian-source articles within that threshold
    """

    for col in ["Final_Relevance", "Russian_Source_Present"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column for frame comparison: {col}")

    df_base = df[df["Final_Relevance"].fillna(0).astype(int) >= RELEVANCE_THRESHOLD].copy()
    df_ru = df_base[df_base["Russian_Source_Present"].fillna(0).astype(int) == 1].copy()
    df_nonru = df_base[df_base["Russian_Source_Present"].fillna(0).astype(int) == 0].copy()

    rows = []

    n_base = len(df_base)
    n_ru = len(df_ru)
    n_nonru = len(df_nonru)

    for frame_col, frame_label in FRAME_LABELS.items():
        if frame_col not in df_base.columns:
            continue

        overall_count = int((df_base[frame_col].fillna(0).astype(int) == 1).sum())
        ru_count = int((df_ru[frame_col].fillna(0).astype(int) == 1).sum())
        nonru_count = int((df_nonru[frame_col].fillna(0).astype(int) == 1).sum())

        overall_pct = percent(overall_count, n_base)
        ru_pct = percent(ru_count, n_ru)
        nonru_pct = percent(nonru_count, n_nonru)

        rows.append({
            "frame_col": frame_col,
            "frame": frame_label,
            "overall_n": n_base,
            "ru_source_n": n_ru,
            "nonru_source_n": n_nonru,

            "overall_count": overall_count,
            "overall_percent": overall_pct,

            "ru_source_count": ru_count,
            "ru_source_percent": ru_pct,

            "nonru_source_count": nonru_count,
            "nonru_source_percent": nonru_pct,

            "ru_minus_overall_pp": ru_pct - overall_pct,
            "ru_minus_nonru_pp": ru_pct - nonru_pct,
            "ru_overall_ratio": (ru_pct / overall_pct) if overall_pct > 0 else None,
        })

    return pd.DataFrame(rows)


def build_russian_source_by_outlet(df):
    if "outlet" not in df.columns:
        return pd.DataFrame()

    df_base = df[df["Final_Relevance"].fillna(0).astype(int) >= RELEVANCE_THRESHOLD].copy()

    rows = []

    for outlet, grp in df_base.groupby("outlet", dropna=False):
        n = len(grp)
        ru_n = int((grp["Russian_Source_Present"].fillna(0).astype(int) == 1).sum())
        ru_dom_n = int((grp["Russian_Source_Dominant"].fillna(0).astype(int) == 1).sum()) if "Russian_Source_Dominant" in grp.columns else 0
        ru_state_n = int((grp["Russian_State_Source_Present"].fillna(0).astype(int) == 1).sum()) if "Russian_State_Source_Present" in grp.columns else 0
        ru_media_n = int((grp["Russian_State_Media_Source_Present"].fillna(0).astype(int) == 1).sum()) if "Russian_State_Media_Source_Present" in grp.columns else 0
        ru_attr_n = int((grp["Russia_Attributed_Claim_Present"].fillna(0).astype(int) == 1).sum()) if "Russia_Attributed_Claim_Present" in grp.columns else 0

        rows.append({
            "outlet": outlet,
            "n_relevance_threshold_plus": n,
            "russian_source_present_n": ru_n,
            "russian_source_present_percent": percent(ru_n, n),
            "russian_source_dominant_n": ru_dom_n,
            "russian_source_dominant_percent": percent(ru_dom_n, n),
            "russian_state_source_n": ru_state_n,
            "russian_state_source_percent": percent(ru_state_n, n),
            "russian_state_media_source_n": ru_media_n,
            "russian_state_media_source_percent": percent(ru_media_n, n),
            "russia_attributed_claim_n": ru_attr_n,
            "russia_attributed_claim_percent": percent(ru_attr_n, n),
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["russian_source_present_percent", "russian_source_present_n"],
        ascending=[False, False]
    )


def build_russian_source_type_distribution(df):
    df_base = df[df["Final_Relevance"].fillna(0).astype(int) >= RELEVANCE_THRESHOLD].copy()

    if "Russian_Source_Type" not in df_base.columns:
        return pd.DataFrame()

    dist = (
        df_base["Russian_Source_Type"]
        .fillna("missing")
        .value_counts(dropna=False)
        .reset_index()
    )
    dist.columns = ["Russian_Source_Type", "count"]
    dist["percent_of_relevance_threshold_plus"] = dist["count"] / max(1, len(df_base)) * 100
    return dist


def build_russian_source_examples(df):
    df_base = df[df["Final_Relevance"].fillna(0).astype(int) >= RELEVANCE_THRESHOLD].copy()
    df_ru = df_base[df_base["Russian_Source_Present"].fillna(0).astype(int) == 1].copy()

    if df_ru.empty:
        return pd.DataFrame()

    sort_cols = ["Russian_Source_Score", "Russian_Attribution_Score"]
    for col in sort_cols:
        if col not in df_ru.columns:
            df_ru[col] = 0

    df_ru = df_ru.sort_values(sort_cols, ascending=[False, False]).head(TOP_EXAMPLES_N)

    cols = [
        "article_id",
        "outlet",
        "date_iso_full",
        "headline_clean",
        "url",
        "Final_Relevance",
        "Russian_Source_Type",
        "Russian_Source_Names",
        "Russian_Source_Confidence",
        "Russian_Source_Score",
        "Russian_Attribution_Score",
        "Russian_Source_Evidence",
        "Russian_Source_Review_Flag",
        "Russian_Source_Review_Reason",
        "explicit_external_source_name",
        "author_external_source_name",
        "likely_republished_basis",
        "source_attributed_flag",
        "likely_republished_flag",
    ]
    cols = [c for c in cols if c in df_ru.columns]
    return df_ru[cols].copy()


# =============================================================================
# HEATMAPS
# =============================================================================

def plot_frame_prevalence_heatmap(frame_comp):
    if frame_comp.empty:
        return None

    plot_df = frame_comp.set_index("frame")[
        ["overall_percent", "ru_source_percent", "nonru_source_percent"]
    ].copy()

    plot_df = plot_df.rename(columns={
        "overall_percent": f"All Relevance {RELEVANCE_THRESHOLD}+",
        "ru_source_percent": "RU-source subset",
        "nonru_source_percent": "Non-RU-source subset",
    })

    out_path = OUTPUT_FIGURES_DIR / f"stepBA_frame_prevalence_heatmap_rel{RELEVANCE_THRESHOLD}plus.png"

    plt.figure(figsize=(9, 6))

    if SEABORN_AVAILABLE:
        sns.heatmap(
            plot_df,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            cbar_kws={"label": "Frame prevalence (%)"}
        )
    else:
        plt.imshow(plot_df.values, aspect="auto")
        plt.colorbar(label="Frame prevalence (%)")
        plt.xticks(range(len(plot_df.columns)), plot_df.columns, rotation=45, ha="right")
        plt.yticks(range(len(plot_df.index)), plot_df.index)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                plt.text(j, i, f"{plot_df.iloc[i, j]:.1f}", ha="center", va="center")

    plt.title(f"Frame prevalence: overall vs Russian-source subset, Relevance {RELEVANCE_THRESHOLD}+")
    plt.ylabel("Frame")
    plt.xlabel("Subset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return str(out_path)


def plot_frame_difference_heatmap(frame_comp):
    if frame_comp.empty:
        return None

    plot_df = frame_comp.set_index("frame")[
        ["ru_minus_overall_pp", "ru_minus_nonru_pp"]
    ].copy()

    plot_df = plot_df.rename(columns={
        "ru_minus_overall_pp": "RU minus overall",
        "ru_minus_nonru_pp": "RU minus non-RU",
    })

    out_path = OUTPUT_FIGURES_DIR / f"stepBA_frame_difference_heatmap_rel{RELEVANCE_THRESHOLD}plus.png"

    plt.figure(figsize=(7, 6))

    vmax = max(abs(plot_df.min().min()), abs(plot_df.max().max()))
    if vmax == 0:
        vmax = 1

    if SEABORN_AVAILABLE:
        sns.heatmap(
            plot_df,
            annot=True,
            fmt=".1f",
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            cbar_kws={"label": "Difference in percentage points"}
        )
    else:
        plt.imshow(plot_df.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(label="Difference in percentage points")
        plt.xticks(range(len(plot_df.columns)), plot_df.columns, rotation=45, ha="right")
        plt.yticks(range(len(plot_df.index)), plot_df.index)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                plt.text(j, i, f"{plot_df.iloc[i, j]:.1f}", ha="center", va="center")

    plt.title(f"Frame over-/under-representation in Russian-source subset, Relevance {RELEVANCE_THRESHOLD}+")
    plt.ylabel("Frame")
    plt.xlabel("Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return str(out_path)


def plot_outlet_russian_source_heatmap(outlet_comp):
    if outlet_comp is None or outlet_comp.empty:
        return None

    keep_cols = [
        "russian_source_present_percent",
        "russian_source_dominant_percent",
        "russian_state_source_percent",
        "russian_state_media_source_percent",
        "russia_attributed_claim_percent",
    ]
    keep_cols = [c for c in keep_cols if c in outlet_comp.columns]

    if not keep_cols:
        return None

    plot_df = outlet_comp.set_index("outlet")[keep_cols].copy()

    plot_df = plot_df.rename(columns={
        "russian_source_present_percent": "RU source present",
        "russian_source_dominant_percent": "RU source dominant",
        "russian_state_source_percent": "RU state source",
        "russian_state_media_source_percent": "RU state media",
        "russia_attributed_claim_percent": "Russia-attributed claim",
    })

    out_path = OUTPUT_FIGURES_DIR / f"stepBA_outlet_russian_source_heatmap_rel{RELEVANCE_THRESHOLD}plus.png"

    height = max(6, min(14, 0.5 * len(plot_df) + 2))
    plt.figure(figsize=(9, height))

    if SEABORN_AVAILABLE:
        sns.heatmap(
            plot_df,
            annot=True,
            fmt=".1f",
            cmap="OrRd",
            cbar_kws={"label": "Share within outlet (%)"}
        )
    else:
        plt.imshow(plot_df.values, aspect="auto", cmap="OrRd")
        plt.colorbar(label="Share within outlet (%)")
        plt.xticks(range(len(plot_df.columns)), plot_df.columns, rotation=45, ha="right")
        plt.yticks(range(len(plot_df.index)), plot_df.index)
        for i in range(plot_df.shape[0]):
            for j in range(plot_df.shape[1]):
                plt.text(j, i, f"{plot_df.iloc[i, j]:.1f}", ha="center", va="center")

    plt.title(f"Russian-source signals by outlet, Relevance {RELEVANCE_THRESHOLD}+")
    plt.ylabel("Outlet")
    plt.xlabel("Russian-source indicator")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return str(out_path)


# =============================================================================
# DISCUSSION NOTES
# =============================================================================

def write_discussion_notes(df_merged, frame_comp, outlet_comp, type_dist,
                           heatmap_path, diff_heatmap_path, outlet_heatmap_path):
    df_base = df_merged[df_merged["Final_Relevance"].fillna(0).astype(int) >= RELEVANCE_THRESHOLD].copy()
    df_ru = df_base[df_base["Russian_Source_Present"].fillna(0).astype(int) == 1].copy()

    n_base = len(df_base)
    n_ru = len(df_ru)
    ru_share = percent(n_ru, n_base)

    if not frame_comp.empty and n_ru > 0:
        over = frame_comp.sort_values("ru_minus_overall_pp", ascending=False).head(3)
        under = frame_comp.sort_values("ru_minus_overall_pp", ascending=True).head(3)

        over_lines = [
            f"- {r['frame']}: {r['ru_source_percent']:.1f}% in RU-source subset vs "
            f"{r['overall_percent']:.1f}% overall "
            f"({r['ru_minus_overall_pp']:+.1f} pp)"
            for _, r in over.iterrows()
        ]

        under_lines = [
            f"- {r['frame']}: {r['ru_source_percent']:.1f}% in RU-source subset vs "
            f"{r['overall_percent']:.1f}% overall "
            f"({r['ru_minus_overall_pp']:+.1f} pp)"
            for _, r in under.iterrows()
        ]
    else:
        over_lines = ["- N/A"]
        under_lines = ["- N/A"]

    if outlet_comp is not None and not outlet_comp.empty:
        top_outlet_lines = []
        for _, r in outlet_comp.head(5).iterrows():
            top_outlet_lines.append(
                f"- {r['outlet']}: {int(r['russian_source_present_n'])} / "
                f"{int(r['n_relevance_threshold_plus'])} "
                f"({r['russian_source_present_percent']:.1f}%)"
            )
    else:
        top_outlet_lines = ["- N/A"]

    if type_dist is not None and not type_dist.empty:
        type_lines = []
        for _, r in type_dist.head(10).iterrows():
            type_lines.append(
                f"- {r['Russian_Source_Type']}: {int(r['count'])} "
                f"({r['percent_of_relevance_threshold_plus']:.1f}% of relevance-threshold subset)"
            )
    else:
        type_lines = ["- N/A"]

    text = f"""
STEPB.A DISCUSSION NOTES: RUSSIAN / RUSSIA-ATTRIBUTED SOURCE ENVIRONMENT
Generated: {datetime.now().isoformat(timespec="seconds")}

Scope:
- Relevance threshold used: Final_Relevance >= {RELEVANCE_THRESHOLD}
- Overall comparison set: {n_base} articles
- Russian-source subset: {n_ru} articles
- Share of Russian-source subset within comparison set: {ru_share:.1f}%

Interpretive use:
This output should be treated as a supplementary source-environment indicator. It does not by itself measure editorial alignment, propaganda, or authorial endorsement. It identifies articles where Russian, Russian state, Russian state-media, pro-Russian, or Russia-attributed source signals are present in the source/republication environment detected by StepB.A.

Russian-source type distribution:
{chr(10).join(type_lines)}

Most over-represented frames in the Russian-source subset:
{chr(10).join(over_lines)}

Most under-represented frames in the Russian-source subset:
{chr(10).join(under_lines)}

Outlet-level Russian-source concentration, top outlets:
{chr(10).join(top_outlet_lines)}

Caveats:
- Detection is rule-based and should be read as conservative support, not final truth.
- Some Russia-attributed claims may be quoted critically rather than endorsed.
- Russian source presence is not equivalent to pro-Russian stance.
- Very small Russian-source subsets require caution.
- Results depend on StepB source/republication detection quality.
- If StepB was run on all articles, the relevance threshold here controls the analytical comparison.
- If StepB was already run only on a filtered subset, this output inherits that prior filtering.
- Frame comparison uses Gemini conservative adjudicated coding when available, otherwise postConsolidated fallback.

Heatmaps:
- Frame prevalence heatmap: {heatmap_path}
- Frame difference heatmap: {diff_heatmap_path}
- Outlet Russian-source heatmap: {outlet_heatmap_path}

Key output tables:
- stepBA_russian_sources_with_final_coding.csv
- stepBA_frame_comparison_rel{RELEVANCE_THRESHOLD}plus.csv
- stepBA_russian_source_by_outlet_rel{RELEVANCE_THRESHOLD}plus.csv
- stepBA_russian_source_type_distribution_rel{RELEVANCE_THRESHOLD}plus.csv
- stepBA_russian_source_examples_rel{RELEVANCE_THRESHOLD}plus.csv
""".strip()

    out_path = OUTPUT_DISCUSSION_DIR / f"stepBA_discussion_notes_rel{RELEVANCE_THRESHOLD}plus.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    return str(out_path)


# =============================================================================
# DISCUSSION OUTPUT MASTER
# =============================================================================

def generate_stepBA_discussion_outputs(df_stepba_output):
    df_merged = attach_final_coding_to_stepba(df_stepba_output)

    merged_path = OUTPUT_DISCUSSION_DIR / "stepBA_russian_sources_with_final_coding.csv"
    save_csv(df_merged, merged_path)

    frame_comp = build_frame_comparison(df_merged)
    frame_comp_path = OUTPUT_DISCUSSION_DIR / f"stepBA_frame_comparison_rel{RELEVANCE_THRESHOLD}plus.csv"
    save_csv(frame_comp, frame_comp_path)

    outlet_comp = build_russian_source_by_outlet(df_merged)
    outlet_comp_path = OUTPUT_DISCUSSION_DIR / f"stepBA_russian_source_by_outlet_rel{RELEVANCE_THRESHOLD}plus.csv"
    save_csv(outlet_comp, outlet_comp_path)

    type_dist = build_russian_source_type_distribution(df_merged)
    type_dist_path = OUTPUT_DISCUSSION_DIR / f"stepBA_russian_source_type_distribution_rel{RELEVANCE_THRESHOLD}plus.csv"
    save_csv(type_dist, type_dist_path)

    examples = build_russian_source_examples(df_merged)
    examples_path = OUTPUT_DISCUSSION_DIR / f"stepBA_russian_source_examples_rel{RELEVANCE_THRESHOLD}plus.csv"
    save_csv(examples, examples_path)

    heatmap_path = plot_frame_prevalence_heatmap(frame_comp)
    diff_heatmap_path = plot_frame_difference_heatmap(frame_comp)
    outlet_heatmap_path = plot_outlet_russian_source_heatmap(outlet_comp)

    notes_path = write_discussion_notes(
        df_merged=df_merged,
        frame_comp=frame_comp,
        outlet_comp=outlet_comp,
        type_dist=type_dist,
        heatmap_path=heatmap_path,
        diff_heatmap_path=diff_heatmap_path,
        outlet_heatmap_path=outlet_heatmap_path,
    )

    print("\nStepB.A discussion outputs generated:")
    print(f"- merged coding/source table: {merged_path}")
    print(f"- frame comparison: {frame_comp_path}")
    print(f"- outlet comparison: {outlet_comp_path}")
    print(f"- source type distribution: {type_dist_path}")
    print(f"- examples: {examples_path}")
    print(f"- prevalence heatmap: {heatmap_path}")
    print(f"- difference heatmap: {diff_heatmap_path}")
    print(f"- outlet heatmap: {outlet_heatmap_path}")
    print(f"- discussion notes: {notes_path}")

    return {
        "merged_path": str(merged_path),
        "frame_comparison_path": str(frame_comp_path),
        "outlet_comparison_path": str(outlet_comp_path),
        "source_type_distribution_path": str(type_dist_path),
        "examples_path": str(examples_path),
        "heatmap_path": heatmap_path,
        "difference_heatmap_path": diff_heatmap_path,
        "outlet_heatmap_path": outlet_heatmap_path,
        "discussion_notes_path": notes_path,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dirs()

    if not INPUT_STEPB.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_STEPB}")

    df = pd.read_csv(INPUT_STEPB, dtype={"article_id": str}, low_memory=False)

    expected_cols = [
        "article_id",
        "outlet",
        "url",
        "date_iso_full",
        "headline_clean",
        "lead_clean",
        "body_postclean",
        "author_clean",
        "rubrique_clean",

        "explicit_external_source_flag",
        "explicit_external_source_type",
        "explicit_external_source_name",
        "explicit_external_source_count",

        "author_external_source_flag",
        "author_external_source_name",
        "author_external_source_type",

        "explicit_malian_media_reference_flag",
        "explicit_malian_media_reference_names",
        "explicit_malian_media_reference_count",
        "explicit_malian_media_reference_type",

        "republication_phrase_flag",
        "republication_phrase_count",
        "republication_phrase_note",

        "attribution_phrase_flag",
        "attribution_phrase_count",
        "attribution_phrase_note",

        "source_attributed_flag",
        "republication_index",
        "likely_republished_basis",
        "republication_confidence",
        "likely_republished_flag",

        "near_duplicate_flag",
        "near_duplicate_cluster_id",
        "near_duplicate_match_count",
        "near_duplicate_cross_outlet_flag",
        "near_duplicate_same_outlet_flag",
        "near_duplicate_top_match_id",
        "near_duplicate_top_match_outlet",
        "near_duplicate_top_match_score",
    ]
    df = ensure_columns(df, expected_cols)

    df["article_id"] = df["article_id"].apply(normalize_article_id)

    print("\nStepB.A: Russian / Russia-attributed source analysis")
    print("=" * 70)
    print(f"Loaded StepB rows: {len(df)}")
    print(f"Input file: {INPUT_STEPB}")
    print(f"Output directory: {STEPBA_OUTPUT_DIR}")
    print(f"Relevance threshold for discussion outputs: {RELEVANCE_THRESHOLD}+")
    print("\nDetecting Russian-source signals...")

    coded = df.apply(code_russian_source, axis=1, result_type="expand")
    df_out = pd.concat([df, coded], axis=1)

    export_cols = [
        "article_id",
        "outlet",
        "date_iso_full",
        "headline_clean",
        "url",

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

        "source_attributed_flag",
        "republication_index",
        "likely_republished_flag",
        "likely_republished_basis",
        "republication_confidence",

        "explicit_external_source_flag",
        "explicit_external_source_type",
        "explicit_external_source_name",
        "author_external_source_flag",
        "author_external_source_name",
        "explicit_malian_media_reference_flag",
        "explicit_malian_media_reference_names",

        "republication_phrase_flag",
        "republication_phrase_count",
        "republication_phrase_note",
        "attribution_phrase_flag",
        "attribution_phrase_count",
        "attribution_phrase_note",

        "near_duplicate_flag",
        "near_duplicate_cluster_id",
        "near_duplicate_match_count",
        "near_duplicate_cross_outlet_flag",
        "near_duplicate_top_match_id",
        "near_duplicate_top_match_outlet",
        "near_duplicate_top_match_score",

        "lead_clean",
        "body_postclean",
    ]
    export_cols = [c for c in export_cols if c in df_out.columns]

    df_export = df_out[export_cols].copy()
    save_csv(df_export, OUTPUT_CSV)

    df_review = df_export[df_export["Russian_Source_Review_Flag"].fillna(0).astype(int) == 1].copy()
    save_csv(df_review, OUTPUT_REVIEW_CSV)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total = len(df_export)
    russian_present = int((df_export["Russian_Source_Present"] == 1).sum())
    russian_dominant = int((df_export["Russian_Source_Dominant"] == 1).sum())
    russian_state = int((df_export["Russian_State_Source_Present"] == 1).sum())
    russian_media = int((df_export["Russian_State_Media_Source_Present"] == 1).sum())
    russian_official = int((df_export["Russian_Official_Source_Present"] == 1).sum())
    russian_embassy = int((df_export["Russian_Embassy_Source_Present"] == 1).sum())
    pro_russian = int((df_export["ProRussian_Source_Present"] == 1).sum())
    attributed_claim = int((df_export["Russia_Attributed_Claim_Present"] == 1).sum())
    review_n = int((df_export["Russian_Source_Review_Flag"] == 1).sum())

    type_dist = df_export["Russian_Source_Type"].value_counts(dropna=False).to_dict()
    conf_dist = df_export["Russian_Source_Confidence"].value_counts(dropna=False).to_dict()

    lines = []
    lines.append("StepB.A Russian / Russia-attributed source analysis summary")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Input: {INPUT_STEPB}")
    lines.append(f"Output directory: {STEPBA_OUTPUT_DIR}")
    lines.append(f"Rows analysed: {total}")
    lines.append("")
    lines.append("Core counts:")
    lines.append(f"- Russian_Source_Present: {russian_present}")
    lines.append(f"- Russian_Source_Dominant: {russian_dominant}")
    lines.append(f"- Russian_State_Source_Present: {russian_state}")
    lines.append(f"- Russian_State_Media_Source_Present: {russian_media}")
    lines.append(f"- Russian_Official_Source_Present: {russian_official}")
    lines.append(f"- Russian_Embassy_Source_Present: {russian_embassy}")
    lines.append(f"- ProRussian_Source_Present: {pro_russian}")
    lines.append(f"- Russia_Attributed_Claim_Present: {attributed_claim}")
    lines.append(f"- Review flagged: {review_n}")
    lines.append("")
    lines.append("Russian_Source_Type distribution:")
    for k, v in type_dist.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Russian_Source_Confidence distribution:")
    for k, v in conf_dist.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Main outputs:")
    lines.append(f"- {OUTPUT_CSV}")
    lines.append(f"- {OUTPUT_REVIEW_CSV}")
    lines.append("")
    lines.append("Discussion outputs:")
    lines.append(f"- {OUTPUT_DISCUSSION_DIR}")
    lines.append(f"- {OUTPUT_FIGURES_DIR}")
    lines.append("")
    lines.append("Methodological note:")
    lines.append(
        "This layer is a supplementary source-environment enrichment. "
        "Russian-source presence is not equivalent to editorial endorsement or pro-Russian stance."
    )

    with open(OUTPUT_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nStepB.A detection diagnostics:")
    print(f"Rows analysed: {total}")
    print(f"Russian_Source_Present: {russian_present}")
    print(f"Russian_Source_Dominant: {russian_dominant}")
    print(f"Russia_Attributed_Claim_Present: {attributed_claim}")
    print(f"Review flagged: {review_n}")
    print(f"\nSaved main output to: {OUTPUT_CSV}")
    print(f"Saved review output tos: {OUTPUT_REVIEW_CSV}")
    print(f"Saved summary to: {OUTPUT_SUMMARY_TXT}")

    # -------------------------------------------------------------------------
    # Discussion-oriented outputs
    # -------------------------------------------------------------------------
    print("\nGenerating discussion-ready outputs and heatmaps...")
    generate_stepBA_discussion_outputs(df_export)

    print("\nStepB.A completed successfully.")


if __name__ == "__main__":
    main()