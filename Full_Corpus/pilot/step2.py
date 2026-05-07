import pandas as pd
import re

INPUT_PATH = "data/postStep1.csv"
OUTPUT_CSV = "data/postStep2.csv"

# =========================
# 1. LEXICONS
# =========================
TARGET_PATTERNS = {
    "wagner": [
        r"\bwagner\b",
        r"\bgroupe\s+wagner\b",
        r"\bwagner\s+group\b",
    ],
    "africa_corps": [
        r"\bafrica\s+corps\b",
        r"\bcorps\s+africain\b",
        r"\bcorps\s+africain\s+russe\b",
    ],
    "russian_mercenaries": [
        r"\bmercenaires?\s+russes?\b",
        r"\bparamilitaires?\s+russes?\b",
    ],
    "russian_instructors": [
        r"\binstructeurs?\s+russes?\b",
        r"\bformateurs?\s+russes?\b",
        r"\bcoop[ée]rants?\s+russes?\b",
    ],
}

MALI_CONTEXT_PATTERNS = [
    r"\bmali\b",
    r"\bbamako\b",
    r"\bmalien\b",
    r"\bmalienne\b",
    r"\bmaliennes\b",
    r"\bmaliens\b",
    r"\bfama\b",
    r"\barmée\s+malienne\b",
    r"\bautorités\s+maliennes\b",
    r"\bgouvernement\s+malien\b",
    r"\btransition\s+malienne\b",
    r"\bassimi\s+go[iï]ta\b",
    r"\bkidal\b",
    r"\bgao\b",
    r"\btombouctou\b",
    r"\bm[ée]naka\b",
    r"\bmopti\b",
    r"\bminusma\b",
    r"\bbarkhane\b",
    r"\btakuba\b",
]

STRONG_MALI_FOCUS_PATTERNS = [
    r"\bbamako\b",
    r"\bfama\b",
    r"\barmée\s+malienne\b",
    r"\bautorités\s+maliennes\b",
    r"\bgouvernement\s+malien\b",
    r"\btransition\s+malienne\b",
    r"\bassimi\s+go[iï]ta\b",
    r"\bkidal\b",
    r"\bgao\b",
    r"\btombouctou\b",
    r"\bm[ée]naka\b",
    r"\bmopti\b",
    r"\bminusma\b",
    r"\bbarkhane\b",
    r"\btakuba\b",
]

NON_MALI_LOCATION_PATTERNS = [
    r"\bniger\b",
    r"\bniamey\b",
    r"\bburkina\b",
    r"\bouagadougou\b",
    r"\bburkina\s+faso\b",
    r"\bcentrafrique\b",
    r"\br[ée]publique\s+centrafricaine\b",
    r"\bukraine\b",
    r"\bbakhmout\b",
    r"\bdonbass\b",
    r"\bsyrie\b",
    r"\blibye\b",
    r"\bmauritanie\b",
    r"\bc[ôo]te d['’]ivoire\b",
    r"\bb[ée]nin\b",
    r"\bs[ée]n[ée]gal\b",
    r"\bgambie\b",
    r"\btchad\b",
    r"\bafrique du sud\b",
    r"\begypte\b",
    r"\bafghanistan\b",
    r"\biran\b",
    r"\bisra[ëe]l\b",
]

MALI_SPECIFIC_LINKAGE_PATTERNS = [
    r"\barmée\s+malienne\b",
    r"\bfama\b",
    r"\bforces?\s+arm[ée]es?\s+maliennes\b",
    r"\bautorités\s+maliennes\b",
    r"\bgouvernement\s+malien\b",
    r"\btransition\s+malienne\b",
    r"\bassimi\s+go[iï]ta\b",
    r"\bjunte\b",
    r"\bsupplétifs?\b",
]

GENERIC_LINKAGE_PATTERNS = [
    r"\bexactions?\b",
    r"\bviolations?\s+des?\s+droits",
    r"\bex[ée]cutions?\b",
    r"\bmassacres?\b",
    r"\bconvoi\b",
    r"\bopérations?\b",
    r"\bdéploiement\b",
    r"\bsanctions?\b",
    r"\bcoopération\b",
    r"\bcontrat\b",
    r"\btroupes?\b",
    r"\bsoldats?\b",
]

# =========================
# 2. HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

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

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def count_pattern_matches(text, patterns):
    text = safe_str(text)
    if not text:
        return 0
    total = 0
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        total += len(matches)
    return total

def find_pattern_labels(text, pattern_dict):
    text = safe_str(text)
    found = []
    if not text:
        return found
    for label, patterns in pattern_dict.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                found.append(label)
                break
    return found

def find_matching_patterns(text, patterns):
    text = safe_str(text)
    found = []
    if not text:
        return found
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            found.append(pat)
    return found

def split_sentences(text):
    text = safe_str(text)
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

def has_any_target(text):
    text = safe_str(text)
    if not text:
        return False
    for pats in TARGET_PATTERNS.values():
        for pat in pats:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
    return False

# =========================
# 3. SCOPE
# =========================
def in_scope_period(row):
    year = safe_int(row.get("date_year"), default=-9999)
    month = safe_int(row.get("date_month"), default=-9999)

    if year == -9999 or month == -9999:
        return 1

    if year < 2021 or year > 2025:
        return 0
    if year == 2021 and month < 7:
        return 0
    return 1

# =========================
# 4. SEGMENT-LEVEL SCORING
# =========================
def score_target_segments(row):
    headline = safe_str(row.get("headline_clean", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))

    target_hits_headline = sum(count_pattern_matches(headline, pats) for pats in TARGET_PATTERNS.values())
    target_hits_lead = sum(count_pattern_matches(lead, pats) for pats in TARGET_PATTERNS.values())
    target_hits_body = sum(count_pattern_matches(body, pats) for pats in TARGET_PATTERNS.values())

    target_types = list(set(
        find_pattern_labels(headline, TARGET_PATTERNS) +
        find_pattern_labels(lead, TARGET_PATTERNS) +
        find_pattern_labels(body, TARGET_PATTERNS)
    ))

    return {
        "target_hits_headline": target_hits_headline,
        "target_hits_lead": target_hits_lead,
        "target_hits_body": target_hits_body,
        "target_hits_total": target_hits_headline + target_hits_lead + target_hits_body,
        "target_types_found": ", ".join(sorted(target_types)) if target_types else ""
    }

def score_mali_segments(row):
    headline = safe_str(row.get("headline_clean", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))

    mali_hits_headline = count_pattern_matches(headline, MALI_CONTEXT_PATTERNS)
    mali_hits_lead = count_pattern_matches(lead, MALI_CONTEXT_PATTERNS)
    mali_hits_body = count_pattern_matches(body, MALI_CONTEXT_PATTERNS)

    found_terms = list(set(
        find_matching_patterns(headline, MALI_CONTEXT_PATTERNS) +
        find_matching_patterns(lead, MALI_CONTEXT_PATTERNS) +
        find_matching_patterns(body, MALI_CONTEXT_PATTERNS)
    ))

    return {
        "mali_hits_headline": mali_hits_headline,
        "mali_hits_lead": mali_hits_lead,
        "mali_hits_body": mali_hits_body,
        "mali_hits_total": mali_hits_headline + mali_hits_lead + mali_hits_body,
        "mali_terms_found": ", ".join(sorted(found_terms)) if found_terms else ""
    }

def score_non_mali_context(row):
    headline = safe_str(row.get("headline_clean", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))

    return (
        count_pattern_matches(headline, NON_MALI_LOCATION_PATTERNS) +
        count_pattern_matches(lead, NON_MALI_LOCATION_PATTERNS) +
        count_pattern_matches(body, NON_MALI_LOCATION_PATTERNS)
    )

def score_strong_mali_focus(row):
    headline = safe_str(row.get("headline_clean", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))

    return (
        count_pattern_matches(headline, STRONG_MALI_FOCUS_PATTERNS) +
        count_pattern_matches(lead, STRONG_MALI_FOCUS_PATTERNS) +
        count_pattern_matches(body, STRONG_MALI_FOCUS_PATTERNS)
    )

def score_mali_specific_linkage(row):
    body = safe_str(row.get("body_postclean", ""))
    hits = 0

    for sent in split_sentences(body):
        if has_any_target(sent) and any(re.search(pat, sent, flags=re.IGNORECASE) for pat in MALI_SPECIFIC_LINKAGE_PATTERNS):
            hits += 1

    return hits

def score_generic_linkage(row):
    body = safe_str(row.get("body_postclean", ""))
    hits = 0

    for sent in split_sentences(body):
        if has_any_target(sent) and any(re.search(pat, sent, flags=re.IGNORECASE) for pat in GENERIC_LINKAGE_PATTERNS):
            hits += 1

    return hits

# =========================
# 5. TARGET CENTRALITY FEATURES
# =========================
def target_sentence_metrics(row):
    headline = safe_str(row.get("headline_clean", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))

    full = " ".join([headline, lead, body]).strip()
    sents = split_sentences(full)

    if not sents:
        return {
            "target_sentence_count": 0,
            "target_sentence_share": 0.0,
            "target_cluster_max_run": 0,
            "target_in_first_third": 0,
        }

    flags = [1 if has_any_target(s) else 0 for s in sents]
    count = sum(flags)
    share = count / len(sents) if len(sents) > 0 else 0.0

    max_run = 0
    current_run = 0
    for f in flags:
        if f == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    first_third_end = max(1, len(sents) // 3)
    target_in_first_third = 1 if any(flags[:first_third_end]) else 0

    return {
        "target_sentence_count": count,
        "target_sentence_share": round(share, 4),
        "target_cluster_max_run": max_run,
        "target_in_first_third": target_in_first_third,
    }

def is_bulletin_style(row):
    headline = safe_str(row.get("headline_clean", "")).lower()
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))

    bullet_markers = body.count("👉") + lead.count("👉")

    if "les titres du" in headline:
        return 1
    if bullet_markers >= 3:
        return 1
    if "ci-dessous, écoutez" in body.lower():
        return 1

    return 0

def approximate_article_length(row):
    headline = safe_str(row.get("headline_clean", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))
    text = " ".join([headline, lead, body]).strip()
    return len(split_sentences(text))

# =========================
# 6. RELEVANCE SCORING
# =========================
def compute_relevance_score(row):
    score = 0

    target_hits_headline = safe_int(row.get("target_hits_headline"))
    target_hits_lead = safe_int(row.get("target_hits_lead"))
    target_hits_body = safe_int(row.get("target_hits_body"))

    mali_hits_headline = safe_int(row.get("mali_hits_headline"))
    mali_hits_lead = safe_int(row.get("mali_hits_lead"))
    mali_hits_body = safe_int(row.get("mali_hits_body"))

    strong_mali_focus_hits = safe_int(row.get("strong_mali_focus_hits"))
    mali_specific_linkage_hits = safe_int(row.get("mali_specific_linkage_hits"))
    generic_linkage_hits = safe_int(row.get("generic_linkage_hits"))

    target_sentence_count = safe_int(row.get("target_sentence_count"))
    target_sentence_share = safe_float(row.get("target_sentence_share"))
    target_cluster_max_run = safe_int(row.get("target_cluster_max_run"))

    mali_hits_total = safe_int(row.get("mali_hits_total"))
    non_mali_hits_total = safe_int(row.get("non_mali_hits_total"))
    is_bulletin = safe_int(row.get("is_bulletin_style"))

    target_types = safe_str(row.get("target_types_found"))

    score += min(target_hits_headline, 2) * 4
    score += min(target_hits_lead, 3) * 3
    score += min(target_hits_body, 6) * 1

    score += min(mali_hits_headline, 2) * 3
    score += min(mali_hits_lead, 3) * 2
    score += min(mali_hits_body, 6) * 1

    score += min(strong_mali_focus_hits, 4) * 1

    score += min(mali_specific_linkage_hits, 3) * 2
    score += min(generic_linkage_hits, 2) * 1

    score += min(target_sentence_count, 4) * 1
    if target_sentence_share >= 0.20:
        score += 2
    elif target_sentence_share >= 0.10:
        score += 1

    if target_cluster_max_run >= 2:
        score += 2
    if safe_int(row.get("target_in_first_third")) == 1:
        score += 1

    if "wagner" in target_types:
        score += 2
    if "africa_corps" in target_types:
        score += 2

    if mali_hits_total == 0 and non_mali_hits_total > 0:
        score -= 3

    if non_mali_hits_total >= 8 and mali_hits_total <= 2 and strong_mali_focus_hits == 0:
        score -= 3

    if is_bulletin == 1:
        score -= 3

    return score

def assign_relevance_code(row):
    if safe_int(row.get("in_scope_period")) == 0:
        return 1

    target_total = safe_int(row.get("target_hits_total"))
    mali_total = safe_int(row.get("mali_hits_total"))
    strong_mali = safe_int(row.get("strong_mali_focus_hits"))
    non_mali_total = safe_int(row.get("non_mali_hits_total"))
    score = safe_int(row.get("relevance_score"))
    mali_linkage = safe_int(row.get("mali_specific_linkage_hits"))
    generic_linkage = safe_int(row.get("generic_linkage_hits"))

    target_sentence_count = safe_int(row.get("target_sentence_count"))
    target_sentence_share = safe_float(row.get("target_sentence_share"))
    target_cluster_max_run = safe_int(row.get("target_cluster_max_run"))
    bulletin = safe_int(row.get("is_bulletin_style"))
    article_sentence_count = safe_int(row.get("article_sentence_count"))

    target_in_headline = safe_int(row.get("target_hits_headline")) > 0
    target_in_lead = safe_int(row.get("target_hits_lead")) > 0
    target_in_body_multiple = safe_int(row.get("target_hits_body")) >= 2

    target_only_once_in_body = (
        safe_int(row.get("target_hits_total")) == 1 and
        safe_int(row.get("target_hits_headline")) == 0 and
        safe_int(row.get("target_hits_lead")) == 0 and
        safe_int(row.get("target_hits_body")) == 1
    )

    if target_total == 0:
        return 1

    if bulletin == 1 and target_sentence_count <= 2 and not target_in_headline:
        return 2

    if target_total > 0 and mali_total == 0:
        return 2

    if target_sentence_count == 1 and not target_in_headline and not target_in_lead and safe_int(row.get("target_hits_body")) <= 2:
        return 2

    if target_only_once_in_body:
        if (
            mali_total >= 2 and
            (strong_mali >= 2 or mali_total >= 6) and
            (
                mali_linkage >= 1 or
                (generic_linkage >= 1 and non_mali_total <= 5 and score >= 10)
            ) and
            target_sentence_count >= 1
        ):
            return 3
        return 2

    if (
        target_total <= 4 and
        target_sentence_count <= 2 and
        target_cluster_max_run <= 1 and
        not target_in_headline and
        safe_int(row.get("target_hits_lead")) <= 2 and
        (non_mali_total >= mali_total or mali_linkage == 0)
    ):
        return 2

    if (
        not target_in_headline and
        target_in_lead and
        target_sentence_count <= 2 and
        target_cluster_max_run <= 1 and
        target_sentence_share < 0.12
    ):
        return 2

    if (
        target_sentence_count <= 2 and
        target_sentence_share < 0.10 and
        target_cluster_max_run <= 1 and
        not target_in_headline and
        non_mali_total >= 6
    ):
        return 2

    if (
        non_mali_total >= 8 and
        mali_total <= 2 and
        strong_mali == 0 and
        not target_in_headline and
        safe_int(row.get("target_hits_lead")) <= 1 and
        safe_int(row.get("target_hits_body")) <= 2
    ):
        return 2

    if (
        target_in_headline and
        non_mali_total >= 5 and
        mali_total <= 6 and
        target_sentence_share >= 0.15 and
        strong_mali <= 3
    ):
        return 3

    if (
        (target_in_headline or target_in_lead)
        and target_total >= 2
        and mali_total >= 1
        and (strong_mali >= 1 or mali_total >= 3)
        and score >= 14
        and (
            target_sentence_count >= 3 or
            target_sentence_share >= 0.18 or
            target_cluster_max_run >= 2 or
            (
                article_sentence_count <= 12 and
                (target_in_headline or target_in_lead) and
                target_total >= 2
            )
        )
        and not (
            non_mali_total >= mali_total * 2 and
            target_sentence_share < 0.25 and
            not target_in_headline
        )
    ):
        return 4

    if (
        mali_total >= 1 and
        (strong_mali >= 1 or mali_total >= 3)
        and (
            target_in_headline or
            target_in_lead or
            target_in_body_multiple or
            mali_linkage >= 1 or
            generic_linkage >= 2 or
            target_sentence_count >= 2
        )
        and score >= 8
    ):
        return 3

    return 2

def relevance_label(code):
    code = safe_int(code, default=-1)
    return {
        1: "not relevant",
        2: "marginal mention only",
        3: "substantively relevant",
        4: "main topic"
    }.get(code, "unknown")

def needs_manual_review(row):
    code = safe_int(row.get("relevance_code"))
    score = safe_int(row.get("relevance_score"))

    target_in_headline = safe_int(row.get("target_hits_headline")) > 0
    target_in_lead = safe_int(row.get("target_hits_lead")) > 0
    repeated_target = safe_int(row.get("target_hits_body")) >= 2
    mixed_geo = safe_int(row.get("mali_hits_total")) > 0 and safe_int(row.get("non_mali_hits_total")) > 0
    mali_linkage = safe_int(row.get("mali_specific_linkage_hits")) >= 1
    generic_linkage = safe_int(row.get("generic_linkage_hits")) >= 1

    if code == 1:
        return 0

    if 8 <= score <= 11:
        return 1

    if mixed_geo and (target_in_headline or target_in_lead or repeated_target or mali_linkage):
        return 1

    if safe_int(row.get("is_bulletin_style")) == 1 and code >= 2:
        return 1

    if code == 2 and (
        target_in_headline or
        target_in_lead or
        repeated_target or
        safe_int(row.get("target_hits_total")) >= 3 or
        mali_linkage
    ):
        return 1

    if code == 3 and safe_int(row.get("target_hits_total")) == 1:
        return 1

    if code in [2, 3] and generic_linkage and not mali_linkage:
        return 1

    if code == 4 and safe_int(row.get("target_sentence_count")) <= 2:
        return 1

    return 0

def build_relevance_note(row):
    note_parts = []

    if safe_int(row.get("in_scope_period")) == 0:
        note_parts.append("out of date scope")
    if safe_int(row.get("target_hits_headline")) > 0:
        note_parts.append("target in headline")
    if safe_int(row.get("target_hits_lead")) > 0:
        note_parts.append("target in lead")
    if safe_int(row.get("target_hits_body")) >= 2:
        note_parts.append("repeated target in body")
    if safe_int(row.get("mali_specific_linkage_hits")) > 0:
        note_parts.append("target linked to Mali-specific actor/state context")
    elif safe_int(row.get("generic_linkage_hits")) > 0:
        note_parts.append("target linked to generic operational/abuse/cooperation context")
    if safe_int(row.get("mali_hits_total")) > 0:
        note_parts.append("Mali context present")
    if safe_int(row.get("strong_mali_focus_hits")) > 0:
        note_parts.append("strong Malian-focus cues present")
    if safe_int(row.get("non_mali_hits_total")) > 0:
        note_parts.append("non-Mali location present")
    if safe_int(row.get("target_sentence_count")) > 0:
        note_parts.append(f"target sentence count={safe_int(row.get('target_sentence_count'))}")
    if safe_int(row.get("target_cluster_max_run")) > 0:
        note_parts.append(f"target max cluster run={safe_int(row.get('target_cluster_max_run'))}")
    if safe_int(row.get("is_bulletin_style")) == 1:
        note_parts.append("bulletin-style article")
    if safe_int(row.get("needs_manual_review")) == 1:
        note_parts.append("manual review suggested")

    return "; ".join(note_parts)

# =========================
# 7. MAIN
# =========================
def main():
    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "article_id": str,
            "outlet_code": str,
            "article_seq": str
        },
        low_memory=False
    )

    required_cols = [
        "article_id", "outlet_code", "article_seq", "outlet", "url",
        "date_iso_full", "date_year", "date_month", "date_day", "date_precision",
        "headline_clean", "lead_clean", "body_postclean"
    ]
    df = ensure_columns(df, required_cols)

    df["article_id"] = df["article_id"].apply(lambda x: re.sub(r"\D", "", safe_str(x)).zfill(6) if safe_str(x) else None)
    df["outlet_code"] = df["outlet_code"].apply(lambda x: re.sub(r"\D", "", safe_str(x)).zfill(2) if safe_str(x) else None)
    df["article_seq"] = df["article_seq"].apply(lambda x: re.sub(r"\D", "", safe_str(x)).zfill(4) if safe_str(x) else None)

    df["in_scope_period"] = df.apply(in_scope_period, axis=1)

    target_scores = df.apply(score_target_segments, axis=1, result_type="expand")
    df = pd.concat([df, target_scores], axis=1)

    mali_scores = df.apply(score_mali_segments, axis=1, result_type="expand")
    df = pd.concat([df, mali_scores], axis=1)

    df["non_mali_hits_total"] = df.apply(score_non_mali_context, axis=1)
    df["strong_mali_focus_hits"] = df.apply(score_strong_mali_focus, axis=1)
    df["mali_specific_linkage_hits"] = df.apply(score_mali_specific_linkage, axis=1)
    df["generic_linkage_hits"] = df.apply(score_generic_linkage, axis=1)

    target_sentence_df = df.apply(target_sentence_metrics, axis=1, result_type="expand")
    df = pd.concat([df, target_sentence_df], axis=1)

    df["is_bulletin_style"] = df.apply(is_bulletin_style, axis=1)
    df["article_sentence_count"] = df.apply(approximate_article_length, axis=1)

    df["relevance_score"] = df.apply(compute_relevance_score, axis=1)
    df["relevance_code"] = df.apply(assign_relevance_code, axis=1)
    df["relevance_label"] = df["relevance_code"].apply(relevance_label)
    df["needs_manual_review"] = df.apply(needs_manual_review, axis=1)
    df["relevance_notes"] = df.apply(build_relevance_note, axis=1)

    output_cols = [
        "article_id",
        "outlet_code",
        "article_seq",
        "outlet",
        "url",
        "date_iso_full",
        "date_year",
        "date_month",
        "date_day",
        "date_precision",
        "headline_clean",

        "in_scope_period",

        "target_hits_headline",
        "target_hits_lead",
        "target_hits_body",
        "target_hits_total",
        "target_types_found",

        "mali_hits_headline",
        "mali_hits_lead",
        "mali_hits_body",
        "mali_hits_total",
        "mali_terms_found",

        "strong_mali_focus_hits",
        "non_mali_hits_total",
        "mali_specific_linkage_hits",
        "generic_linkage_hits",

        "target_sentence_count",
        "target_sentence_share",
        "target_cluster_max_run",
        "target_in_first_third",
        "is_bulletin_style",
        "article_sentence_count",

        "relevance_score",
        "relevance_code",
        "relevance_label",
        "needs_manual_review",
        "relevance_notes"
    ]

    output_cols = [c for c in output_cols if c in df.columns]
    df_out = df[output_cols].copy()

    print("\nStep2 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    print(f"Relevance 1: {(df_out['relevance_code'] == 1).sum()}")
    print(f"Relevance 2: {(df_out['relevance_code'] == 2).sum()}")
    print(f"Relevance 3: {(df_out['relevance_code'] == 3).sum()}")
    print(f"Relevance 4: {(df_out['relevance_code'] == 4).sum()}")
    print(f"Manual review: {df_out['needs_manual_review'].sum()}")

    preview_cols = [
        "article_id",
        "outlet",
        "headline_clean",
        "target_hits_total",
        "mali_hits_total",
        "non_mali_hits_total",
        "target_sentence_count",
        "target_sentence_share",
        "target_cluster_max_run",
        "is_bulletin_style",
        "article_sentence_count",
        "relevance_score",
        "relevance_code",
        "relevance_label",
        "needs_manual_review",
        "relevance_notes"
    ]
    preview_cols = [c for c in preview_cols if c in df_out.columns]

    print(df_out[preview_cols].head(20))

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()