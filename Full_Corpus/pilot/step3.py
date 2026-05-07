import pandas as pd
import re

INPUT_STEP2 = "data/postStep2.csv"
INPUT_STEP1 = "data/postStep1.csv"
OUTPUT_CSV = "data/postStep3.csv"

# =========================
# 1. HELPERS
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
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def normalize_article_id(x):
    raw = safe_str(x).strip()
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return digits.zfill(6)

def count_matches(text, patterns):
    text = safe_str(text)
    if not text:
        return 0
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, flags=re.IGNORECASE))
    return total

def has_any(text, patterns):
    text = safe_str(text)
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def split_sentences(text):
    text = safe_str(text)
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

def print_progress(i, total, every=100):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"Progress: {i}/{total} ({pct}%)")

# =========================
# 2. PATTERNS
# =========================
WAGNER_PATTERNS = [
    r"\bwagner\b",
    r"\bgroupe\s+wagner\b",
    r"\bwagner\s+group\b",
]

AFRICA_CORPS_PATTERNS = [
    r"\bafrica\s+corps\b",
    r"\bcorps\s+africain\b",
    r"\bcorps\s+africain\s+russe\b",
]

INDIRECT_RUSSIAN_PATTERNS = [
    r"\bmercenaires?\s+russes?\b",
    r"\bparamilitaires?\s+russes?\b",
    r"\binstructeurs?\s+russes?\b",
    r"\bformateurs?\s+russes?\b",
    r"\bcoop[ée]rants?\s+russes?\b",
    r"\bforces?\s+russes?\b",
    r"\bmilitaires?\s+russes?\b",
]

SUCCESSOR_PATTERNS = [
    r"\bsuccesseur\b",
    r"\bsuccède\b",
    r"\bremplace(?:r|ment)?\b",
    r"\bapr[eè]s\s+wagner\b",
    r"\bpost-wagner\b",
    r"\bex-wagner\b",
    r"\bh[ée]ritier\b",
    r"\bcontinuation\b",
    r"\bnouvelle\s+structure\b",
    r"\bprend\s+le\s+relais\b",
    r"\bréorganis[ée]\s+au\s+sein\b",
    r"\bnouvel\s+avatar\b",
    r"\bnouvelle\s+marque\b",
    r"\bdémantel[ée]?\b",
    r"\bappartiendra\s+bient[oô]t\s+[àa]\s+l['’]histoire\b",
]

LABEL_PATTERNS = {
    1: [  # mercenaries
        r"\bmercenaires?\b",
        r"\bmercenariat\b",
    ],
    2: [  # instructors/advisers
        r"\binstructeurs?\b",
        r"\bformateurs?\b",
        r"\bconseillers?\b",
        r"\bcoop[ée]rants?\b",
        r"\badvisers?\b",
        r"\badvisors?\b",
    ],
    3: [  # allies/partners
        r"\balli[ée]s?\b",
        r"\balli[ée]s?\s+russes?\b",
        r"\bpartenaires?\s+russes?\b",
        r"\bpartenaire\s+historique\b",
        r"\bpartenaire\s+strat[ée]gique\b",
    ],
    4: [  # foreign/occupying forces
        r"\bforces?\s+[ée]trang[èe]res?\b",
        r"\bforces?\s+d['’]occupation\b",
        r"\boccupants?\b",
        r"\bmilice\b",
        r"\bmilices\b",
    ],
    5: [  # neutral designation
        r"\bsoci[ée]t[ée]\s+de\s+s[ée]curit[ée]\s+priv[ée]e?\b",
        r"\bentreprise\s+militaire\s+priv[ée]e?\b",
        r"\bgroupe\s+paramilitaire\b",
        r"\bsoci[ée]t[ée]\s+russe\b",
        r"\bpmc\b",
        r"\bgroupe\b",
    ]
}

# =========================
# 3. TEXT ACCESS
# =========================
def get_analysis_text(row):
    parts = [
        safe_str(row.get("headline_clean", "")),
        safe_str(row.get("lead_clean", "")),
        safe_str(row.get("body_postclean", ""))
    ]
    parts = [p.strip() for p in parts if safe_str(p).strip()]
    return " ".join(parts).strip()

def extract_target_context(text, window=2):
    sentences = split_sentences(text)
    if not sentences:
        return ""

    selected = set()
    target_patterns = WAGNER_PATTERNS + AFRICA_CORPS_PATTERNS + INDIRECT_RUSSIAN_PATTERNS

    for i, sent in enumerate(sentences):
        if has_any(sent, target_patterns):
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                selected.add(j)

    if not selected:
        return text

    return " ".join([sentences[i] for i in sorted(selected)])

# =========================
# 4. ACTOR MENTION
# =========================
def code_actor_mention(row):
    text = get_analysis_text(row)

    has_wagner = has_any(text, WAGNER_PATTERNS)
    has_ac = has_any(text, AFRICA_CORPS_PATTERNS)
    has_indirect = has_any(text, INDIRECT_RUSSIAN_PATTERNS)

    if has_wagner and has_ac:
        return 3, "both explicitly mentioned"
    if has_wagner:
        return 1, "Wagner explicitly mentioned"
    if has_ac:
        return 2, "Africa Corps explicitly mentioned"
    if has_indirect:
        return 4, "only indirect Russian military-contractor terminology found"
    return 5, "cannot be determined"

# =========================
# 5. SUCCESSOR FRAME
# =========================
def code_successor_frame(row):
    text = get_analysis_text(row)

    has_wagner = has_any(text, WAGNER_PATTERNS)
    has_ac = has_any(text, AFRICA_CORPS_PATTERNS)
    has_successor_language = has_any(text, SUCCESSOR_PATTERNS)

    if has_ac and has_wagner and has_successor_language:
        return 1, "Africa Corps linked to Wagner through successor/replacement language"

    if has_ac and has_wagner:
        if re.search(r"\bwagner\b.*\bafrica\s+corps\b", text, flags=re.IGNORECASE | re.DOTALL):
            return 1, "Wagner and Africa Corps co-mentioned in transition/reorganization context"
        if re.search(r"\bafrica\s+corps\b.*\bwagner\b", text, flags=re.IGNORECASE | re.DOTALL):
            return 1, "Africa Corps and Wagner co-mentioned in transition/reorganization context"

    return 0, "no explicit successor framing detected"

# =========================
# 6. DOMINANT LABEL
# =========================
def weighted_label_score(text, patterns):
    sentences = split_sentences(text)
    if not sentences:
        return 0.0

    score = 0.0
    for i, sent in enumerate(sentences):
        hits = count_matches(sent, patterns)
        if hits == 0:
            continue

        weight = 1.0
        if i == 0:
            weight = 2.0
        elif i <= 2:
            weight = 1.5

        score += hits * weight

    return score

def code_dominant_label(row):
    text_full = get_analysis_text(row)
    text_local = extract_target_context(text_full, window=2)

    label_scores = {}
    for label_code, patterns in LABEL_PATTERNS.items():
        local_score = weighted_label_score(text_local, patterns)
        full_score = weighted_label_score(text_full, patterns) * 0.25
        score = local_score + full_score

        if label_code == 3:
            score = score * 0.85

        label_scores[label_code] = score

    positive_scores = {k: v for k, v in label_scores.items() if v > 0}

    if not positive_scores:
        return 5, "no strong characterizing label found; coded as neutral designation"

    max_score = max(positive_scores.values())
    top_labels = [k for k, v in positive_scores.items() if abs(v - max_score) < 0.01]

    if len(top_labels) > 1:
        return 6, f"multiple labels tied: {top_labels}"

    chosen = top_labels[0]

    notes_map = {
        1: "mercenary language dominates",
        2: "instructor/adviser language dominates",
        3: "ally/partner language dominates",
        4: "foreign/occupying-force language dominates",
        5: "neutral designation dominates",
        6: "multiple labels, no clear dominance"
    }

    return chosen, notes_map.get(chosen, "label coded")

# =========================
# 7. REVIEW FLAG
# =========================
def step3_manual_review(row):
    actor_mention = safe_int(row.get("Actor_Mention"))
    successor_frame = safe_int(row.get("Successor_Frame"))
    dominant_label = safe_int(row.get("Dominant_Label"))

    if actor_mention == 5:
        return 1
    if dominant_label == 6:
        return 1
    if actor_mention == 2 and successor_frame == 0:
        return 1
    return 0

# =========================
# 8. ROW PROCESSOR
# =========================
def process_row(row):
    actor_code, actor_note = code_actor_mention(row)
    succ_code, succ_note = code_successor_frame(row)
    label_code, label_note = code_dominant_label(row)

    out = row.to_dict()
    out["Actor_Mention"] = actor_code
    out["Actor_Mention_Note"] = actor_note
    out["Successor_Frame"] = succ_code
    out["Successor_Frame_Note"] = succ_note
    out["Dominant_Label"] = label_code
    out["Dominant_Label_Note"] = label_note
    return out

# =========================
# 9. MAIN
# =========================
def main():
    df2 = pd.read_csv(
        INPUT_STEP2,
        dtype={
            "article_id": str,
            "outlet_code": str,
            "article_seq": str
        },
        low_memory=False
    )

    df1 = pd.read_csv(
        INPUT_STEP1,
        dtype={
            "article_id": str,
            "outlet_code": str,
            "article_seq": str
        },
        low_memory=False
    )

    df2 = ensure_columns(df2, [
        "article_id", "outlet", "date_iso_full", "headline_clean",
        "relevance_code", "relevance_label", "url"
    ])
    df1 = ensure_columns(df1, [
        "article_id", "lead_clean", "body_postclean"
    ])

    df2["article_id"] = df2["article_id"].apply(normalize_article_id)
    df1["article_id"] = df1["article_id"].apply(normalize_article_id)

    df2 = df2[df2["relevance_code"].apply(lambda x: safe_int(x) in [2, 3, 4])].copy()

    merge_cols = ["article_id", "lead_clean", "body_postclean"]
    df = df2.merge(df1[merge_cols], on="article_id", how="left")

    total = len(df)
    processed_rows = []

    print(f"\nStep3: processing {total} articles")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        processed_rows.append(process_row(row))
        print_progress(i, total, every=100)

    df_processed = pd.DataFrame(processed_rows)

    df_processed["Step3_Manual_Review"] = df_processed.apply(step3_manual_review, axis=1)

    df_out = pd.DataFrame({
        "Article_ID": df_processed["article_id"],
        "Outlet": df_processed["outlet"],
        "Date": df_processed["date_iso_full"],
        "Headline": df_processed["headline_clean"],
        "Relevance": df_processed["relevance_code"],
        "Relevance_Label": df_processed["relevance_label"],
        "Actor_Mention": df_processed["Actor_Mention"],
        "Actor_Mention_Note": df_processed["Actor_Mention_Note"],
        "Successor_Frame": df_processed["Successor_Frame"],
        "Successor_Frame_Note": df_processed["Successor_Frame_Note"],
        "Dominant_Label": df_processed["Dominant_Label"],
        "Dominant_Label_Note": df_processed["Dominant_Label_Note"],
        "Step3_Manual_Review": df_processed["Step3_Manual_Review"],
        "URL": df_processed["url"]
    })

    print("\nStep3 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    print(f"Actor_Mention=1 (Wagner): {(df_out['Actor_Mention'] == 1).sum()}")
    print(f"Actor_Mention=2 (Africa Corps): {(df_out['Actor_Mention'] == 2).sum()}")
    print(f"Actor_Mention=3 (Both): {(df_out['Actor_Mention'] == 3).sum()}")
    print(f"Actor_Mention=4 (Indirect only): {(df_out['Actor_Mention'] == 4).sum()}")
    print(f"Actor_Mention=5 (Cannot determine): {(df_out['Actor_Mention'] == 5).sum()}")
    print(f"Successor_Frame=1: {(df_out['Successor_Frame'] == 1).sum()}")
    print(f"Dominant_Label=6: {(df_out['Dominant_Label'] == 6).sum()}")
    print(f"Manual review: {df_out['Step3_Manual_Review'].sum()}")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Actor_Mention",
        "Successor_Frame",
        "Dominant_Label",
        "Step3_Manual_Review"
    ]
    print(df_out[preview_cols].head(30))

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()