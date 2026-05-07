import pandas as pd
import re

INPUT_STEP2 = "data/postStep2.xlsx"
INPUT_STEP1 = "data/postStep1.xlsx"
OUTPUT_XLSX = "data/postStep3.xlsx"
OUTPUT_CSV = "data/postStep3.csv"

# =========================
# 1. HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

def count_matches(text, patterns):
    if not text:
        return 0
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, flags=re.IGNORECASE))
    return total

def has_any(text, patterns):
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def split_sentences(text):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

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
    return " ".join([p for p in parts if p]).strip()

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
        return 0

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

        # soften ally/partner as a dominant fallback category
        if label_code == 3:
            score = score * 0.85

        label_scores[label_code] = score

    positive_scores = {k: v for k, v in label_scores.items() if v > 0}

    if not positive_scores:
        return 5, "no strong characterizing label found; coded as neutral designation"

    max_score = max(positive_scores.values())
    top_labels = [k for k, v in positive_scores.items() if abs(v - max_score) < 0.01]

    if len(top_labels) > 1:
        # prefer more explicit characterizing labels over neutral fallback
        preferred_order = [1, 2, 4, 3, 5]
        for pref in preferred_order:
            if pref in top_labels:
                if len(top_labels) > 1:
                    return 6, f"multiple labels tied: {top_labels}"
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
# 7. OPTIONAL REVIEW FLAG
# =========================
def step3_manual_review(row):
    if row["actor_mention"] == 5:
        return 1
    if row["dominant_label"] == 6:
        return 1
    if row["actor_mention"] == 2 and row["successor_frame"] == 0:
        return 1
    return 0

# =========================
# 8. MAIN
# =========================
def main():
    df2 = pd.read_excel(
        INPUT_STEP2,
        dtype={
            "article_id": str,
            "outlet_code": str,
            "article_seq": str
        }
    )

    df1 = pd.read_excel(
        INPUT_STEP1,
        dtype={
            "article_id": str,
            "outlet_code": str,
            "article_seq": str
        }
    )

    for df in [df1, df2]:
        df["article_id"] = df["article_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    df2 = df2[df2["relevance_code"].isin([2, 3, 4])].copy()

    merge_cols = [
        "article_id",
        "lead_clean",
        "body_postclean"
    ]
    df = df2.merge(df1[merge_cols], on="article_id", how="left")

    actor_results = df.apply(code_actor_mention, axis=1)
    df["actor_mention"] = actor_results.apply(lambda x: x[0])
    df["actor_mention_note"] = actor_results.apply(lambda x: x[1])

    successor_results = df.apply(code_successor_frame, axis=1)
    df["successor_frame"] = successor_results.apply(lambda x: x[0])
    df["successor_frame_note"] = successor_results.apply(lambda x: x[1])

    label_results = df.apply(code_dominant_label, axis=1)
    df["dominant_label"] = label_results.apply(lambda x: x[0])
    df["dominant_label_note"] = label_results.apply(lambda x: x[1])

    df["step3_manual_review"] = df.apply(step3_manual_review, axis=1)

    df_out = pd.DataFrame({
        "Article_ID": df["article_id"],
        "Outlet": df["outlet"],
        "Date": df["date_iso_full"],
        "Headline": df["headline_clean"],
        "Relevance": df["relevance_code"],
        "Relevance_Label": df["relevance_label"],
        "Actor_Mention": df["actor_mention"],
        "Actor_Mention_Note": df["actor_mention_note"],
        "Successor_Frame": df["successor_frame"],
        "Successor_Frame_Note": df["successor_frame_note"],
        "Dominant_Label": df["dominant_label"],
        "Dominant_Label_Note": df["dominant_label_note"],
        "Step3_Manual_Review": df["step3_manual_review"],
        "URL": df["url"]
    })

    df_out.to_excel(OUTPUT_XLSX, index=False)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

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
    print(f"\nSaved to {OUTPUT_XLSX}")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()