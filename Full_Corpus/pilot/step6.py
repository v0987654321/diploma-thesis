import pandas as pd
import re

INPUT_STEP5 = "data/postStep5.csv"
INPUT_STEP1 = "data/postStep1.csv"
OUTPUT_CSV = "data/postStep6.csv"

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
    sentences = re.split(r"(?<=[\.\?\!;:])\s+", text)
    return [s.strip() for s in sentences if s.strip()]

def print_progress(i, total, every=100):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"Progress: {i}/{total} ({pct}%)")

# =========================
# 2. WG/AC MARKERS
# =========================
WGAC_PATTERNS = [
    r"\bwagner\b",
    r"\bgroupe\s+wagner\b",
    r"\bwagner\s+group\b",
    r"\bafrica\s+corps\b",
    r"\bcorps\s+africain\b",
    r"\bcorps\s+africain\s+russe\b",
    r"\bmercenaires?\s+russes?\b",
    r"\bparamilitaires?\s+russes?\b",
    r"\binstructeurs?\s+russes?\b",
    r"\bformateurs?\s+russes?\b",
    r"\bcoop[ée]rants?\s+russes?\b",
]

# =========================
# 3. TEXT ACCESS
# =========================
def get_analysis_text(row):
    parts = [
        safe_str(row.get("Headline", "")),
        safe_str(row.get("lead_clean", "")),
        safe_str(row.get("body_postclean", ""))
    ]
    parts = [p.strip() for p in parts if safe_str(p).strip()]
    return " ".join(parts).strip()

def extract_target_context_sentences(text, window=2):
    sentences = split_sentences(text)

    if not sentences:
        return ""

    selected_indices = set()

    for i, sent in enumerate(sentences):
        if has_any(sent, WGAC_PATTERNS):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            for j in range(start, end):
                selected_indices.add(j)

    if not selected_indices:
        return ""

    ordered = [sentences[i] for i in sorted(selected_indices)]
    return " ".join(ordered)

def is_bulletin_like(row):
    headline = safe_str(row.get("Headline", "")).lower()
    body = safe_str(row.get("body_postclean", ""))
    lead = safe_str(row.get("lead_clean", ""))

    if "les titres du" in headline:
        return 1
    if body.count("👉") + lead.count("👉") >= 3:
        return 1
    return 0

# =========================
# 4. FRAME PATTERNS
# =========================
FRAME_PATTERNS = {
    "Counterterrorism": [
        r"\bterrorisme\b",
        r"\bterroriste[s]?\b",
        r"\bjihadistes?\b",
        r"\banti-terroriste\b",
        r"\blutte\s+contre\s+le\s+terrorisme\b",
        r"\blutte\s+antijihadiste\b",
        r"\binsurg[ée]s?\b",
        r"\bgroupes?\s+arm[ée]s?\b",
        r"\bgsim\b",
        r"\bjnim\b",
        r"\baqmi\b",
        r"\beis\b",
        r"\betat\s+islamique\b",
    ],
    "Sovereignty": [
        r"\bsouverainet[ée]\b",
        r"\bpays\s+souverain\b",
        r"\betat\s+souverain\b",
        r"\brespect\s+de\s+la\souverainet[ée]\b",
        r"\bchoix\s+de\s+partenaires\b",
        r"\bchoix\s+strat[ée]giques\b",
        r"\bpartenaire\s+historique\b",
        r"\bautonomie\s+strat[ée]gique\b",
        r"\bint[ée]r[êe]ts\s+vitaux\b",
        r"\bne\s+justifierons\s+plus\s+notre\s+choix\s+de\s+partenaire\b",
    ],
    "Human_Rights_Abuse": [
        r"\bviolations?\s+des?\s+droits\s+de\s+l['’]homme\b",
        r"\bviolations?\s+r[ée]p[ée]t[ée]es?\b",
        r"\bviolations?\s+manifestes?\b",
        r"\bexactions?\b",
        r"\btorture\b",
        r"\bex[ée]cutions?\b",
        r"\bmassacres?\b",
        r"\br[ée]pression\b",
        r"\bviolence\s+contre\s+les\s+civils\b",
        r"\babus\s+contre\s+les\s+civils\b",
        r"\bextrajudiciaires?\b",
        r"\bd[ée]tentions?\s+arbitraires?\b",
        r"\bdisparitions?\s+forc[ée]es?\b",
        r"\benl[eè]vements?\b",
        r"\bpopulations?\s+civiles?\b",
    ],
    "Anti_or_Neocolonialism": [
        r"\bcolonial\b",
        r"\bcoloniale?\b",
        r"\bpuissance\s+coloniale\b",
        r"\bn[ée]ocolonial\b",
        r"\bn[ée]ocolonialisme\b",
        r"\bimp[ée]rialisme\b",
        r"\banticolonial\b",
        r"\bfran[çc]afrique\b",
        r"\blib[ée]ration\b",
        r"\b[ée]mancipation\b",
    ],
    "Western_Failure": [
        r"\b[ée]chec\b",
        r"\b[ée]chec\s+de\s+la\s+france\b",
        r"\babandon\b",
        r"\bhypocrisie\b",
        r"\binefficace\b",
        r"\bincapable\b",
        r"\bdouble\s+standard\b",
        r"\boccidentaux?\s+ont\s+[ée]chou[ée]\b",
        r"\bretrait\s+pr[ée]cipit[ée]\b",
        r"\bincompatible\s+avec\s+sa\s+pr[ée]sence\b",
        r"\bperte\s+de\s+souverainet[ée]\b",
        r"\b[ée]chec\s+sur\s+le\s+plan\s+op[ée]rationnel\b",
    ],
    "Security_Effectiveness": [
        r"\befficacit[ée]\b",
        r"\befficace\b",
        r"\bcapacit[ée]s?\s+op[ée]rationnelles?\b",
        r"\bcapacit[ée]\s+de\s+combat\b",
        r"\bstabiliser\b",
        r"\bstabilisation\b",
        r"\bsucc[èe]s\b",
        r"\br[ée]sultats?\b",
        r"\bmont[ée]e\s+en\s+puissance\b",
        r"\bperformance\s+s[ée]curitaire\b",
        r"\bmission\s+accomplie\b",
        r"\bsans\s+incident\s+majeur\b",
        r"\bcoordination\s+inter-forces\b",
        r"\bdispositif\s+de\s+s[ée]curisation\b",
        r"\btotalement\s+ma[iî]tris[ée]\b",
        r"\bsucc[èe]s\s+strat[ée]gique\b",
    ],
    "Economic_Interests": [
        r"\bmine[s]?\b",
        r"\bminier[s]?\b",
        r"\bsite[s]?\s+miniers?\b",
        r"\bgisements?\b",
        r"\bressources?\s+naturelles?\b",
        r"\bcontrat[s]?\s+miniers?\b",
    ],
    "Geopolitical_Rivalry": [
        r"\bg[ée]opolitique\b",
        r"\brivalit[ée]\b",
        r"\bcomp[ée]tition\b",
        r"\bmonde\s+multipolaire\b",
        r"\bmultipola(?:rit[ée]|ire)\b",
        r"\binfluence\b",
        r"\bbras\s+de\s+fer\b",
        r"\bd[ée]clin\s+fulgurant\b",
        r"\bpartenaires?\s+internationaux\b",
        r"\boccident\b",
        r"\boccidentaux\b",
        r"\botan\b",
        r"\brussie\b",
        r"\bfrance\b",
        r"\betats?-unis\b",
        r"\bunion\s+europ[ée]enne\b",
    ],
}

# =========================
# 5. THRESHOLDS
# =========================
FRAME_THRESHOLDS = {
    "Counterterrorism": 2,
    "Sovereignty": 2,
    "Human_Rights_Abuse": 2,
    "Anti_or_Neocolonialism": 3,
    "Western_Failure": 2,
    "Security_Effectiveness": 2,
    "Economic_Interests": 4,
    "Geopolitical_Rivalry": 3,
}

# =========================
# 6. FRAME SCORING
# =========================
def score_frame_by_segments(row, patterns):
    headline = safe_str(row.get("Headline", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body_full = safe_str(row.get("body_postclean", ""))
    body_local = extract_target_context_sentences(body_full, window=2)

    h = count_matches(headline, patterns)
    l = count_matches(lead, patterns)
    b = count_matches(body_local, patterns)

    segment_score = (h * 3) + (l * 2) + (b * 1)
    return segment_score, h, l, b

def frame_proximity_bonus(row, frame_name, patterns):
    text = get_analysis_text(row)
    bonus = 0
    sentences = split_sentences(text)

    for sent in sentences:
        if has_any(sent, WGAC_PATTERNS) and has_any(sent, patterns):
            if frame_name == "Human_Rights_Abuse":
                bonus += 3
            else:
                bonus += 2

    return bonus

def anti_neocolonial_core_term_present(row):
    text = get_analysis_text(row)
    local = extract_target_context_sentences(text, window=2)
    core_terms = [
        r"\bcolonial\b",
        r"\bcoloniale?\b",
        r"\bpuissance\s+coloniale\b",
        r"\bn[ée]ocolonial\b",
        r"\bn[ée]ocolonialisme\b",
        r"\bimp[ée]rialisme\b",
        r"\banticolonial\b",
        r"\bfran[çc]afrique\b",
    ]
    return 1 if has_any(local, core_terms) else 0

def geopolitical_core_term_present(row):
    text = get_analysis_text(row)
    local = extract_target_context_sentences(text, window=2)
    explicit_geo_terms = [
        r"\bg[ée]opolitique\b",
        r"\brivalit[ée]\b",
        r"\bcomp[ée]tition\b",
        r"\bmonde\s+multipolaire\b",
        r"\bmultipola(?:rit[ée]|ire)\b",
        r"\binfluence\b",
        r"\bbras\s+de\s+fer\b",
    ]
    return 1 if has_any(local, explicit_geo_terms) else 0

def economic_core_term_present(row):
    text = get_analysis_text(row)
    local = extract_target_context_sentences(text, window=2)
    core_terms = [
        r"\bmine[s]?\b",
        r"\bminier[s]?\b",
        r"\bsite[s]?\s+miniers?\b",
        r"\bgisements?\b",
        r"\bressources?\s+naturelles?\b",
        r"\bcontrat[s]?\s+miniers?\b",
    ]
    return 1 if has_any(local, core_terms) else 0

def code_frame(row, frame_name, patterns):
    local_text = extract_target_context_sentences(get_analysis_text(row), window=2)
    local_sent_count = len(split_sentences(local_text))
    relevance = safe_int(row.get("Relevance"))

    if is_bulletin_like(row) == 1 and relevance == 2:
        if local_sent_count <= 2 and frame_name in ["Human_Rights_Abuse", "Counterterrorism"]:
            return 0, f"{frame_name}: suppressed in bulletin-style article with weak local target context"

    if frame_name == "Counterterrorism" and relevance == 2 and local_sent_count <= 3:
        return 0, (
            f"{frame_name}: suppressed in marginal relevance article "
            f"(local_sent_count={local_sent_count})"
        )

    segment_score, h, l, b = score_frame_by_segments(row, patterns)
    prox_bonus = frame_proximity_bonus(row, frame_name, patterns)
    total_score = segment_score + prox_bonus
    threshold = FRAME_THRESHOLDS[frame_name]

    effective_threshold = threshold
    if relevance == 2:
        effective_threshold += 2

    if frame_name == "Anti_or_Neocolonialism":
        core_present = anti_neocolonial_core_term_present(row)
        value = 1 if (total_score >= effective_threshold and core_present == 1) else 0
        note = (
            f"{frame_name}: headline={h}, lead={l}, body_local={b}, "
            f"segment_score={segment_score}, proximity_bonus={prox_bonus}, "
            f"total_score={total_score}, threshold={effective_threshold}, "
            f"core_term_present={core_present}, coded={value}"
        )
        return value, note

    if frame_name == "Geopolitical_Rivalry":
        core_present = geopolitical_core_term_present(row)
        value = 1 if (total_score >= effective_threshold and core_present == 1) else 0
        note = (
            f"{frame_name}: headline={h}, lead={l}, body_local={b}, "
            f"segment_score={segment_score}, proximity_bonus={prox_bonus}, "
            f"total_score={total_score}, threshold={effective_threshold}, "
            f"explicit_geo_term_present={core_present}, coded={value}"
        )
        return value, note

    if frame_name == "Economic_Interests":
        core_present = economic_core_term_present(row)
        value = 1 if (total_score >= effective_threshold and core_present == 1) else 0
        note = (
            f"{frame_name}: headline={h}, lead={l}, body_local={b}, "
            f"segment_score={segment_score}, proximity_bonus={prox_bonus}, "
            f"total_score={total_score}, threshold={effective_threshold}, "
            f"economic_core_present={core_present}, coded={value}"
        )
        return value, note

    value = 1 if total_score >= effective_threshold else 0
    note = (
        f"{frame_name}: headline={h}, lead={l}, body_local={b}, "
        f"segment_score={segment_score}, proximity_bonus={prox_bonus}, "
        f"total_score={total_score}, threshold={effective_threshold}, coded={value}"
    )
    return value, note

# =========================
# 7. NOTES & REVIEW
# =========================
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

def build_frame_note(row):
    active = [col for col in FRAME_COLS if safe_int(row.get(col)) == 1]
    if not active:
        return "no coded frame triggered"
    return ", ".join(active)

def step6_manual_review(row):
    active_count = sum([safe_int(row.get(col)) for col in FRAME_COLS])
    relevance = safe_int(row.get("Relevance"))

    if relevance >= 3 and active_count == 0:
        return 1

    if active_count >= 4:
        return 1

    if relevance == 2 and active_count >= 2:
        return 1

    return 0

# =========================
# 8. ROW PROCESSOR
# =========================
def process_row(row):
    out = row.to_dict()

    for frame_name, patterns in FRAME_PATTERNS.items():
        value, note = code_frame(row, frame_name, patterns)
        out[frame_name] = value
        out[f"{frame_name}_Note"] = note

    return out

# =========================
# 9. MAIN
# =========================
def main():
    df5 = pd.read_csv(
        INPUT_STEP5,
        dtype={"Article_ID": str},
        low_memory=False
    )

    df1 = pd.read_csv(
        INPUT_STEP1,
        dtype={"article_id": str},
        low_memory=False
    )

    df5 = ensure_columns(df5, [
        "Article_ID", "Outlet", "Date", "Headline",
        "Relevance", "Relevance_Label",
        "Actor_Mention", "Successor_Frame",
        "Dominant_Label", "Dominant_Location",
        "Main_Associated_Actor", "URL"
    ])

    df1 = ensure_columns(df1, [
        "article_id", "lead_clean", "body_postclean"
    ])

    df5["Article_ID"] = df5["Article_ID"].apply(normalize_article_id)
    df1["article_id"] = df1["article_id"].apply(normalize_article_id)

    merge_cols = [
        "article_id",
        "lead_clean",
        "body_postclean"
    ]
    df = df5.merge(df1[merge_cols], left_on="Article_ID", right_on="article_id", how="left")

    total = len(df)
    processed_rows = []

    print(f"\nStep6: processing {total} articles")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        processed_rows.append(process_row(row))
        print_progress(i, total, every=100)

    df_processed = pd.DataFrame(processed_rows)

    df_processed["Frame_Note"] = df_processed.apply(build_frame_note, axis=1)
    df_processed["Step6_Manual_Review"] = df_processed.apply(step6_manual_review, axis=1)

    df_out = pd.DataFrame({
        "Article_ID": df_processed["Article_ID"],
        "Outlet": df_processed["Outlet"],
        "Date": df_processed["Date"],
        "Headline": df_processed["Headline"],
        "Relevance": df_processed["Relevance"],
        "Relevance_Label": df_processed["Relevance_Label"],
        "Actor_Mention": df_processed["Actor_Mention"],
        "Successor_Frame": df_processed["Successor_Frame"],
        "Dominant_Label": df_processed["Dominant_Label"],
        "Dominant_Location": df_processed["Dominant_Location"],
        "Main_Associated_Actor": df_processed["Main_Associated_Actor"],

        "Counterterrorism": df_processed["Counterterrorism"],
        "Sovereignty": df_processed["Sovereignty"],
        "Human_Rights_Abuse": df_processed["Human_Rights_Abuse"],
        "Anti_or_Neocolonialism": df_processed["Anti_or_Neocolonialism"],
        "Western_Failure": df_processed["Western_Failure"],
        "Security_Effectiveness": df_processed["Security_Effectiveness"],
        "Economic_Interests": df_processed["Economic_Interests"],
        "Geopolitical_Rivalry": df_processed["Geopolitical_Rivalry"],

        "Frame_Note": df_processed["Frame_Note"],
        "Step6_Manual_Review": df_processed["Step6_Manual_Review"],
        "URL": df_processed["URL"]
    })

    print("\nStep6 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    for col in FRAME_COLS:
        print(f"{col}=1: {(df_out[col] == 1).sum()}")
    print(f"Manual review: {df_out['Step6_Manual_Review'].sum()}")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Counterterrorism",
        "Sovereignty",
        "Human_Rights_Abuse",
        "Anti_or_Neocolonialism",
        "Western_Failure",
        "Security_Effectiveness",
        "Economic_Interests",
        "Geopolitical_Rivalry",
        "Step6_Manual_Review"
    ]
    print(df_out[preview_cols].head(30))

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()