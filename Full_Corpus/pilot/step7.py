import pandas as pd
import re

INPUT_STEP6 = "data/postStep6.csv"
INPUT_STEP1 = "data/postStep1.csv"
OUTPUT_CSV = "data/postStep7.csv"

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
# 2. WG/AC TARGET MARKERS
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
# 3. SIGNAL PATTERNS
# =========================
POSITIVE_STANCE_PATTERNS = [
    r"\befficace\b",
    r"\befficacit[ée]\b",
    r"\br[ée]sultats?\b",
    r"\bmont[ée]e\s+en\s+puissance\b",
    r"\brenforcer\b",
    r"\bstabiliser\b",
    r"\bstabilisation\b",
    r"\bpartenaire\s+historique\b",
    r"\bpartenaire\s+strat[ée]gique\b",
    r"\bpartenaires?\s+russes?\b",
    r"\balli[ée]s?\s+russes?\b",
    r"\bappr[ée]ci[ée]s?\b",
    r"\br[ée]pond\s+efficacement\b",
    r"\bcapacit[ée]s?\s+op[ée]rationnelles?\b",
    r"\bmission\s+accomplie\b",
    r"\bsans\s+incident\s+majeur\b",
    r"\bcoordination\s+inter-forces\b",
    r"\bdispositif\s+de\s+s[ée]curisation\b",
    r"\btotalement\s+ma[iî]tris[ée]\b",
    r"\bsucc[èe]s\s+strat[ée]gique\b",
]

HARD_NEGATIVE_STANCE_PATTERNS = [
    r"\bexactions?\b",
    r"\btorture\b",
    r"\bex[ée]cutions?\b",
    r"\bmassacres?\b",
    r"\bviolations?\s+des?\s+droits\s+de\s+l['’]homme\b",
    r"\bviolence\b",
    r"\boccupants?\b",
    r"\bforces?\s+d['’]occupation\b",
    r"\bd[ée]sastre\b",
]

SOFT_NEGATIVE_STANCE_PATTERNS = [
    r"\bmercenaires?\b",
    r"\bparamilitaires?\b",
    r"\bsulfureux\b",
    r"\baccus[ée]s?\b",
    r"\bcontrovers[ée]?\b",
    r"\bmena[çc]e\b",
]

LEGITIMIZING_PATTERNS = [
    r"\bpartenaire\s+historique\b",
    r"\bpartenaire\s+strat[ée]gique\b",
    r"\bchoix\s+de\s+partenaires\b",
    r"\brespect\s+de\s+la\souverainet[ée]\b",
    r"\br[ée]pond\s+efficacement\b",
    r"\brenforcer\s+les\s+capacit[ée]s?\b",
    r"\bmont[ée]e\s+en\s+puissance\b",
    r"\bappr[ée]ci[ée]s?\b",
    r"\butiles?\b",
    r"\bn[ée]cessaire[s]?\b",
    r"\bpr[ée]sents?\s+pour\s+renforcer\b",
    r"\balli[ée]s?\s+russes?\b",
    r"\bpartenaires?\s+russes?\b",
    r"\bmission\s+accomplie\b",
    r"\bsans\s+incident\s+majeur\b",
    r"\bcoordination\s+inter-forces\b",
    r"\bdispositif\s+de\s+s[ée]curisation\b",
    r"\bsucc[èe]s\s+strat[ée]gique\b",
]

HARD_DELEGITIMIZING_PATTERNS = [
    r"\bexactions?\b",
    r"\bviolations?\s+des?\s+droits\s+de\s+l['’]homme\b",
    r"\btorture\b",
    r"\bex[ée]cutions?\b",
    r"\bmassacres?\b",
    r"\bill[ée]gal\b",
    r"\boccupants?\b",
    r"\bforces?\s+d['’]occupation\b",
    r"\bsupplétifs?\b",
    r"\bpr[ée]dation\b",
    r"\bd[ée]sastre\b",
]

SOFT_DELEGITIMIZING_PATTERNS = [
    r"\bmercenaires?\b",
    r"\bparamilitaires?\b",
    r"\bcontrovers[ée]?\b",
    r"\baccus[aé]?\b",
    r"\bmena[çc]e\b",
]

REPORTING_PATTERNS = [
    r"\bselon\b",
    r"\bd['’]apr[eè]s\b",
    r"\ba déclaré\b",
    r"\ba indiqué\b",
    r"\baffirme\b",
    r"\bexplique\b",
    r"\brapporte\b",
    r"\bcit[ée]?\b",
    r"\bcommuniqu[ée]\b",
    r"\bprétendu\b",
    r"\ba soutenu\b",
    r"\ba estimé\b",
]

# =========================
# 4. TEXT ACCESS
# =========================
def get_analysis_text(row):
    parts = [
        safe_str(row.get("Headline", "")),
        safe_str(row.get("lead_clean", "")),
        safe_str(row.get("body_postclean", ""))
    ]
    parts = [p.strip() for p in parts if safe_str(p).strip()]
    return " ".join(parts).strip()

def extract_wgac_sentences_with_context(row, window=2):
    text = get_analysis_text(row)
    sentences = split_sentences(text)

    if not sentences:
        return "", 0

    selected_indices = set()

    for i, sent in enumerate(sentences):
        if has_any(sent, WGAC_PATTERNS):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            for j in range(start, end):
                selected_indices.add(j)

    if not selected_indices:
        return "", 0

    ordered = [sentences[i] for i in sorted(selected_indices)]
    return " ".join(ordered), len(ordered)

def get_support_texts(row):
    local_text, local_sent_count = extract_wgac_sentences_with_context(row, window=2)
    full_text = get_analysis_text(row)

    if local_text and local_sent_count >= 1:
        return local_text, full_text, "local_wgac_context"

    return full_text, full_text, "full_text_fallback"

# =========================
# 5. WEIGHTED SIGNAL COUNTING
# =========================
def sentence_reporting_weight(sent):
    sent = safe_str(sent)
    if has_any(sent, REPORTING_PATTERNS) or '«' in sent or '»' in sent or '"' in sent:
        return 0.5
    return 1.0

def count_weighted_matches(text, patterns, base_weight=1.0):
    text = safe_str(text)
    if not text:
        return 0.0

    total = 0.0
    sentences = split_sentences(text)

    for sent in sentences:
        base = count_matches(sent, patterns)
        if base == 0:
            continue
        total += base * base_weight * sentence_reporting_weight(sent)

    return total

def count_negative_weighted(text):
    hard = count_weighted_matches(text, HARD_NEGATIVE_STANCE_PATTERNS, base_weight=1.0)
    soft = count_weighted_matches(text, SOFT_NEGATIVE_STANCE_PATTERNS, base_weight=0.5)
    return hard + soft

def count_delegit_weighted(text):
    hard = count_weighted_matches(text, HARD_DELEGITIMIZING_PATTERNS, base_weight=1.0)
    soft = count_weighted_matches(text, SOFT_DELEGITIMIZING_PATTERNS, base_weight=0.5)
    return hard + soft

# =========================
# 6. STANCE SUPPORT
# =========================
def code_stance_support(row):
    primary_text, full_text, source_mode = get_support_texts(row)
    relevance = safe_int(row.get("Relevance"))

    pos = count_weighted_matches(primary_text, POSITIVE_STANCE_PATTERNS, base_weight=1.0)
    neg = count_negative_weighted(primary_text)
    total = pos + neg

    # factual neutrality guard
    if (
        relevance >= 3 and
        safe_int(row.get("Human_Rights_Abuse")) == 0 and
        safe_int(row.get("Western_Failure")) == 0 and
        pos == 0 and
        neg <= 2.5
    ):
        return pos, neg, 2, "limited evaluative negativity without reinforcing frames; treated as neutral/factual"

    if total == 0:
        fallback_pos = count_weighted_matches(full_text, POSITIVE_STANCE_PATTERNS, base_weight=1.0)
        fallback_neg = count_negative_weighted(full_text)

        if fallback_pos + fallback_neg == 0:
            return pos, neg, 5, f"no strong evaluative signals detected ({source_mode})"

        if relevance >= 3:
            if fallback_neg >= 3 and fallback_neg >= fallback_pos + 2:
                return pos, neg, 1, (
                    f"local WG/AC context weak; broader article strongly negative "
                    f"(positive={round(fallback_pos,2)}, negative={round(fallback_neg,2)})"
                )
            if fallback_pos >= 2 and fallback_pos >= fallback_neg + 1:
                return pos, neg, 3, (
                    f"local WG/AC context weak; broader article positive "
                    f"(positive={round(fallback_pos,2)}, negative={round(fallback_neg,2)})"
                )

        return pos, neg, 5, (
            f"no strong evaluative signals in local WG/AC context; broader article has "
            f"positive={round(fallback_pos,2)}, negative={round(fallback_neg,2)}, treated conservatively"
        )

    if pos >= 2 and neg >= 2:
        return pos, neg, 4, f"mixed signals in {source_mode} (positive={round(pos,2)}, negative={round(neg,2)})"

    if neg >= 3 and neg >= pos + 2:
        return pos, neg, 1, f"negative signals dominate in {source_mode} (positive={round(pos,2)}, negative={round(neg,2)})"

    if pos >= 2 and pos >= neg + 1:
        return pos, neg, 3, f"positive signals dominate in {source_mode} (positive={round(pos,2)}, negative={round(neg,2)})"

    if total <= 2 and relevance <= 2:
        return pos, neg, 2, f"low-intensity evaluative signals in marginal text ({source_mode}, positive={round(pos,2)}, negative={round(neg,2)})"

    return pos, neg, 2, f"weak or low-intensity mixed signals in {source_mode} (positive={round(pos,2)}, negative={round(neg,2)})"

# =========================
# 7. AMBIVALENCE SUPPORT
# =========================
def code_ambivalence_support(row):
    pos = safe_float(row.get("Positive_Signal_Count"))
    neg = safe_float(row.get("Negative_Signal_Count"))

    if pos >= 2 and neg >= 2:
        return 1, f"both positive and negative signals present in WG/AC-focused context (positive={round(pos,2)}, negative={round(neg,2)})"

    return 0, f"no strong ambivalence pattern in WG/AC-focused context (positive={round(pos,2)}, negative={round(neg,2)})"

# =========================
# 8. LEGITIMATION SUPPORT
# =========================
def code_legitimation_support(row):
    primary_text, full_text, source_mode = get_support_texts(row)
    relevance = safe_int(row.get("Relevance"))

    pos = count_weighted_matches(primary_text, LEGITIMIZING_PATTERNS, base_weight=1.0)
    neg = count_delegit_weighted(primary_text)
    total = pos + neg

    # factual neutrality guard
    if (
        relevance >= 3 and
        safe_int(row.get("Human_Rights_Abuse")) == 0 and
        pos == 0 and
        neg <= 2.5
    ):
        return pos, neg, 4, "limited delegitimizing signals without reinforcing abuse frame; treated conservatively"

    if total == 0:
        fallback_pos = count_weighted_matches(full_text, LEGITIMIZING_PATTERNS, base_weight=1.0)
        fallback_neg = count_delegit_weighted(full_text)

        if fallback_pos + fallback_neg == 0:
            return pos, neg, 4, f"no strong legitimation signals detected ({source_mode})"

        if relevance >= 3:
            if fallback_neg >= 3 and fallback_neg >= fallback_pos + 2:
                return pos, neg, 1, (
                    f"local WG/AC context weak; broader article strongly delegitimizing "
                    f"(legitimizing={round(fallback_pos,2)}, delegitimizing={round(fallback_neg,2)})"
                )
            if fallback_pos >= 2 and fallback_pos >= fallback_neg + 1:
                return pos, neg, 3, (
                    f"local WG/AC context weak; broader article legitimizing "
                    f"(legitimizing={round(fallback_pos,2)}, delegitimizing={round(fallback_neg,2)})"
                )
            if fallback_pos >= 1 and fallback_neg == 0:
                return pos, neg, 2, (
                    f"local WG/AC context weak; broader article suggests normalization "
                    f"(legitimizing={round(fallback_pos,2)}, delegitimizing={round(fallback_neg,2)})"
                )

        return pos, neg, 4, (
            f"no strong legitimation signals in local WG/AC context; broader article has "
            f"legitimizing={round(fallback_pos,2)}, delegitimizing={round(fallback_neg,2)}, treated conservatively"
        )

    if neg >= 3 and neg >= pos + 2:
        return pos, neg, 1, f"delegitimizing language dominates in {source_mode} (legitimizing={round(pos,2)}, delegitimizing={round(neg,2)})"

    if pos >= 2 and pos >= neg + 1:
        return pos, neg, 3, f"explicit legitimizing language dominates in {source_mode} (legitimizing={round(pos,2)}, delegitimizing={round(neg,2)})"

    if pos >= 1 and neg == 0:
        return pos, neg, 2, f"normalized / implicitly legitimizing language present in {source_mode} (legitimizing={round(pos,2)}, delegitimizing={round(neg,2)})"

    return pos, neg, 4, f"unclear legitimation pattern in {source_mode} (legitimizing={round(pos,2)}, delegitimizing={round(neg,2)})"

# =========================
# 9. REVIEW FLAG
# =========================
def step7_manual_review(row):
    relevance = safe_int(row.get("Relevance"))
    stance = safe_int(row.get("Stance_Support"))
    legit = safe_int(row.get("Legitimation_Support"))
    amb = safe_int(row.get("Ambivalence_Support"))
    pos = safe_float(row.get("Positive_Signal_Count"))
    neg = safe_float(row.get("Negative_Signal_Count"))

def step7_manual_review(row):
    relevance = safe_int(row.get("Relevance"))
    stance = safe_int(row.get("Stance_Support"))
    legit = safe_int(row.get("Legitimation_Support"))
    amb = safe_int(row.get("Ambivalence_Support"))
    pos = safe_float(row.get("Positive_Signal_Count"))
    neg = safe_float(row.get("Negative_Signal_Count"))

    # mixed stance is worth checking
    if stance == 4:
        return 1

    # ambivalence is worth checking
    if amb == 1:
        return 1

    # only review unclear legitimation when article is substantively relevant
    # and some non-trivial legitimation/delegitimation signal exists
    if legit == 4 and relevance >= 3 and (pos + neg) >= 2:
        return 1

    # no strong evaluative signal should not automatically trigger review
    # unless article is highly relevant and frame environment suggests tension
    if stance == 5 and relevance >= 4:
        return 1

    return 0

# =========================
# 10. ROW PROCESSOR
# =========================
def process_row(row):
    out = row.to_dict()

    pos_count, neg_count, stance_code, stance_note = code_stance_support(row)
    out["Positive_Signal_Count"] = round(pos_count, 3)
    out["Negative_Signal_Count"] = round(neg_count, 3)
    out["Stance_Support"] = stance_code
    out["Stance_Support_Note"] = stance_note

    amb_code, amb_note = code_ambivalence_support(out)
    out["Ambivalence_Support"] = amb_code
    out["Ambivalence_Support_Note"] = amb_note

    leg_pos, leg_neg, legit_code, legit_note = code_legitimation_support(row)
    out["Legitimation_Positive_Count"] = round(leg_pos, 3)
    out["Legitimation_Negative_Count"] = round(leg_neg, 3)
    out["Legitimation_Support"] = legit_code
    out["Legitimation_Support_Note"] = legit_note

    return out

# =========================
# 11. MAIN
# =========================
def main():
    df6 = pd.read_csv(
        INPUT_STEP6,
        dtype={"Article_ID": str},
        low_memory=False
    )

    df1 = pd.read_csv(
        INPUT_STEP1,
        dtype={"article_id": str},
        low_memory=False
    )

    df6 = ensure_columns(df6, [
        "Article_ID", "Outlet", "Date", "Headline",
        "Relevance", "Relevance_Label",
        "Actor_Mention", "Successor_Frame",
        "Dominant_Label", "Dominant_Location",
        "Main_Associated_Actor",
        "Counterterrorism", "Sovereignty", "Human_Rights_Abuse",
        "Anti_or_Neocolonialism", "Western_Failure",
        "Security_Effectiveness", "Economic_Interests",
        "Geopolitical_Rivalry", "URL"
    ])

    df1 = ensure_columns(df1, [
        "article_id", "lead_clean", "body_postclean"
    ])

    df6["Article_ID"] = df6["Article_ID"].apply(normalize_article_id)
    df1["article_id"] = df1["article_id"].apply(normalize_article_id)

    merge_cols = [
        "article_id",
        "lead_clean",
        "body_postclean"
    ]
    df = df6.merge(df1[merge_cols], left_on="Article_ID", right_on="article_id", how="left")

    total = len(df)
    processed_rows = []

    print(f"\nStep7: processing {total} articles")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        processed_rows.append(process_row(row))
        print_progress(i, total, every=100)

    df_processed = pd.DataFrame(processed_rows)

    df_processed["Step7_Manual_Review"] = df_processed.apply(step7_manual_review, axis=1)

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

        "Positive_Signal_Count": df_processed["Positive_Signal_Count"],
        "Negative_Signal_Count": df_processed["Negative_Signal_Count"],
        "Stance_Support": df_processed["Stance_Support"],
        "Stance_Support_Note": df_processed["Stance_Support_Note"],

        "Ambivalence_Support": df_processed["Ambivalence_Support"],
        "Ambivalence_Support_Note": df_processed["Ambivalence_Support_Note"],

        "Legitimation_Positive_Count": df_processed["Legitimation_Positive_Count"],
        "Legitimation_Negative_Count": df_processed["Legitimation_Negative_Count"],
        "Legitimation_Support": df_processed["Legitimation_Support"],
        "Legitimation_Support_Note": df_processed["Legitimation_Support_Note"],

        "Step7_Manual_Review": df_processed["Step7_Manual_Review"],
        "URL": df_processed["URL"]
    })

    print("\nStep7 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    print(f"Stance_Support=1: {(df_out['Stance_Support'] == 1).sum()}")
    print(f"Stance_Support=2: {(df_out['Stance_Support'] == 2).sum()}")
    print(f"Stance_Support=3: {(df_out['Stance_Support'] == 3).sum()}")
    print(f"Stance_Support=4: {(df_out['Stance_Support'] == 4).sum()}")
    print(f"Stance_Support=5: {(df_out['Stance_Support'] == 5).sum()}")

    print(f"Ambivalence_Support=1: {(df_out['Ambivalence_Support'] == 1).sum()}")

    print(f"Legitimation_Support=1: {(df_out['Legitimation_Support'] == 1).sum()}")
    print(f"Legitimation_Support=2: {(df_out['Legitimation_Support'] == 2).sum()}")
    print(f"Legitimation_Support=3: {(df_out['Legitimation_Support'] == 3).sum()}")
    print(f"Legitimation_Support=4: {(df_out['Legitimation_Support'] == 4).sum()}")

    print(f"Manual review: {df_out['Step7_Manual_Review'].sum()}")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Stance_Support",
        "Ambivalence_Support",
        "Legitimation_Support",
        "Step7_Manual_Review"
    ]
    print(df_out[preview_cols].head(30))

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()