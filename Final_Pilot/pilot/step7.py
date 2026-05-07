import pandas as pd
import re

INPUT_STEP7 = "data/postStep6.xlsx"
INPUT_STEP1 = "data/postStep1.xlsx"
OUTPUT_XLSX = "data/postStep7.xlsx"
OUTPUT_CSV = "data/postStep7.csv"

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
    text = re.sub(r"\s+", " ", str(text)).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

def get_analysis_text(row):
    parts = [
        safe_str(row.get("Headline", "")),
        safe_str(row.get("lead_clean", "")),
        safe_str(row.get("body_postclean", ""))
    ]
    return " ".join([p for p in parts if p]).strip()

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
# 4. LOCAL WG/AC CONTEXT EXTRACTION
# =========================
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
    if has_any(sent, REPORTING_PATTERNS) or '«' in sent or '»' in sent or '"' in sent:
        return 0.5
    return 1.0

def count_weighted_matches(text, patterns, base_weight=1.0):
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
    relevance = row["Relevance"]

    pos = count_weighted_matches(primary_text, POSITIVE_STANCE_PATTERNS, base_weight=1.0)
    neg = count_negative_weighted(primary_text)
    total = pos + neg

    # factual neutrality guard
    if (
        relevance >= 3 and
        row.get("Human_Rights_Abuse", 0) == 0 and
        row.get("Western_Failure", 0) == 0 and
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
                    f"(positive={fallback_pos}, negative={fallback_neg})"
                )
            if fallback_pos >= 2 and fallback_pos >= fallback_neg + 1:
                return pos, neg, 3, (
                    f"local WG/AC context weak; broader article positive "
                    f"(positive={fallback_pos}, negative={fallback_neg})"
                )

        return pos, neg, 5, (
            f"no strong evaluative signals in local WG/AC context; broader article has "
            f"positive={fallback_pos}, negative={fallback_neg}, treated conservatively"
        )

    if pos >= 2 and neg >= 2:
        return pos, neg, 4, f"mixed signals in {source_mode} (positive={pos}, negative={neg})"

    if neg >= 3 and neg >= pos + 2:
        return pos, neg, 1, f"negative signals dominate in {source_mode} (positive={pos}, negative={neg})"

    if pos >= 2 and pos >= neg + 1:
        return pos, neg, 3, f"positive signals dominate in {source_mode} (positive={pos}, negative={neg})"

    if total <= 2 and relevance <= 2:
        return pos, neg, 2, f"low-intensity evaluative signals in marginal text ({source_mode}, positive={pos}, negative={neg})"

    return pos, neg, 2, f"weak or low-intensity mixed signals in {source_mode} (positive={pos}, negative={neg})"

# =========================
# 7. AMBIVALENCE SUPPORT
# =========================
def code_ambivalence_support(row):
    pos = row["Positive_Signal_Count"]
    neg = row["Negative_Signal_Count"]

    if pos >= 2 and neg >= 2:
        return 1, f"both positive and negative signals present in WG/AC-focused context (positive={pos}, negative={neg})"

    return 0, f"no strong ambivalence pattern in WG/AC-focused context (positive={pos}, negative={neg})"

# =========================
# 8. LEGITIMATION SUPPORT
# =========================
def code_legitimation_support(row):
    primary_text, full_text, source_mode = get_support_texts(row)
    relevance = row["Relevance"]

    pos = count_weighted_matches(primary_text, LEGITIMIZING_PATTERNS, base_weight=1.0)
    neg = count_delegit_weighted(primary_text)
    total = pos + neg

    # factual neutrality guard
    if (
        relevance >= 3 and
        row.get("Human_Rights_Abuse", 0) == 0 and
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
                    f"(legitimizing={fallback_pos}, delegitimizing={fallback_neg})"
                )
            if fallback_pos >= 2 and fallback_pos >= fallback_neg + 1:
                return pos, neg, 3, (
                    f"local WG/AC context weak; broader article legitimizing "
                    f"(legitimizing={fallback_pos}, delegitimizing={fallback_neg})"
                )
            if fallback_pos >= 1 and fallback_neg == 0:
                return pos, neg, 2, (
                    f"local WG/AC context weak; broader article suggests normalization "
                    f"(legitimizing={fallback_pos}, delegitimizing={fallback_neg})"
                )

        return pos, neg, 4, (
            f"no strong legitimation signals in local WG/AC context; broader article has "
            f"legitimizing={fallback_pos}, delegitimizing={fallback_neg}, treated conservatively"
        )

    if neg >= 3 and neg >= pos + 2:
        return pos, neg, 1, f"delegitimizing language dominates in {source_mode} (legitimizing={pos}, delegitimizing={neg})"

    if pos >= 2 and pos >= neg + 1:
        return pos, neg, 3, f"explicit legitimizing language dominates in {source_mode} (legitimizing={pos}, delegitimizing={neg})"

    if pos >= 1 and neg == 0:
        return pos, neg, 2, f"normalized / implicitly legitimizing language present in {source_mode} (legitimizing={pos}, delegitimizing={neg})"

    return pos, neg, 4, f"unclear legitimation pattern in {source_mode} (legitimizing={pos}, delegitimizing={neg})"

# =========================
# 9. REVIEW FLAG
# =========================
def step7_manual_review(row):
    relevance = row["Relevance"]

    if row["Stance_Support"] in [4, 5]:
        return 1

    if row["Legitimation_Support"] == 4:
        return 1

    if row["Ambivalence_Support"] == 1:
        return 1

    if relevance >= 3 and row["Stance_Support"] == 2 and row["Positive_Signal_Count"] + row["Negative_Signal_Count"] <= 2:
        return 1

    return 0

# =========================
# 10. MAIN
# =========================
def main():
    df6 = pd.read_excel(
        INPUT_STEP7,
        dtype={"Article_ID": str}
    )

    df1 = pd.read_excel(
        INPUT_STEP1,
        dtype={"article_id": str}
    )

    df6["Article_ID"] = df6["Article_ID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    df1["article_id"] = df1["article_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    merge_cols = [
        "article_id",
        "lead_clean",
        "body_postclean"
    ]
    df = df6.merge(df1[merge_cols], left_on="Article_ID", right_on="article_id", how="left")

    stance_results = df.apply(code_stance_support, axis=1)
    df["Positive_Signal_Count"] = stance_results.apply(lambda x: x[0])
    df["Negative_Signal_Count"] = stance_results.apply(lambda x: x[1])
    df["Stance_Support"] = stance_results.apply(lambda x: x[2])
    df["Stance_Support_Note"] = stance_results.apply(lambda x: x[3])

    amb_results = df.apply(code_ambivalence_support, axis=1)
    df["Ambivalence_Support"] = amb_results.apply(lambda x: x[0])
    df["Ambivalence_Support_Note"] = amb_results.apply(lambda x: x[1])

    leg_results = df.apply(code_legitimation_support, axis=1)
    df["Legitimation_Positive_Count"] = leg_results.apply(lambda x: x[0])
    df["Legitimation_Negative_Count"] = leg_results.apply(lambda x: x[1])
    df["Legitimation_Support"] = leg_results.apply(lambda x: x[2])
    df["Legitimation_Support_Note"] = leg_results.apply(lambda x: x[3])

    df["Step7_Manual_Review"] = df.apply(step7_manual_review, axis=1)

    df_out = pd.DataFrame({
        "Article_ID": df["Article_ID"],
        "Outlet": df["Outlet"],
        "Date": df["Date"],
        "Headline": df["Headline"],
        "Relevance": df["Relevance"],
        "Relevance_Label": df["Relevance_Label"],
        "Actor_Mention": df["Actor_Mention"],
        "Successor_Frame": df["Successor_Frame"],
        "Dominant_Label": df["Dominant_Label"],
        "Dominant_Location": df["Dominant_Location"],
        "Main_Associated_Actor": df["Main_Associated_Actor"],

        "Counterterrorism": df["Counterterrorism"],
        "Sovereignty": df["Sovereignty"],
        "Human_Rights_Abuse": df["Human_Rights_Abuse"],
        "Anti_or_Neocolonialism": df["Anti_or_Neocolonialism"],
        "Western_Failure": df["Western_Failure"],
        "Security_Effectiveness": df["Security_Effectiveness"],
        "Economic_Interests": df["Economic_Interests"],
        "Geopolitical_Rivalry": df["Geopolitical_Rivalry"],

        "Positive_Signal_Count": df["Positive_Signal_Count"],
        "Negative_Signal_Count": df["Negative_Signal_Count"],
        "Stance_Support": df["Stance_Support"],
        "Stance_Support_Note": df["Stance_Support_Note"],

        "Ambivalence_Support": df["Ambivalence_Support"],
        "Ambivalence_Support_Note": df["Ambivalence_Support_Note"],

        "Legitimation_Positive_Count": df["Legitimation_Positive_Count"],
        "Legitimation_Negative_Count": df["Legitimation_Negative_Count"],
        "Legitimation_Support": df["Legitimation_Support"],
        "Legitimation_Support_Note": df["Legitimation_Support_Note"],

        "Step7_Manual_Review": df["Step7_Manual_Review"],
        "URL": df["URL"]
    })

    df_out.to_excel(OUTPUT_XLSX, index=False)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

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
    print(f"\nSaved to {OUTPUT_XLSX}")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()