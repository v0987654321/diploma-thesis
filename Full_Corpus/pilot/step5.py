import pandas as pd
import re

INPUT_STEP4 = "data/postStep4.csv"
INPUT_STEP1 = "data/postStep1.csv"
OUTPUT_CSV = "data/postStep5.csv"

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
# 3. PATTERNS FOR ASSOCIATED ACTORS
# =========================
ACTOR_PATTERNS = {
    1: [  # Malian army / junta
        r"\bfama\b",
        r"\barmée\s+malienne\b",
        r"\bforces?\s+arm[ée]es?\s+maliennes\b",
        r"\bjunte\b",
        r"\bautorités\s+maliennes\b",
        r"\bgouvernement\s+malien\b",
        r"\btransition\s+malienne\b",
        r"\bassimi\s+go[iï]ta\b",
        r"\bcolonel\s+go[iï]ta\b",
        r"\bcolonels?\s+maliens\b",
    ],
    2: [  # Russia / Russian state
        r"\brussie\b",
        r"\bmoscou\b",
        r"\bkremlin\b",
        r"\bf[ée]d[ée]ration\s+de\s+russie\b",
        r"\bgouvernement\s+russe\b",
        r"\bminist[èe]re\s+russe\s+de\s+la\s+d[ée]fense\b",
        r"\blavrov\b",
        r"\bpoutine\b",
    ],
    3: [  # France
        r"\bfrance\b",
        r"\bparis\b",
        r"\bbarkhane\b",
        r"\barm[ée]e?\s+fran[çc]aise\b",
        r"\bforces?\s+fran[çc]aises\b",
        r"\bop[ée]ration\s+serval\b",
        r"\blecornu\b",
        r"\bflorence\s+parly\b",
    ],
    4: [  # UN / MINUSMA
        r"\bminusma\b",
        r"\bonu\b",
        r"\bnations\s+unies\b",
        r"\bunowas\b",
        r"\bmission\s+onusienne\b",
        r"\bconseil\s+de\s+s[ée]curit[ée]\b",
    ],
    5: [  # ECOWAS / regional actors
        r"\bcedeao\b",
        r"\becowas\b",
        r"\bunion\s+africaine\b",
        r"\bua\b",
        r"\borganisation\s+r[ée]gionale\b",
        r"\bcommunaut[ée]\s+[ée]conomique\s+des\s+[ée]tats\s+de\s+l['’]afrique\s+de\s+l['’]ouest\b",
    ],
    6: [  # local civilians
        r"\bcivils?\b",
        r"\bpopulation\b",
        r"\bpopulations\b",
        r"\bhabitants?\b",
        r"\bcommunaut[ée]s?\s+locales?\b",
        r"\blocaux?\b",
    ],
    7: [  # jihadist / terrorist groups
        r"\bjihadistes?\b",
        r"\bterroristes?\b",
        r"\bgsim\b",
        r"\bjnim\b",
        r"\baqmi\b",
        r"\beis\b",
        r"\betat\s+islamique\b",
        r"\bgroupes?\s+arm[ée]s?\b",
        r"\binsurg[ée]s?\b",
    ],
    8: [  # Western states more broadly
        r"\boccident\b",
        r"\boccidentaux\b",
        r"\betats?-unis\b",
        r"\bunion\s+europ[ée]enne\b",
        r"\bue\b",
        r"\bpartenaires?\s+internationaux\b",
        r"\bpays\s+occidentaux\b",
        r"\bcanada\b",
        r"\ballemagne\b",
        r"\bitalie\b",
        r"\bespagne\b",
        r"\bestonie\b",
        r"\bdanemark\b",
        r"\bsu[èe]de\b",
        r"\bbelgique\b",
        r"\bpays-bas\b",
        r"\bportugal\b",
        r"\blituanie\b",
        r"\bnorv[èe]ge\b",
        r"\br[ée]publique\s+tch[èe]que\b",
        r"\broumanie\b",
        r"\broy[a|u]me-uni\b",
    ],
}

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

def extract_target_context(text, window=2):
    sentences = split_sentences(text)
    if not sentences:
        return ""

    selected = set()
    for i, sent in enumerate(sentences):
        if has_any(sent, WGAC_PATTERNS):
            for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
                selected.add(j)

    if not selected:
        return text

    return " ".join([sentences[i] for i in sorted(selected)])

# =========================
# 5. SEGMENT-WEIGHTED SCORING
# =========================
def score_actor_by_segments(row, patterns):
    headline = safe_str(row.get("Headline", ""))
    lead = safe_str(row.get("lead_clean", ""))
    body = safe_str(row.get("body_postclean", ""))

    headline_hits = count_matches(headline, patterns)
    lead_hits = count_matches(lead, patterns)
    body_hits = count_matches(body, patterns)

    weighted_score = (headline_hits * 3) + (lead_hits * 2) + (body_hits * 1)
    return weighted_score, headline_hits, lead_hits, body_hits

def score_actor_in_local_context(row, patterns):
    text_local = extract_target_context(get_analysis_text(row), window=2)
    return count_matches(text_local, patterns)

# =========================
# 6. PROXIMITY / ASSOCIATION BONUSES
# =========================
def proximity_bonus(row, patterns):
    text = get_analysis_text(row)
    bonus = 0
    sentences = split_sentences(text)

    for sent in sentences:
        if has_any(sent, WGAC_PATTERNS) and has_any(sent, patterns):
            bonus += 2

    return bonus

def sentence_level_association_bonus(row, patterns):
    text = get_analysis_text(row)
    sents = split_sentences(text)
    bonus = 0

    for i, sent in enumerate(sents):
        actor_here = has_any(sent, patterns)
        wgac_here = has_any(sent, WGAC_PATTERNS)

        if actor_here and wgac_here:
            bonus += 3
            continue

        if actor_here:
            prev_wgac = i > 0 and has_any(sents[i - 1], WGAC_PATTERNS)
            next_wgac = i < len(sents) - 1 and has_any(sents[i + 1], WGAC_PATTERNS)
            if prev_wgac or next_wgac:
                bonus += 1

    return bonus

# =========================
# 7. MAIN ASSOCIATED ACTOR CODING
# =========================
def code_main_associated_actor(row):
    actor_scores = {}
    actor_notes = {}

    for code, patterns in ACTOR_PATTERNS.items():
        seg_score, h, l, b = score_actor_by_segments(row, patterns)
        local_context_hits = score_actor_in_local_context(row, patterns)
        prox = proximity_bonus(row, patterns)
        sent_assoc = sentence_level_association_bonus(row, patterns)

        total_score = seg_score + (local_context_hits * 1.5) + prox + sent_assoc

        if code == 8:
            total_score = total_score * 0.8

        if code == 6:
            total_score = total_score * 0.85

        actor_scores[code] = total_score
        actor_notes[code] = {
            "headline": h,
            "lead": l,
            "body": b,
            "local_context_hits": local_context_hits,
            "proximity_bonus": prox,
            "sentence_assoc_bonus": sent_assoc,
            "total_score": total_score
        }

    positive_scores = {k: v for k, v in actor_scores.items() if v > 0}

    if not positive_scores:
        return 9, "no clear dominant associated actor detected"

    max_score = max(positive_scores.values())
    top_codes = [k for k, v in positive_scores.items() if abs(v - max_score) < 0.01]

    label_map = {
        1: "Malian army / junta references dominate",
        2: "Russia / Russian state references dominate",
        3: "France references dominate",
        4: "UN / MINUSMA references dominate",
        5: "ECOWAS / regional actor references dominate",
        6: "local civilian references dominate",
        7: "jihadist / terrorist actor references dominate",
        8: "Western-state references dominate",
        9: "no clear dominant associated actor",
        10: "other"
    }

    if len(top_codes) == 1:
        code = top_codes[0]
        n = actor_notes[code]
        return code, (
            f"{label_map[code]} "
            f"(headline={n['headline']}, lead={n['lead']}, body={n['body']}, "
            f"local_context_hits={n['local_context_hits']}, "
            f"proximity_bonus={n['proximity_bonus']}, "
            f"sentence_assoc_bonus={n['sentence_assoc_bonus']}, "
            f"total_score={round(n['total_score'], 2)})"
        )

    preferred_order = [2, 1, 3, 7, 4, 5, 8, 6]

    for pref in preferred_order:
        if pref in top_codes:
            n = actor_notes[pref]
            return pref, (
                f"tie resolved in favor of code {pref}; raw top codes = {top_codes} "
                f"(headline={n['headline']}, lead={n['lead']}, body={n['body']}, "
                f"local_context_hits={n['local_context_hits']}, "
                f"proximity_bonus={n['proximity_bonus']}, "
                f"sentence_assoc_bonus={n['sentence_assoc_bonus']}, "
                f"total_score={round(n['total_score'], 2)})"
            )

    return 10, f"other / unresolved tie; raw top codes = {top_codes}"

# =========================
# 8. REVIEW FLAG
# =========================
def step5_manual_review(row):
    assoc = safe_int(row.get("Main_Associated_Actor"))
    note = safe_str(row.get("Main_Associated_Actor_Note")).lower()
    relevance = safe_int(row.get("Relevance"))

    if assoc in [9, 10]:
        return 1

    if "tie resolved" in note:
        return 1

    if relevance >= 3 and assoc == 8:
        return 1

    return 0

# =========================
# 9. ROW PROCESSOR
# =========================
def process_row(row):
    code, note = code_main_associated_actor(row)

    out = row.to_dict()
    out["Main_Associated_Actor"] = code
    out["Main_Associated_Actor_Note"] = note
    return out

# =========================
# 10. MAIN
# =========================
def main():
    df4 = pd.read_csv(
        INPUT_STEP4,
        dtype={"Article_ID": str},
        low_memory=False
    )

    df1 = pd.read_csv(
        INPUT_STEP1,
        dtype={"article_id": str},
        low_memory=False
    )

    df4 = ensure_columns(df4, [
        "Article_ID", "Outlet", "Date", "Headline",
        "Relevance", "Relevance_Label",
        "Actor_Mention", "Successor_Frame",
        "Dominant_Label", "Dominant_Location", "URL"
    ])

    df1 = ensure_columns(df1, [
        "article_id", "lead_clean", "body_postclean"
    ])

    df4["Article_ID"] = df4["Article_ID"].apply(normalize_article_id)
    df1["article_id"] = df1["article_id"].apply(normalize_article_id)

    merge_cols = [
        "article_id",
        "lead_clean",
        "body_postclean"
    ]
    df = df4.merge(df1[merge_cols], left_on="Article_ID", right_on="article_id", how="left")

    total = len(df)
    processed_rows = []

    print(f"\nStep5: processing {total} articles")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        processed_rows.append(process_row(row))
        print_progress(i, total, every=100)

    df_processed = pd.DataFrame(processed_rows)

    df_processed["Step5_Manual_Review"] = df_processed.apply(step5_manual_review, axis=1)

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
        "Main_Associated_Actor_Note": df_processed["Main_Associated_Actor_Note"],
        "Step5_Manual_Review": df_processed["Step5_Manual_Review"],
        "URL": df_processed["URL"]
    })

    print("\nStep5 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    for code in range(1, 11):
        print(f"Main_Associated_Actor={code}: {(df_out['Main_Associated_Actor'] == code).sum()}")
    print(f"Manual review: {df_out['Step5_Manual_Review'].sum()}")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Dominant_Location",
        "Main_Associated_Actor",
        "Step5_Manual_Review"
    ]
    print(df_out[preview_cols].head(30))

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()