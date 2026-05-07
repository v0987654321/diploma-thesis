import pandas as pd
import re

INPUT_STEP3 = "data/postStep3.csv"
INPUT_STEP1 = "data/postStep1.csv"
OUTPUT_CSV = "data/postStep4.csv"

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

def has_any(text, patterns):
    text = safe_str(text)
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def count_matches(text, patterns):
    text = safe_str(text)
    if not text:
        return 0
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, flags=re.IGNORECASE))
    return total

def print_progress(i, total, every=100):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"Progress: {i}/{total} ({pct}%)")

# =========================
# 2. LOCATION PATTERNS
# =========================
MALI_PATTERNS = [
    r"\bmali\b",
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
    r"\bber\b",
    r"\bsenou\b",
]

OTHER_AFRICA_PATTERNS = [
    r"\bniger\b",
    r"\bniamey\b",
    r"\bburkina\b",
    r"\bouagadougou\b",
    r"\bburkina\s+faso\b",
    r"\bcentrafrique\b",
    r"\br[ée]publique\s+centrafricaine\b",
    r"\brca\b",
    r"\bmauritanie\b",
    r"\btchad\b",
    r"\bsoudan\b",
    r"\bs[ée]n[ée]gal\b",
    r"\bgambie\b",
    r"\bc[ôo]te d['’]ivoire\b",
    r"\bb[ée]nin\b",
    r"\bafrique du sud\b",
]

UKRAINE_PATTERNS = [
    r"\bukraine\b",
    r"\bbakhmout\b",
    r"\bdonbass\b",
    r"\bzelensky\b",
    r"\bfront\s+ukrainien\b",
]

OTHER_LOCATION_PATTERNS = [
    r"\bsyrie\b",
    r"\blibye\b",
    r"\bafghanistan\b",
    r"\biran\b",
    r"\bisra[ëe]l\b",
    r"\brussie\b",
    r"\bmoscou\b",
    r"\begypte\b",
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

# =========================
# 4. DOMINANT LOCATION CODING
# =========================
def code_dominant_location(row):
    text = get_analysis_text(row)

    mali_hits = count_matches(text, MALI_PATTERNS)
    africa_hits = count_matches(text, OTHER_AFRICA_PATTERNS)
    ukraine_hits = count_matches(text, UKRAINE_PATTERNS)
    other_hits = count_matches(text, OTHER_LOCATION_PATTERNS)

    other_max = max(africa_hits, ukraine_hits, other_hits)

    # stricter mixed-location logic:
    # code 5 only when Mali and another location are both strongly present
    # and relatively balanced
    if (
        mali_hits >= 3 and
        other_max >= 4 and
        other_max >= mali_hits * 0.75
    ):
        return 5, (
            f"Mali and other location both strongly and relatively evenly present "
            f"(Mali={mali_hits}, Africa={africa_hits}, Ukraine={ukraine_hits}, Other={other_hits})"
        )

    scores = {
        1: mali_hits,
        2: africa_hits,
        3: ukraine_hits,
        4: other_hits
    }

    max_score = max(scores.values())

    if max_score == 0:
        return 4, "no strong location markers found; coded as Other location by fallback"

    top_codes = [k for k, v in scores.items() if v == max_score]

    if len(top_codes) == 1:
        code = top_codes[0]
        note_map = {
            1: f"Mali-coded signals dominate (hits={mali_hits})",
            2: f"Other African location signals dominate (hits={africa_hits})",
            3: f"Ukraine-coded signals dominate (hits={ukraine_hits})",
            4: f"Other location signals dominate (hits={other_hits})",
        }
        return code, note_map[code]

    # tie-breaking preference:
    # Mali > Other Africa > Other > Ukraine
    if 1 in top_codes:
        return 1, f"tie resolved in favor of Mali (Mali={mali_hits}, Africa={africa_hits}, Ukraine={ukraine_hits}, Other={other_hits})"
    if 2 in top_codes:
        return 2, f"tie resolved in favor of Other African location (Mali={mali_hits}, Africa={africa_hits}, Ukraine={ukraine_hits}, Other={other_hits})"
    if 4 in top_codes:
        return 4, f"tie resolved in favor of Other location (Mali={mali_hits}, Africa={africa_hits}, Ukraine={ukraine_hits}, Other={other_hits})"

    return 3, f"tie resolved in favor of Ukraine (Mali={mali_hits}, Africa={africa_hits}, Ukraine={ukraine_hits}, Other={other_hits})"

# =========================
# 5. REVIEW FLAG
# =========================
def step4_manual_review(row):
    dominant_location = safe_int(row.get("Dominant_Location"))
    note = safe_str(row.get("Dominant_Location_Note")).lower()

    if dominant_location == 5:
        return 1

    if "fallback" in note or "tie resolved" in note:
        return 1

    return 0

# =========================
# 6. ROW PROCESSOR
# =========================
def process_row(row):
    code, note = code_dominant_location(row)

    out = row.to_dict()
    out["Dominant_Location"] = code
    out["Dominant_Location_Note"] = note
    return out

# =========================
# 7. MAIN
# =========================
def main():
    df3 = pd.read_csv(
        INPUT_STEP3,
        dtype={"Article_ID": str},
        low_memory=False
    )

    df1 = pd.read_csv(
        INPUT_STEP1,
        dtype={"article_id": str},
        low_memory=False
    )

    df3 = ensure_columns(df3, [
        "Article_ID", "Outlet", "Date", "Headline",
        "Relevance", "Relevance_Label",
        "Actor_Mention", "Successor_Frame",
        "Dominant_Label", "URL"
    ])

    df1 = ensure_columns(df1, [
        "article_id", "lead_clean", "body_postclean"
    ])

    df3["Article_ID"] = df3["Article_ID"].apply(normalize_article_id)
    df1["article_id"] = df1["article_id"].apply(normalize_article_id)

    merge_cols = [
        "article_id",
        "lead_clean",
        "body_postclean"
    ]
    df = df3.merge(df1[merge_cols], left_on="Article_ID", right_on="article_id", how="left")

    total = len(df)
    processed_rows = []

    print(f"\nStep4: processing {total} articles")

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        processed_rows.append(process_row(row))
        print_progress(i, total, every=100)

    df_processed = pd.DataFrame(processed_rows)

    df_processed["Step4_Manual_Review"] = df_processed.apply(step4_manual_review, axis=1)

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
        "Dominant_Location_Note": df_processed["Dominant_Location_Note"],
        "Step4_Manual_Review": df_processed["Step4_Manual_Review"],
        "URL": df_processed["URL"]
    })

    print("\nStep4 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    print(f"Dominant_Location=1 (Mali): {(df_out['Dominant_Location'] == 1).sum()}")
    print(f"Dominant_Location=2 (Other Africa): {(df_out['Dominant_Location'] == 2).sum()}")
    print(f"Dominant_Location=3 (Ukraine): {(df_out['Dominant_Location'] == 3).sum()}")
    print(f"Dominant_Location=4 (Other): {(df_out['Dominant_Location'] == 4).sum()}")
    print(f"Dominant_Location=5 (Mali + other): {(df_out['Dominant_Location'] == 5).sum()}")
    print(f"Manual review: {df_out['Step4_Manual_Review'].sum()}")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Actor_Mention",
        "Dominant_Label",
        "Dominant_Location",
        "Step4_Manual_Review"
    ]
    print(df_out[preview_cols].head(30))

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()