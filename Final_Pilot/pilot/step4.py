import pandas as pd
import re

INPUT_STEP4 = "data/postStep3.xlsx"
INPUT_STEP1 = "data/postStep1.xlsx"
OUTPUT_XLSX = "data/postStep4.xlsx"
OUTPUT_CSV = "data/postStep4.csv"

# =========================
# 1. HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

def has_any(text, patterns):
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def count_matches(text, patterns):
    if not text:
        return 0
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, flags=re.IGNORECASE))
    return total

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
    return " ".join([p for p in parts if p]).strip()

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
    # and relatively balanced, not just when article is Mali-centered
    # with some comparative regional references
    if (
        mali_hits >= 3 and
        other_max >= 4 and
        other_max >= mali_hits * 0.75
    ):
        return 5, (
            f"Mali and other location both strongly and relatively evenly present "
            f"(Mali={mali_hits}, Africa={africa_hits}, Ukraine={ukraine_hits}, Other={other_hits})"
        )

    # otherwise choose dominant single bloc
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
    note = safe_str(row["Dominant_Location_Note"]).lower()

    if row["Dominant_Location"] == 5:
        return 1

    if "fallback" in note or "tie resolved" in note:
        return 1

    return 0

# =========================
# 6. MAIN
# =========================
def main():
    df3 = pd.read_excel(
        INPUT_STEP4,
        dtype={"Article_ID": str}
    )

    df1 = pd.read_excel(
        INPUT_STEP1,
        dtype={"article_id": str}
    )

    # normalize IDs
    df3["Article_ID"] = df3["Article_ID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    df1["article_id"] = df1["article_id"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    # merge text source from step1
    merge_cols = [
        "article_id",
        "lead_clean",
        "body_postclean"
    ]
    df = df3.merge(df1[merge_cols], left_on="Article_ID", right_on="article_id", how="left")

    # Dominant location
    location_results = df.apply(code_dominant_location, axis=1)
    df["Dominant_Location"] = location_results.apply(lambda x: x[0])
    df["Dominant_Location_Note"] = location_results.apply(lambda x: x[1])

    # review
    df["Step4_Manual_Review"] = df.apply(step4_manual_review, axis=1)

    # clean export
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
        "Dominant_Location_Note": df["Dominant_Location_Note"],
        "Step4_Manual_Review": df["Step4_Manual_Review"],
        "URL": df["URL"]
    })

    df_out.to_excel(OUTPUT_XLSX, index=False)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

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
    print(f"\nSaved to {OUTPUT_XLSX}")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()