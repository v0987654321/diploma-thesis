import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_REVIEW_MASTER = SCRIPT_DIR / "data" / "review_master.csv"
INPUT_KWIC_ALL = SCRIPT_DIR / "concordance" / "kwic_all_keywords.csv"
INPUT_KWIC_KEYWORD_SUMMARY = SCRIPT_DIR / "concordance" / "kwic_keyword_summary.csv"
INPUT_KWIC_GROUP_SUMMARY = SCRIPT_DIR / "concordance" / "kwic_group_summary.csv"
INPUT_KEYWORD_OUTLET_PROFILE = SCRIPT_DIR / "concordance" / "keyword_outlet_profile.csv"
INPUT_KEYWORD_SUBSET_PROFILE = SCRIPT_DIR / "concordance" / "keyword_subset_profile.csv"

LEXICAL_DIR = SCRIPT_DIR / "lexical"
FIGURES_DIR = SCRIPT_DIR / "figures"
SUMMARY_DIR = SCRIPT_DIR / "summary"

TOP_N = 20

# =========================
# HELPERS
# =========================
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
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default

def ensure_dirs():
    LEXICAL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def save_bar_chart(df, label_col, value_col, title, filename, rotate=True, figsize=(11, 6)):
    if df.empty:
        return
    plt.figure(figsize=figsize)
    plt.bar(df[label_col].astype(str), df[value_col])
    plt.title(title)
    plt.ylabel(value_col)
    plt.xlabel(label_col)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close()

def load_required_csv(path, dtype=None):
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path, dtype=dtype, low_memory=False)

def normalize_sentence_for_pattern(text):
    text = safe_str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    review_master = load_required_csv(INPUT_REVIEW_MASTER, dtype={"Article_ID": str})
    kwic_all = load_required_csv(INPUT_KWIC_ALL, dtype={"Article_ID": str})
    kwic_keyword_summary = load_required_csv(INPUT_KWIC_KEYWORD_SUMMARY)
    kwic_group_summary = load_required_csv(INPUT_KWIC_GROUP_SUMMARY)
    keyword_outlet_profile = load_required_csv(INPUT_KEYWORD_OUTLET_PROFILE)
    keyword_subset_profile = load_required_csv(INPUT_KEYWORD_SUBSET_PROFILE)

    # =========================
    # BASIC NORMALIZATION
    # =========================
    if "Relevance" in review_master.columns:
        review_master["Relevance"] = review_master["Relevance"].fillna(0).astype(int)

    if "Actor_Mention" in review_master.columns:
        review_master["Actor_Mention"] = review_master["Actor_Mention"].fillna(0).astype(int)

    if "Successor_Frame" in review_master.columns:
        review_master["Successor_Frame"] = review_master["Successor_Frame"].fillna(0).astype(int)

    if "Any_Review_Flag" in review_master.columns:
        review_master["Any_Review_Flag"] = review_master["Any_Review_Flag"].fillna(0).astype(int)

    if "source_attributed_flag" in review_master.columns:
        review_master["source_attributed_flag"] = review_master["source_attributed_flag"].fillna(0).astype(int)

    if "Outlet" in keyword_outlet_profile.columns:
        keyword_outlet_profile["Outlet"] = keyword_outlet_profile["Outlet"].fillna("missing")

    # =========================
    # TOP KEYWORDS OVERALL
    # =========================
    top_keywords_overall = (
        kwic_keyword_summary
        .sort_values(["matches_total", "unique_articles"], ascending=[False, False])
        .head(TOP_N)
        .reset_index(drop=True)
    )
    save_csv(top_keywords_overall, LEXICAL_DIR / "top_keywords_overall.csv")

    # =========================
    # TOP KEYWORD GROUPS OVERALL
    # =========================
    top_keyword_groups = (
        kwic_group_summary
        .sort_values(["matches_total", "unique_articles"], ascending=[False, False])
        .reset_index(drop=True)
    )
    save_csv(top_keyword_groups, LEXICAL_DIR / "keyword_group_profile.csv")

    # =========================
    # TOP KEYWORDS BY SUBSET
    # =========================
    subset_names = [
        "relevance_3plus",
        "relevance_4",
        "africa_corps_cases",
        "successor_cases",
        "review_flagged",
        "source_attributed",
    ]

    subset_outputs = []

    for subset_name in subset_names:
        sub = keyword_subset_profile[keyword_subset_profile["subset"] == subset_name].copy()
        if sub.empty:
            continue

        top_sub = (
            sub.sort_values(["matches_total", "unique_articles"], ascending=[False, False])
            .head(TOP_N)
            .reset_index(drop=True)
        )
        save_csv(top_sub, LEXICAL_DIR / f"top_keywords_{subset_name}.csv")
        subset_outputs.append((subset_name, top_sub))

    # =========================
    # OUTLET KEYWORD PROFILE
    # =========================
    outlet_keyword_profile = (
        keyword_outlet_profile
        .groupby(["Outlet", "keyword_group", "keyword_name"], dropna=False)["matches_total"]
        .sum()
        .reset_index()
        .sort_values(["matches_total"], ascending=False)
    )
    save_csv(outlet_keyword_profile, LEXICAL_DIR / "outlet_keyword_profile.csv")

    # =========================
    # MOST FREQUENT KEYWORDS PER OUTLET
    # =========================
    top_keywords_by_outlet_rows = []
    if not outlet_keyword_profile.empty:
        for outlet, grp in outlet_keyword_profile.groupby("Outlet", dropna=False):
            top_grp = grp.sort_values("matches_total", ascending=False).head(10)
            for _, row in top_grp.iterrows():
                top_keywords_by_outlet_rows.append({
                    "Outlet": outlet,
                    "keyword_group": row["keyword_group"],
                    "keyword_name": row["keyword_name"],
                    "matches_total": row["matches_total"]
                })

    top_keywords_by_outlet = pd.DataFrame(top_keywords_by_outlet_rows)
    save_csv(top_keywords_by_outlet, LEXICAL_DIR / "top_keywords_by_outlet.csv")

    # =========================
    # KEYWORD GROUP PROFILE BY SUBSET
    # =========================
    keyword_group_subset_profile = (
        keyword_subset_profile
        .groupby(["keyword_group", "subset"], dropna=False)[["matches_total", "unique_articles"]]
        .sum()
        .reset_index()
        .sort_values(["subset", "matches_total"], ascending=[True, False])
    )
    save_csv(keyword_group_subset_profile, LEXICAL_DIR / "keyword_group_subset_profile.csv")

    # =========================
    # TOP MATCHED TEXT FORMS
    # =========================
    if not kwic_all.empty and "matched_text" in kwic_all.columns:
        top_matched_forms = (
            kwic_all.assign(matched_text_norm=kwic_all["matched_text"].apply(lambda x: safe_str(x).lower()))
            .groupby(["keyword_group", "keyword_name", "matched_text_norm"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        top_matched_forms = pd.DataFrame(columns=["keyword_group", "keyword_name", "matched_text_norm", "count"])

    save_csv(top_matched_forms, LEXICAL_DIR / "top_matched_forms.csv")

    # =========================
    # REPEATED SENTENCE PATTERNS
    # =========================
    if not kwic_all.empty and "sentence_full" in kwic_all.columns:
        repeated_sentence_patterns = (
            kwic_all.assign(sentence_norm=kwic_all["sentence_full"].apply(normalize_sentence_for_pattern))
            .groupby(["keyword_group", "keyword_name", "sentence_norm"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        repeated_sentence_patterns = repeated_sentence_patterns[repeated_sentence_patterns["count"] >= 2].copy()
    else:
        repeated_sentence_patterns = pd.DataFrame(columns=["keyword_group", "keyword_name", "sentence_norm", "count"])

    save_csv(repeated_sentence_patterns, LEXICAL_DIR / "repeated_sentence_patterns.csv")

    # =========================
    # TOP SENTENCE PATTERNS PER KEYWORD GROUP
    # =========================
    if not repeated_sentence_patterns.empty:
        top_sentence_patterns_by_group = (
            repeated_sentence_patterns
            .sort_values(["keyword_group", "count"], ascending=[True, False])
            .groupby("keyword_group", as_index=False)
            .head(10)
            .reset_index(drop=True)
        )
    else:
        top_sentence_patterns_by_group = pd.DataFrame(columns=["keyword_group", "keyword_name", "sentence_norm", "count"])

    save_csv(top_sentence_patterns_by_group, LEXICAL_DIR / "top_sentence_patterns_by_group.csv")

    # =========================
    # KWIC COVERAGE PROFILE
    # =========================
    kwic_coverage_rows = []
    total_articles = len(review_master)

    if not kwic_all.empty and "Article_ID" in kwic_all.columns:
        kwic_article_ids = set(kwic_all["Article_ID"].dropna().astype(str).tolist())
    else:
        kwic_article_ids = set()

    review_master["has_any_kwic_hit"] = review_master["Article_ID"].astype(str).isin(kwic_article_ids).astype(int)

    kwic_coverage_rows.append({
        "subset": "all_articles",
        "articles_total": total_articles,
        "articles_with_kwic_hit": int(review_master["has_any_kwic_hit"].sum()),
        "coverage_percent": (review_master["has_any_kwic_hit"].sum() / total_articles * 100) if total_articles else 0
    })

    subset_masks = {
        "relevance_3plus": review_master["Relevance"].isin([3, 4]) if "Relevance" in review_master.columns else pd.Series(False, index=review_master.index),
        "relevance_4": review_master["Relevance"].eq(4) if "Relevance" in review_master.columns else pd.Series(False, index=review_master.index),
        "africa_corps_cases": review_master["Actor_Mention"].isin([2, 3]) if "Actor_Mention" in review_master.columns else pd.Series(False, index=review_master.index),
        "successor_cases": review_master["Successor_Frame"].eq(1) if "Successor_Frame" in review_master.columns else pd.Series(False, index=review_master.index),
        "review_flagged": review_master["Any_Review_Flag"].eq(1) if "Any_Review_Flag" in review_master.columns else pd.Series(False, index=review_master.index),
        "source_attributed": review_master["source_attributed_flag"].eq(1) if "source_attributed_flag" in review_master.columns else pd.Series(False, index=review_master.index),
    }

    for subset_name, mask in subset_masks.items():
        subset_df = review_master[mask].copy()
        total_n = len(subset_df)
        hit_n = int(subset_df["has_any_kwic_hit"].sum()) if "has_any_kwic_hit" in subset_df.columns else 0
        kwic_coverage_rows.append({
            "subset": subset_name,
            "articles_total": total_n,
            "articles_with_kwic_hit": hit_n,
            "coverage_percent": (hit_n / total_n * 100) if total_n else 0
        })

    kwic_coverage_profile = pd.DataFrame(kwic_coverage_rows)
    save_csv(kwic_coverage_profile, LEXICAL_DIR / "kwic_coverage_profile.csv")

    # =========================
    # OUTLET × KEYWORD GROUP PROFILE
    # =========================
    if not kwic_all.empty and "Outlet" in kwic_all.columns:
        outlet_group_profile = (
            kwic_all.groupby(["Outlet", "keyword_group"], dropna=False)
            .size()
            .reset_index(name="matches_total")
            .sort_values(["Outlet", "matches_total"], ascending=[True, False])
        )
    else:
        outlet_group_profile = pd.DataFrame(columns=["Outlet", "keyword_group", "matches_total"])

    save_csv(outlet_group_profile, LEXICAL_DIR / "outlet_keyword_group_profile.csv")

    # =========================
    # FIGURES
    # =========================
    save_bar_chart(
        top_keywords_overall.head(15),
        label_col="keyword_name",
        value_col="matches_total",
        title="Top keywords overall",
        filename="lexical_top_keywords_overall.png"
    )

    save_bar_chart(
        top_keyword_groups,
        label_col="keyword_group",
        value_col="matches_total",
        title="Keyword group profile",
        filename="lexical_keyword_groups.png"
    )

    if not kwic_coverage_profile.empty:
        save_bar_chart(
            kwic_coverage_profile,
            label_col="subset",
            value_col="coverage_percent",
            title="KWIC coverage by subset",
            filename="kwic_coverage_by_subset.png"
        )

    if not outlet_group_profile.empty:
        top_outlet_group = (
            outlet_group_profile.sort_values("matches_total", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        top_outlet_group["outlet_group"] = top_outlet_group["Outlet"].astype(str) + " | " + top_outlet_group["keyword_group"].astype(str)

        save_bar_chart(
            top_outlet_group,
            label_col="outlet_group",
            value_col="matches_total",
            title="Top outlet × keyword group combinations",
            filename="outlet_keyword_group_profile.png"
        )

    # =========================
    # SUMMARY TEXT
    # =========================
    if not kwic_keyword_summary.empty:
        top_keyword_row = kwic_keyword_summary.sort_values("matches_total", ascending=False).iloc[0]
        top_keyword_name = safe_str(top_keyword_row["keyword_name"])
        top_keyword_group = safe_str(top_keyword_row["keyword_group"])
        top_keyword_matches = int(top_keyword_row["matches_total"])
    else:
        top_keyword_name = "N/A"
        top_keyword_group = "N/A"
        top_keyword_matches = 0

    if not kwic_group_summary.empty:
        top_group_row = kwic_group_summary.sort_values("matches_total", ascending=False).iloc[0]
        top_group_name = safe_str(top_group_row["keyword_group"])
        top_group_matches = int(top_group_row["matches_total"])
    else:
        top_group_name = "N/A"
        top_group_matches = 0

    if not keyword_subset_profile.empty:
        strongest_subset_keyword = (
            keyword_subset_profile.sort_values("matches_total", ascending=False)
            .iloc[0]
        )
        strongest_subset_desc = (
            f"{safe_str(strongest_subset_keyword['keyword_name'])} "
            f"in {safe_str(strongest_subset_keyword['subset'])} "
            f"({int(strongest_subset_keyword['matches_total'])} matches)"
        )
    else:
        strongest_subset_desc = "N/A"

    if not repeated_sentence_patterns.empty:
        repeated_sentence_count = len(repeated_sentence_patterns)
    else:
        repeated_sentence_count = 0

    summary_text = f"""
CORPUS4 SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Inputs:
- review_master rows: {len(review_master)}
- kwic_all_keywords rows: {len(kwic_all)}
- keyword summary rows: {len(kwic_keyword_summary)}
- keyword group summary rows: {len(kwic_group_summary)}

Lexical evaluation outputs:
- Top keywords overall exported
- Top keywords by subset exported
- Outlet keyword profile exported
- Keyword group subset profile exported
- Repeated sentence pattern profile exported
- KWIC coverage profile exported

Headline lexical findings:
- Top keyword overall: {top_keyword_name} (group: {top_keyword_group}, matches: {top_keyword_matches})
- Top keyword group overall: {top_group_name} ({top_group_matches} matches)
- Strongest keyword/subset combination: {strongest_subset_desc}
- Repeated sentence patterns retained (count >= 2): {repeated_sentence_count}

Output files:
- lexical/top_keywords_overall.csv
- lexical/top_keywords_<subset>.csv
- lexical/keyword_group_profile.csv
- lexical/outlet_keyword_profile.csv
- lexical/top_keywords_by_outlet.csv
- lexical/keyword_group_subset_profile.csv
- lexical/top_matched_forms.csv
- lexical/repeated_sentence_patterns.csv
- lexical/top_sentence_patterns_by_group.csv
- lexical/kwic_coverage_profile.csv
- lexical/outlet_keyword_group_profile.csv
- figures/*.png
- summary/corpus4_summary.txt
""".strip()

    with open(SUMMARY_DIR / "corpus4_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open(SUMMARY_DIR / "corpus4_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    lexical_findings_text = f"""
CORPUS4 LEXICAL FINDINGS
Generated: {datetime.now().isoformat(timespec="seconds")}

1. General observations
- Total KWIC matches across all keywords: {len(kwic_all)}
- Total unique articles with KWIC hits: {review_master['has_any_kwic_hit'].sum() if 'has_any_kwic_hit' in review_master.columns else 0}
- Most prominent keyword: {top_keyword_name}
- Most prominent keyword group: {top_group_name}

2. Subset observations
- The strongest subset-specific lexical concentration observed in the current outputs is: {strongest_subset_desc}

3. Repeated pattern observations
- Number of repeated sentence patterns retained (count >= 2): {repeated_sentence_count}

4. Suggested uses
- Use top_keywords_relevance_4.csv to identify lexically salient high-relevance patterns.
- Use top_keywords_africa_corps_cases.csv to compare Africa Corps lexicalization against Wagner-centered cases.
- Use top_keywords_successor_cases.csv to inspect transition / succession vocabulary.
- Use outlet_keyword_profile.csv and outlet_keyword_group_profile.csv to identify outlet-level lexical asymmetries.
- Use repeated_sentence_patterns.csv to identify potentially formulaic or recurrent representational constructions.
""".strip()

    with open(SUMMARY_DIR / "corpus4_lexical_findings.txt", "w", encoding="utf-8") as f:
        f.write(lexical_findings_text)

    print("\n=== CORPUS4 DIAGNOSTICS ===\n")
    print(f"Review master rows: {len(review_master)}")
    print(f"KWIC all rows: {len(kwic_all)}")
    print(f"Keyword summary rows: {len(kwic_keyword_summary)}")
    print(f"Keyword group summary rows: {len(kwic_group_summary)}")
    print(f"Top keyword: {top_keyword_name} ({top_keyword_matches} matches)")
    print(f"Top keyword group: {top_group_name} ({top_group_matches} matches)")
    print(f"Repeated sentence patterns retained: {repeated_sentence_count}")
    print("\nSaved outputs to:")
    print(f"- {LEXICAL_DIR}")
    print(f"- {FIGURES_DIR}")
    print(f"- {SUMMARY_DIR}")

if __name__ == "__main__":
    main()