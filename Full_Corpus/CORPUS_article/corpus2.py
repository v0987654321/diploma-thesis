import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_REVIEW_MASTER = SCRIPT_DIR / "data" / "review_master.csv"

SUBSETS_DIR = SCRIPT_DIR / "subsets"
EVIDENCE_DIR = SCRIPT_DIR / "evidence"
SUMMARY_DIR = SCRIPT_DIR / "summary"

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

SUBSET_GROUP_DIRS = [
    "relevance",
    "actors",
    "labels",
    "locations",
    "associations",
    "frames",
    "discourse",
    "review",
    "source_republication",
    "special",
]

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

EVIDENCE_BASE_COLS = [
    "Article_ID",
    "Outlet",
    "Date",
    "Headline",
    "Lead",
    "Target_Context_Excerpt",
    "Relevance",
    "Relevance_Label",
    "Actor_Mention",
    "Successor_Frame",
    "Dominant_Label",
    "Dominant_Location",
    "Main_Associated_Actor",
    "Stance_Support",
    "Legitimation_Support",
    "Dominant_Discourse_Support",
    "Any_Review_Flag",
    "Review_Flag_Count",
    "Review_Sources",
    "source_attributed_flag",
    "likely_republished_flag",
    "near_duplicate_flag",
    "near_duplicate_cross_outlet_flag",
    "has_gemini_conservative",
    "has_gemini_authoritative",
    "has_gemini_manual_verification",
    "has_gemini_high_confidence",
    "chapter52_priority_flag",
]

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
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for group_name in SUBSET_GROUP_DIRS:
        (SUBSETS_DIR / group_name).mkdir(parents=True, exist_ok=True)

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def split_sentences(text):
    text = safe_str(text)
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

def has_any(text, patterns):
    text = safe_str(text)
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def build_target_context_excerpt(row, window=2, max_sentences=8):
    parts = [
        safe_str(row.get("Headline")),
        safe_str(row.get("Lead")),
        safe_str(row.get("Body_Postclean")),
    ]
    full_text = " ".join([p for p in parts if p]).strip()
    sentences = split_sentences(full_text)

    if not sentences:
        return ""

    selected = set()

    for i, sent in enumerate(sentences):
        if has_any(sent, WGAC_PATTERNS):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            for j in range(start, end):
                selected.add(j)

    if not selected:
        excerpt = sentences[:max_sentences]
    else:
        excerpt = [sentences[i] for i in sorted(selected)][:max_sentences]

    return " ".join(excerpt).strip()

def export_subset(df, group_dir, filename):
    out = SUBSETS_DIR / group_dir / filename
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out

def export_evidence(df, filename):
    out = EVIDENCE_DIR / filename
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out

def make_evidence_pack(df, extra_cols=None):
    extra_cols = extra_cols or []
    cols = EVIDENCE_BASE_COLS + extra_cols
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()

def record_subset(summary_rows, group_name, filename, df):
    summary_rows.append({
        "group": group_name,
        "filename": filename,
        "rows": len(df)
    })

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    if not INPUT_REVIEW_MASTER.exists():
        raise FileNotFoundError(f"Missing required input: {INPUT_REVIEW_MASTER}")

    review_master = pd.read_csv(INPUT_REVIEW_MASTER, dtype={"Article_ID": str}, low_memory=False)

    review_master = ensure_columns(review_master, [
        "Article_ID", "Outlet", "Date", "Headline", "Lead", "Body_Postclean",
        "Relevance", "Relevance_Label",
        "Actor_Mention", "Successor_Frame", "Dominant_Label",
        "Dominant_Location", "Main_Associated_Actor",
        "Stance_Support", "Legitimation_Support", "Dominant_Discourse_Support",
        "Any_Review_Flag", "Review_Flag_Count", "Review_Sources",
        "source_attributed_flag", "likely_republished_flag", "near_duplicate_flag",
        "near_duplicate_cross_outlet_flag",
        "has_gemini_conservative", "has_gemini_authoritative",
        "has_gemini_manual_verification", "has_gemini_high_confidence",
        "chapter52_priority_flag", *FRAME_COLS
    ], fill_value=None)

    review_master["Target_Context_Excerpt"] = review_master.apply(build_target_context_excerpt, axis=1)

    summary_rows = []

    # =========================
    # RELEVANCE SUBSETS
    # =========================
    subset_relevance_4 = review_master[review_master["Relevance"].fillna(0).astype(int) == 4].copy()
    export_subset(subset_relevance_4, "relevance", "subset_relevance_4.csv")
    record_subset(summary_rows, "relevance", "subset_relevance_4.csv", subset_relevance_4)

    subset_relevance_3plus = review_master[review_master["Relevance"].fillna(0).astype(int).isin([3, 4])].copy()
    export_subset(subset_relevance_3plus, "relevance", "subset_relevance_3plus.csv")
    record_subset(summary_rows, "relevance", "subset_relevance_3plus.csv", subset_relevance_3plus)

    subset_relevance_2plus = review_master[review_master["Relevance"].fillna(0).astype(int).isin([2, 3, 4])].copy()
    export_subset(subset_relevance_2plus, "relevance", "subset_relevance_2plus.csv")
    record_subset(summary_rows, "relevance", "subset_relevance_2plus.csv", subset_relevance_2plus)

    # =========================
    # ACTOR / TRANSITION SUBSETS
    # =========================
    subset_africa_corps = review_master[review_master["Actor_Mention"].fillna(0).astype(int).isin([2, 3])].copy()
    export_subset(subset_africa_corps, "actors", "subset_africa_corps_cases.csv")
    record_subset(summary_rows, "actors", "subset_africa_corps_cases.csv", subset_africa_corps)

    subset_wagner_only = review_master[review_master["Actor_Mention"].fillna(0).astype(int) == 1].copy()
    export_subset(subset_wagner_only, "actors", "subset_wagner_only_cases.csv")
    record_subset(summary_rows, "actors", "subset_wagner_only_cases.csv", subset_wagner_only)

    subset_both = review_master[review_master["Actor_Mention"].fillna(0).astype(int) == 3].copy()
    export_subset(subset_both, "actors", "subset_both_actor_cases.csv")
    record_subset(summary_rows, "actors", "subset_both_actor_cases.csv", subset_both)

    subset_successor = review_master[review_master["Successor_Frame"].fillna(0).astype(int) == 1].copy()
    export_subset(subset_successor, "actors", "subset_successor_cases.csv")
    record_subset(summary_rows, "actors", "subset_successor_cases.csv", subset_successor)

    # =========================
    # LABEL SUBSETS
    # =========================
    label_map = {
        1: "subset_label_mercenaries.csv",
        2: "subset_label_instructors.csv",
        3: "subset_label_allies_partners.csv",
        4: "subset_label_foreign_occupying.csv",
        5: "subset_label_neutral_designation.csv",
        6: "subset_label_multiple.csv",
    }
    for code, filename in label_map.items():
        df_sub = review_master[review_master["Dominant_Label"].fillna(0).astype(int) == code].copy()
        export_subset(df_sub, "labels", filename)
        record_subset(summary_rows, "labels", filename, df_sub)

    # =========================
    # LOCATION SUBSETS
    # =========================
    location_map = {
        1: "subset_location_mali.csv",
        2: "subset_location_other_africa.csv",
        3: "subset_location_ukraine.csv",
        4: "subset_location_other.csv",
        5: "subset_location_mali_plus_other.csv",
    }
    for code, filename in location_map.items():
        df_sub = review_master[review_master["Dominant_Location"].fillna(0).astype(int) == code].copy()
        export_subset(df_sub, "locations", filename)
        record_subset(summary_rows, "locations", filename, df_sub)

    # =========================
    # ASSOCIATION SUBSETS
    # =========================
    assoc_map = {
        1: "subset_assoc_malian_army.csv",
        2: "subset_assoc_russia.csv",
        3: "subset_assoc_france.csv",
        4: "subset_assoc_un_minusma.csv",
        5: "subset_assoc_ecowas.csv",
        6: "subset_assoc_civilians.csv",
        7: "subset_assoc_jihadists.csv",
        8: "subset_assoc_west_broad.csv",
        9: "subset_assoc_unclear.csv",
        10: "subset_assoc_other.csv",
    }
    for code, filename in assoc_map.items():
        df_sub = review_master[review_master["Main_Associated_Actor"].fillna(0).astype(int) == code].copy()
        export_subset(df_sub, "associations", filename)
        record_subset(summary_rows, "associations", filename, df_sub)

    # =========================
    # FRAME SUBSETS
    # =========================
    for frame in FRAME_COLS:
        if frame in review_master.columns:
            df_sub = review_master[review_master[frame].fillna(0).astype(int) == 1].copy()
            filename = f"subset_frame_{frame.lower()}.csv"
            export_subset(df_sub, "frames", filename)
            record_subset(summary_rows, "frames", filename, df_sub)

    # =========================
    # DISCOURSE SUBSETS
    # =========================
    discourse_map = {
        1: "subset_discourse_sovereignty.csv",
        2: "subset_discourse_security.csv",
        3: "subset_discourse_violence_abuse.csv",
        4: "subset_discourse_geopolitics.csv",
        5: "subset_discourse_technocratic.csv",
        6: "subset_discourse_mixed.csv",
    }
    for code, filename in discourse_map.items():
        df_sub = review_master[review_master["Dominant_Discourse_Support"].fillna(0).astype(int) == code].copy()
        export_subset(df_sub, "discourse", filename)
        record_subset(summary_rows, "discourse", filename, df_sub)

    # =========================
    # REVIEW SUBSETS
    # =========================
    subset_review_flagged = review_master[review_master["Any_Review_Flag"].fillna(0).astype(int) == 1].copy()
    export_subset(subset_review_flagged, "review", "subset_review_flagged.csv")
    record_subset(summary_rows, "review", "subset_review_flagged.csv", subset_review_flagged)

    subset_high_review_burden = review_master[review_master["Review_Flag_Count"].fillna(0).astype(int) >= 3].copy()
    export_subset(subset_high_review_burden, "review", "subset_high_review_burden.csv")
    record_subset(summary_rows, "review", "subset_high_review_burden.csv", subset_high_review_burden)

    if "has_gemini_manual_verification" in review_master.columns:
        subset_manual_verification = review_master[review_master["has_gemini_manual_verification"].fillna(0).astype(int) == 1].copy()
        export_subset(subset_manual_verification, "review", "subset_manual_verification.csv")
        record_subset(summary_rows, "review", "subset_manual_verification.csv", subset_manual_verification)

    if "has_gemini_high_confidence" in review_master.columns:
        subset_high_conf = review_master[review_master["has_gemini_high_confidence"].fillna(0).astype(int) == 1].copy()
        export_subset(subset_high_conf, "review", "subset_high_confidence.csv")
        record_subset(summary_rows, "review", "subset_high_confidence.csv", subset_high_conf)

    # =========================
    # SOURCE / REPUBLICATION SUBSETS
    # =========================
    source_subset_map = {
        "source_attributed_flag": "subset_source_attributed.csv",
        "explicit_external_source_flag": "subset_explicit_external_source.csv",
        "explicit_malian_media_reference_flag": "subset_malian_media_reference.csv",
        "likely_republished_flag": "subset_likely_republished.csv",
        "near_duplicate_flag": "subset_near_duplicate.csv",
        "near_duplicate_cross_outlet_flag": "subset_cross_outlet_duplicate.csv",
    }

    for col, filename in source_subset_map.items():
        if col in review_master.columns:
            df_sub = review_master[review_master[col].fillna(0).astype(int) == 1].copy()
            export_subset(df_sub, "source_republication", filename)
            record_subset(summary_rows, "source_republication", filename, df_sub)

    # =========================
    # SPECIAL SUBSETS
    # =========================
    subset_ch52 = review_master[review_master["chapter52_priority_flag"].fillna(0).astype(int) == 1].copy()
    export_subset(subset_ch52, "special", "subset_chapter52_priority.csv")
    record_subset(summary_rows, "special", "subset_chapter52_priority.csv", subset_ch52)

    subset_africa_corps_rel4 = review_master[
        (review_master["Actor_Mention"].fillna(0).astype(int).isin([2, 3])) &
        (review_master["Relevance"].fillna(0).astype(int) == 4)
    ].copy()
    export_subset(subset_africa_corps_rel4, "special", "subset_africa_corps_relevance4.csv")
    record_subset(summary_rows, "special", "subset_africa_corps_relevance4.csv", subset_africa_corps_rel4)

    subset_successor_rel4 = review_master[
        (review_master["Successor_Frame"].fillna(0).astype(int) == 1) &
        (review_master["Relevance"].fillna(0).astype(int) == 4)
    ].copy()
    export_subset(subset_successor_rel4, "special", "subset_successor_relevance4.csv")
    record_subset(summary_rows, "special", "subset_successor_relevance4.csv", subset_successor_rel4)

    if "Human_Rights_Abuse" in review_master.columns:
        subset_hra_rel4 = review_master[
            (review_master["Human_Rights_Abuse"].fillna(0).astype(int) == 1) &
            (review_master["Relevance"].fillna(0).astype(int) == 4)
        ].copy()
        export_subset(subset_hra_rel4, "special", "subset_hra_relevance4.csv")
        record_subset(summary_rows, "special", "subset_hra_relevance4.csv", subset_hra_rel4)

    if "Security_Effectiveness" in review_master.columns:
        subset_sec_rel4 = review_master[
            (review_master["Security_Effectiveness"].fillna(0).astype(int) == 1) &
            (review_master["Relevance"].fillna(0).astype(int) == 4)
        ].copy()
        export_subset(subset_sec_rel4, "special", "subset_security_effectiveness_relevance4.csv")
        record_subset(summary_rows, "special", "subset_security_effectiveness_relevance4.csv", subset_sec_rel4)

    if "Geopolitical_Rivalry" in review_master.columns:
        subset_geo_rel4 = review_master[
            (review_master["Geopolitical_Rivalry"].fillna(0).astype(int) == 1) &
            (review_master["Relevance"].fillna(0).astype(int) == 4)
        ].copy()
        export_subset(subset_geo_rel4, "special", "subset_geopolitics_relevance4.csv")
        record_subset(summary_rows, "special", "subset_geopolitics_relevance4.csv", subset_geo_rel4)

    if "likely_republished_flag" in review_master.columns:
        subset_rep_rel3plus = review_master[
            (review_master["likely_republished_flag"].fillna(0).astype(int) == 1) &
            (review_master["Relevance"].fillna(0).astype(int).isin([3, 4]))
        ].copy()
        export_subset(subset_rep_rel3plus, "special", "subset_source_republished_relevance3plus.csv")
        record_subset(summary_rows, "special", "subset_source_republished_relevance3plus.csv", subset_rep_rel3plus)

    # =========================
    # EVIDENCE PACKS
    # =========================
    evidence_specs = [
        ("evidence_africa_corps.csv", subset_africa_corps, ["Successor_Frame", "Dominant_Discourse_Support"]),
        ("evidence_successor_cases.csv", subset_successor, ["Actor_Mention", "Dominant_Discourse_Support"]),
        ("evidence_human_rights_abuse.csv", review_master[review_master["Human_Rights_Abuse"].fillna(0).astype(int) == 1].copy() if "Human_Rights_Abuse" in review_master.columns else pd.DataFrame(), ["Human_Rights_Abuse", "Dominant_Discourse_Support"]),
        ("evidence_security_effectiveness.csv", review_master[review_master["Security_Effectiveness"].fillna(0).astype(int) == 1].copy() if "Security_Effectiveness" in review_master.columns else pd.DataFrame(), ["Security_Effectiveness", "Dominant_Discourse_Support"]),
        ("evidence_geopolitical_rivalry.csv", review_master[review_master["Geopolitical_Rivalry"].fillna(0).astype(int) == 1].copy() if "Geopolitical_Rivalry" in review_master.columns else pd.DataFrame(), ["Geopolitical_Rivalry", "Dominant_Discourse_Support"]),
        ("evidence_source_attributed.csv", review_master[review_master["source_attributed_flag"].fillna(0).astype(int) == 1].copy() if "source_attributed_flag" in review_master.columns else pd.DataFrame(), ["source_attributed_flag", "likely_republished_flag"]),
        ("evidence_likely_republished.csv", review_master[review_master["likely_republished_flag"].fillna(0).astype(int) == 1].copy() if "likely_republished_flag" in review_master.columns else pd.DataFrame(), ["source_attributed_flag", "likely_republished_flag", "near_duplicate_flag"]),
        ("evidence_review_flagged.csv", subset_review_flagged, ["Any_Review_Flag", "Review_Flag_Count"]),
        ("evidence_chapter52_priority.csv", subset_ch52, ["chapter52_priority_flag", "Dominant_Discourse_Support"]),
    ]

    for filename, df_sub, extra_cols in evidence_specs:
        if df_sub is not None and not df_sub.empty:
            ev = make_evidence_pack(df_sub, extra_cols=extra_cols)
            export_evidence(ev, filename)
            record_subset(summary_rows, "evidence", filename, ev)

    # =========================
    # SUMMARY
    # =========================
    subset_summary = pd.DataFrame(summary_rows).sort_values(["group", "filename"]).reset_index(drop=True)
    subset_summary_out = SUMMARY_DIR / "corpus2_subset_summary.csv"
    subset_summary.to_csv(subset_summary_out, index=False, encoding="utf-8-sig")

    total_subsets = len(subset_summary[subset_summary["group"] != "evidence"])
    total_evidence = len(subset_summary[subset_summary["group"] == "evidence"])

    largest_subset = subset_summary.sort_values("rows", ascending=False).iloc[0] if not subset_summary.empty else None

    summary_text = f"""
CORPUS2 SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Input:
- review_master.csv rows: {len(review_master)}

Outputs:
- subset files created: {total_subsets}
- evidence packs created: {total_evidence}

Largest output:
- {largest_subset['filename'] if largest_subset is not None else 'N/A'} ({largest_subset['rows'] if largest_subset is not None else 0} rows)

Key subset sizes:
- relevance 4: {len(subset_relevance_4)}
- relevance 3+: {len(subset_relevance_3plus)}
- relevance 2+: {len(subset_relevance_2plus)}
- Africa Corps cases: {len(subset_africa_corps)}
- Wagner-only cases: {len(subset_wagner_only)}
- both-actor cases: {len(subset_both)}
- successor cases: {len(subset_successor)}
- review-flagged cases: {len(subset_review_flagged)}
- chapter52 priority cases: {len(subset_ch52)}

Files:
- summary/corpus2_subset_summary.csv
- subsets/.../*.csv
- evidence/*.csv
""".strip()

    with open(SUMMARY_DIR / "corpus2_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open(SUMMARY_DIR / "corpus2_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # =========================
    # CONSOLE OUTPUT
    # =========================
    print("\n=== CORPUS2 DIAGNOSTICS ===\n")
    print(f"Input review_master rows: {len(review_master)}")
    print(f"Subset files created: {total_subsets}")
    print(f"Evidence packs created: {total_evidence}")
    print(f"Relevance 4 subset: {len(subset_relevance_4)}")
    print(f"Relevance 3+ subset: {len(subset_relevance_3plus)}")
    print(f"Africa Corps subset: {len(subset_africa_corps)}")
    print(f"Successor subset: {len(subset_successor)}")
    print(f"Review-flagged subset: {len(subset_review_flagged)}")
    print(f"Chapter 5.2 priority subset: {len(subset_ch52)}")
    print(f"\nSaved outputs to:")
    print(f"- {SUBSETS_DIR}")
    print(f"- {EVIDENCE_DIR}")
    print(f"- {SUMMARY_DIR}")


if __name__ == "__main__":
    main()