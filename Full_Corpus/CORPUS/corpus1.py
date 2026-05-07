import os
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

PILOT_DIR = ROOT_DIR / "pilot"
PILOT_DATA_DIR = PILOT_DIR / "data"
GEMINI_DATA_DIR = PILOT_DIR / "GEMINI" / "data"
LOCAL_DATA_DIR = PILOT_DIR / "LOCAL" / "data"

CORPUS_DATA_DIR = SCRIPT_DIR / "data"
CORPUS_TABLES_DIR = SCRIPT_DIR / "tables"
CORPUS_FIGURES_DIR = SCRIPT_DIR / "figures"
CORPUS_SCHEMATA_DIR = SCRIPT_DIR / "schemata"
CORPUS_SUMMARY_DIR = SCRIPT_DIR / "summary"

# Files to copy from pilot/data
PILOT_DATA_FILES = [
    "postConsolidated.csv",
    "postDiagnostic.csv",
    "postStepB.csv",
]

# Files to copy from pilot/GEMINI/data
GEMINI_DATA_FILES = [
    "gemini_parsed_outputs.csv",
    "gemini_parse_errors.csv",
    "final_llm_coding_working.csv",
    "final_llm_merge_issues.csv",
    "final_conservative_adjudicated_table.csv",
    "final_llm_authoritative_table.csv",
    "final_manual_verification_table.csv",
    "final_high_confidence_coding_table.csv",
]

# Optional LOCAL files
LOCAL_DATA_FILES = [
    "local_parsed_outputs.csv",
    "local_parse_errors.csv",
    "final_local_coding_working.csv",
    "final_local_merge_issues.csv",
    "final_local_conservative_adjudicated_table.csv",
    "final_local_llm_authoritative_table.csv",
    "final_local_manual_verification_table.csv",
    "final_local_high_confidence_coding_table.csv",
]

OUTLET_LABELS = {
    "malijet.com": "Malijet",
    "maliweb.net": "Maliweb",
    "bamada.net": "Bamada",
    "mali24.info": "Mali24",
    "studiotamani.org": "Studio Tamani",
    "journaldumali.com": "Journal du Mali",
    "malitribune.com": "Mali Tribune",
    "info-matin.ml": "Info Matin",
    "lejalon.com": "Le Jalon",
    "lessor.ml": "L’Essor",
    "malikonews.com": "Maliko News",
}

ACTOR_MENTION_LABELS = {
    1: "Wagner Group",
    2: "Africa Corps",
    3: "Both explicitly",
    4: "Indirect contractors/forces",
    5: "Cannot determine",
}

DOMINANT_LABEL_LABELS = {
    1: "Mercenaries",
    2: "Instructors/advisers",
    3: "Allies/partners",
    4: "Foreign/occupying forces",
    5: "Neutral designation",
    6: "Multiple/no clear dominance",
}

DOMINANT_LOCATION_LABELS = {
    1: "Mali",
    2: "Other African countries",
    3: "Ukraine",
    4: "Other location",
    5: "Mali and other location",
}

MAIN_ASSOCIATED_ACTOR_LABELS = {
    1: "Malian army / junta",
    2: "Russia / Russian state",
    3: "France",
    4: "UN / MINUSMA",
    5: "ECOWAS / regional actors",
    6: "Local civilians",
    7: "Jihadist / terrorist groups",
    8: "Western states broadly",
    9: "No clear dominant actor",
    10: "Other",
}

STANCE_SUPPORT_LABELS = {
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Mixed/ambivalent",
    5: "Cannot determine",
}

LEGITIMATION_SUPPORT_LABELS = {
    1: "Delegitimized",
    2: "Normalized / implicitly legitimized",
    3: "Explicitly legitimized",
    4: "Cannot be determined",
}

DISCOURSE_SUPPORT_LABELS = {
    1: "Sovereignty and emancipation",
    2: "Security and stabilization",
    3: "Violence and abuse",
    4: "Geopolitical competition",
    5: "Technocratic / factual reporting",
    6: "Mixed / no clear dominance",
}

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
    for p in [
        CORPUS_DATA_DIR,
        CORPUS_TABLES_DIR,
        CORPUS_FIGURES_DIR,
        CORPUS_SCHEMATA_DIR,
        CORPUS_SUMMARY_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src_dir, filename, target_dir, found_records):
    src = src_dir / filename
    dst = target_dir / filename

    if src.exists():
        shutil.copy2(src, dst)
        found_records.append({
            "source_dir": str(src_dir),
            "filename": filename,
            "copied": 1
        })
        print(f"Copied: {src} -> {dst}")
        return True
    else:
        found_records.append({
            "source_dir": str(src_dir),
            "filename": filename,
            "copied": 0
        })
        print(f"Missing: {src}")
        return False


def read_csv_if_exists(path, dtype=None):
    if path.exists():
        return pd.read_csv(path, dtype=dtype, low_memory=False)
    return None


def ensure_columns(df, columns, fill_value=None):
    if df is None:
        return None
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


def normalize_article_id(x):
    raw = safe_str(x)
    if not raw:
        return None
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(6)


def add_presence_flag(df, colname, yesno):
    if yesno:
        df[colname] = 1
    else:
        df[colname] = 0
    return df


def export_table(df, filename):
    out = CORPUS_TABLES_DIR / filename
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def export_data(df, filename):
    out = CORPUS_DATA_DIR / filename
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_bar_chart(series, title, xlabel, ylabel, filename, rotate=False):
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = CORPUS_FIGURES_DIR / filename
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def build_distribution_table(df, col, label_map=None, denominator=None):
    temp = df[col].value_counts(dropna=False).reset_index()
    temp.columns = ["code", "count"]

    if label_map is not None:
        temp["label"] = temp["code"].map(label_map)
    else:
        temp["label"] = temp["code"].apply(lambda x: safe_str(x))

    if denominator is None:
        denominator = len(df)

    temp["percent"] = temp["count"] / denominator * 100 if denominator else 0
    temp["variable"] = col
    return temp[["variable", "code", "label", "count", "percent"]]


def boolean_profile(df, col, denominator=None):
    if col not in df.columns:
        return pd.DataFrame([{
            "variable": col,
            "count_1": 0,
            "percent_1": 0.0
        }])

    if denominator is None:
        denominator = len(df)

    count_1 = (df[col].fillna(0).astype(int) == 1).sum()
    pct_1 = (count_1 / denominator * 100) if denominator else 0.0

    return pd.DataFrame([{
        "variable": col,
        "count_1": count_1,
        "percent_1": pct_1
    }])


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    copied_manifest = []

    print("\n=== CORPUS1: copying source files ===\n")

    for fn in PILOT_DATA_FILES:
        copy_if_exists(PILOT_DATA_DIR, fn, CORPUS_DATA_DIR, copied_manifest)

    for fn in GEMINI_DATA_FILES:
        copy_if_exists(GEMINI_DATA_DIR, fn, CORPUS_DATA_DIR, copied_manifest)

    for fn in LOCAL_DATA_FILES:
        copy_if_exists(LOCAL_DATA_DIR, fn, CORPUS_DATA_DIR, copied_manifest)

    manifest_df = pd.DataFrame(copied_manifest)
    export_data(manifest_df, "corpus1_manifest.csv")

    print("\n=== CORPUS1: loading copied files ===\n")

    post_consolidated = read_csv_if_exists(CORPUS_DATA_DIR / "postConsolidated.csv", dtype={"Article_ID": str})
    post_diagnostic = read_csv_if_exists(CORPUS_DATA_DIR / "postDiagnostic.csv", dtype={"Article_ID": str})
    post_stepb = read_csv_if_exists(CORPUS_DATA_DIR / "postStepB.csv", dtype={"article_id": str})

    gemini_working = read_csv_if_exists(CORPUS_DATA_DIR / "final_llm_coding_working.csv", dtype={"Article_ID": str})
    gemini_conservative = read_csv_if_exists(CORPUS_DATA_DIR / "final_conservative_adjudicated_table.csv", dtype={"Article_ID": str})
    gemini_authoritative = read_csv_if_exists(CORPUS_DATA_DIR / "final_llm_authoritative_table.csv", dtype={"Article_ID": str})
    gemini_manual = read_csv_if_exists(CORPUS_DATA_DIR / "final_manual_verification_table.csv", dtype={"Article_ID": str})
    gemini_high_conf = read_csv_if_exists(CORPUS_DATA_DIR / "final_high_confidence_coding_table.csv", dtype={"Article_ID": str})
    gemini_parsed = read_csv_if_exists(CORPUS_DATA_DIR / "gemini_parsed_outputs.csv", dtype={"Article_ID": str})
    gemini_parse_errors = read_csv_if_exists(CORPUS_DATA_DIR / "gemini_parse_errors.csv", dtype={"Article_ID": str})
    gemini_merge_issues = read_csv_if_exists(CORPUS_DATA_DIR / "final_llm_merge_issues.csv", dtype={"Article_ID": str})

    if post_consolidated is None:
        raise FileNotFoundError("postConsolidated.csv is required for corpus1.py")

    post_consolidated["Article_ID"] = post_consolidated["Article_ID"].apply(normalize_article_id)

    review_master = post_consolidated.copy()
    review_master["has_pipeline_consolidated"] = 1

    if post_diagnostic is not None:
        post_diagnostic["Article_ID"] = post_diagnostic["Article_ID"].apply(normalize_article_id)
        diag_keep = [c for c in post_diagnostic.columns if c not in review_master.columns or c == "Article_ID"]
        review_master = review_master.merge(
            post_diagnostic[diag_keep],
            on="Article_ID",
            how="left",
            suffixes=("", "_diag")
        )
        review_master["has_pipeline_diagnostic"] = 1
    else:
        review_master["has_pipeline_diagnostic"] = 0

    if post_stepb is not None:
        post_stepb["article_id"] = post_stepb["article_id"].apply(normalize_article_id)
        stepb_cols = [
            "article_id",
            "stepB_relevance_filter_mode",
            "explicit_external_source_flag",
            "explicit_external_source_type",
            "explicit_external_source_name",
            "explicit_external_source_count",
            "author_external_source_flag",
            "author_external_source_name",
            "author_external_source_type",
            "explicit_malian_media_reference_flag",
            "explicit_malian_media_reference_names",
            "explicit_malian_media_reference_count",
            "explicit_malian_media_reference_type",
            "republication_phrase_flag",
            "republication_phrase_count",
            "attribution_phrase_flag",
            "attribution_phrase_count",
            "source_attributed_flag",
            "republication_index",
            "likely_republished_basis",
            "republication_confidence",
            "likely_republished_flag",
            "near_duplicate_flag",
            "near_duplicate_cluster_id",
            "near_duplicate_match_count",
            "near_duplicate_cross_outlet_flag",
            "near_duplicate_same_outlet_flag",
            "near_duplicate_top_match_id",
            "near_duplicate_top_match_outlet",
            "near_duplicate_top_match_score",
        ]
        stepb_cols = [c for c in stepb_cols if c in post_stepb.columns]
        stepb_sub = post_stepb[stepb_cols].copy().rename(columns={"article_id": "Article_ID"})
        review_master = review_master.merge(stepb_sub, on="Article_ID", how="left")
        review_master["has_stepB"] = 1
    else:
        review_master["has_stepB"] = 0

    if gemini_working is not None:
        gemini_working["Article_ID"] = gemini_working["Article_ID"].apply(normalize_article_id)
        review_master = review_master.merge(
            gemini_working[["Article_ID"]].drop_duplicates().assign(has_gemini_working=1),
            on="Article_ID",
            how="left"
        )
    else:
        review_master["has_gemini_working"] = 0

    if gemini_conservative is not None:
        gemini_conservative["Article_ID"] = gemini_conservative["Article_ID"].apply(normalize_article_id)
        review_master = review_master.merge(
            gemini_conservative[["Article_ID"]].drop_duplicates().assign(has_gemini_conservative=1),
            on="Article_ID",
            how="left"
        )
    else:
        review_master["has_gemini_conservative"] = 0

    if gemini_authoritative is not None:
        gemini_authoritative["Article_ID"] = gemini_authoritative["Article_ID"].apply(normalize_article_id)
        review_master = review_master.merge(
            gemini_authoritative[["Article_ID"]].drop_duplicates().assign(has_gemini_authoritative=1),
            on="Article_ID",
            how="left"
        )
    else:
        review_master["has_gemini_authoritative"] = 0

    if gemini_manual is not None:
        gemini_manual["Article_ID"] = gemini_manual["Article_ID"].apply(normalize_article_id)
        review_master = review_master.merge(
            gemini_manual[["Article_ID"]].drop_duplicates().assign(has_gemini_manual_verification=1),
            on="Article_ID",
            how="left"
        )
    else:
        review_master["has_gemini_manual_verification"] = 0

    if gemini_high_conf is not None:
        gemini_high_conf["Article_ID"] = gemini_high_conf["Article_ID"].apply(normalize_article_id)
        review_master = review_master.merge(
            gemini_high_conf[["Article_ID"]].drop_duplicates().assign(has_gemini_high_confidence=1),
            on="Article_ID",
            how="left"
        )
    else:
        review_master["has_gemini_high_confidence"] = 0

    if gemini_parsed is not None:
        gemini_parsed["Article_ID"] = gemini_parsed["Article_ID"].apply(normalize_article_id)
        review_master = review_master.merge(
            gemini_parsed[["Article_ID"]].drop_duplicates().assign(has_gemini_parsed=1),
            on="Article_ID",
            how="left"
        )
    else:
        review_master["has_gemini_parsed"] = 0

    for col in [
        "has_gemini_working",
        "has_gemini_conservative",
        "has_gemini_authoritative",
        "has_gemini_manual_verification",
        "has_gemini_high_confidence",
        "has_gemini_parsed",
    ]:
        if col not in review_master.columns:
            review_master[col] = 0
        review_master[col] = review_master[col].fillna(0).astype(int)

    # =========================
    # DERIVED FLAGS
    # =========================
    review_master["is_relevance_2plus"] = review_master["Relevance"].fillna(0).astype(int).isin([2, 3, 4]).astype(int)
    review_master["is_relevance_3plus"] = review_master["Relevance"].fillna(0).astype(int).isin([3, 4]).astype(int)
    review_master["is_relevance_4"] = (review_master["Relevance"].fillna(0).astype(int) == 4).astype(int)

    if "Actor_Mention" in review_master.columns:
        review_master["is_africa_corps_case"] = review_master["Actor_Mention"].fillna(0).astype(int).isin([2, 3]).astype(int)
        review_master["is_wagner_only_case"] = (review_master["Actor_Mention"].fillna(0).astype(int) == 1).astype(int)
        review_master["is_both_actor_case"] = (review_master["Actor_Mention"].fillna(0).astype(int) == 3).astype(int)
    else:
        review_master["is_africa_corps_case"] = 0
        review_master["is_wagner_only_case"] = 0
        review_master["is_both_actor_case"] = 0

    if "Successor_Frame" in review_master.columns:
        review_master["is_successor_case"] = (review_master["Successor_Frame"].fillna(0).astype(int) == 1).astype(int)
    else:
        review_master["is_successor_case"] = 0

    if "Any_Review_Flag" in review_master.columns:
        review_master["is_high_review_case"] = (review_master["Any_Review_Flag"].fillna(0).astype(int) == 1).astype(int)
    else:
        review_master["is_high_review_case"] = 0

    if "source_attributed_flag" in review_master.columns:
        review_master["has_stepB_source_signal"] = (review_master["source_attributed_flag"].fillna(0).astype(int) == 1).astype(int)
    else:
        review_master["has_stepB_source_signal"] = 0

    if "likely_republished_flag" in review_master.columns:
        review_master["is_likely_republished"] = (review_master["likely_republished_flag"].fillna(0).astype(int) == 1).astype(int)
    else:
        review_master["is_likely_republished"] = 0

    review_master["chapter52_priority_flag"] = (
        (
            review_master["is_relevance_3plus"] == 1
        ) & (
            (review_master["is_successor_case"] == 1) |
            (review_master["is_africa_corps_case"] == 1) |
            (review_master.get("Human_Rights_Abuse", pd.Series(0, index=review_master.index)).fillna(0).astype(int) == 1) |
            (review_master.get("Security_Effectiveness", pd.Series(0, index=review_master.index)).fillna(0).astype(int) == 1) |
            (review_master.get("Dominant_Discourse_Support", pd.Series(0, index=review_master.index)).fillna(0).astype(int).isin([2, 3, 4])) |
            (review_master["has_stepB_source_signal"] == 1)
        )
    ).astype(int)

    export_data(review_master, "review_master.csv")

    # =========================
    # CORPUS OVERVIEW
    # =========================
    overview_rows = []

    total_rows = len(review_master)
    relevance1 = (review_master["Relevance"].fillna(0).astype(int) == 1).sum() if "Relevance" in review_master.columns else 0
    relevance2 = (review_master["Relevance"].fillna(0).astype(int) == 2).sum() if "Relevance" in review_master.columns else 0
    relevance3 = (review_master["Relevance"].fillna(0).astype(int) == 3).sum() if "Relevance" in review_master.columns else 0
    relevance4 = (review_master["Relevance"].fillna(0).astype(int) == 4).sum() if "Relevance" in review_master.columns else 0

    overview_rows.extend([
        {"metric": "retrieved_articles", "value": total_rows},
        {"metric": "relevance_1", "value": relevance1},
        {"metric": "relevance_2", "value": relevance2},
        {"metric": "relevance_3", "value": relevance3},
        {"metric": "relevance_4", "value": relevance4},
        {"metric": "relevance_2plus", "value": relevance2 + relevance3 + relevance4},
        {"metric": "relevance_3plus", "value": relevance3 + relevance4},
        {"metric": "outlets_n", "value": review_master["Outlet"].nunique() if "Outlet" in review_master.columns else 0},
        {"metric": "any_review_flag_1", "value": (review_master["Any_Review_Flag"].fillna(0).astype(int) == 1).sum() if "Any_Review_Flag" in review_master.columns else 0},
        {"metric": "chapter52_priority_cases", "value": review_master["chapter52_priority_flag"].sum()},
        {"metric": "stepB_rows_present", "value": int(post_stepb is not None)},
        {"metric": "gemini_working_present", "value": int(gemini_working is not None)},
        {"metric": "gemini_conservative_present", "value": int(gemini_conservative is not None)},
        {"metric": "gemini_authoritative_present", "value": int(gemini_authoritative is not None)},
        {"metric": "gemini_high_confidence_present", "value": int(gemini_high_conf is not None)},
    ])

    if gemini_authoritative is not None:
        overview_rows.append({"metric": "gemini_authoritative_rows", "value": len(gemini_authoritative)})
    if gemini_conservative is not None:
        overview_rows.append({"metric": "gemini_conservative_rows", "value": len(gemini_conservative)})
    if gemini_manual is not None:
        overview_rows.append({"metric": "gemini_manual_verification_rows", "value": len(gemini_manual)})
    if gemini_high_conf is not None:
        overview_rows.append({"metric": "gemini_high_confidence_rows", "value": len(gemini_high_conf)})
    if gemini_parsed is not None:
        overview_rows.append({"metric": "gemini_valid_parsed_rows", "value": len(gemini_parsed)})
    if gemini_parse_errors is not None:
        overview_rows.append({"metric": "gemini_parse_error_rows", "value": len(gemini_parse_errors)})
    if gemini_merge_issues is not None:
        overview_rows.append({"metric": "gemini_merge_issue_rows", "value": len(gemini_merge_issues)})

    corpus_overview = pd.DataFrame(overview_rows)
    export_data(corpus_overview, "corpus_overview.csv")
    export_table(corpus_overview, "corpus_overview_table.csv")

    # =========================
    # RELEVANCE PROFILE
    # =========================
    relevance_profile = build_distribution_table(
        review_master.assign(Relevance=review_master["Relevance"].fillna(0).astype(int)),
        "Relevance",
        label_map={
            1: "Not relevant",
            2: "Marginal mention only",
            3: "Substantively relevant",
            4: "Main topic"
        }
    )
    export_data(relevance_profile, "relevance_profile.csv")
    export_table(relevance_profile, "relevance_profile_table.csv")

    # =========================
    # OUTLET PROFILE
    # =========================
    if "Outlet" in review_master.columns:
        outlet_profile = review_master.groupby("Outlet", dropna=False).agg(
            n_total=("Article_ID", "count"),
            n_relevance_2plus=("is_relevance_2plus", "sum"),
            n_relevance_3plus=("is_relevance_3plus", "sum"),
            n_relevance_4=("is_relevance_4", "sum"),
            n_review_flag=("is_high_review_case", "sum"),
            n_chapter52_priority=("chapter52_priority_flag", "sum"),
        ).reset_index()

        outlet_profile["Outlet_Label"] = outlet_profile["Outlet"].map(OUTLET_LABELS).fillna(outlet_profile["Outlet"])
        outlet_profile["share_total_percent"] = outlet_profile["n_total"] / total_rows * 100 if total_rows else 0
        outlet_profile["share_relevance_3plus_percent"] = outlet_profile["n_relevance_3plus"] / max(1, relevance3 + relevance4) * 100
        outlet_profile = outlet_profile.sort_values("n_total", ascending=False)

        export_data(outlet_profile, "outlet_profile.csv")
        export_table(outlet_profile, "outlet_profile_table.csv")
    else:
        outlet_profile = pd.DataFrame()

    # =========================
    # REPRESENTATION PROFILE
    # =========================
    representation_parts = []

    if "Actor_Mention" in review_master.columns:
        representation_parts.append(build_distribution_table(
            review_master.assign(Actor_Mention=review_master["Actor_Mention"].fillna(0).astype(int)),
            "Actor_Mention",
            ACTOR_MENTION_LABELS
        ))

    if "Successor_Frame" in review_master.columns:
        successor_dist = build_distribution_table(
            review_master.assign(Successor_Frame=review_master["Successor_Frame"].fillna(0).astype(int)),
            "Successor_Frame",
            {0: "No", 1: "Yes"}
        )
        representation_parts.append(successor_dist)

    if "Dominant_Label" in review_master.columns:
        representation_parts.append(build_distribution_table(
            review_master.assign(Dominant_Label=review_master["Dominant_Label"].fillna(0).astype(int)),
            "Dominant_Label",
            DOMINANT_LABEL_LABELS
        ))

    if "Dominant_Location" in review_master.columns:
        representation_parts.append(build_distribution_table(
            review_master.assign(Dominant_Location=review_master["Dominant_Location"].fillna(0).astype(int)),
            "Dominant_Location",
            DOMINANT_LOCATION_LABELS
        ))

    if "Main_Associated_Actor" in review_master.columns:
        representation_parts.append(build_distribution_table(
            review_master.assign(Main_Associated_Actor=review_master["Main_Associated_Actor"].fillna(0).astype(int)),
            "Main_Associated_Actor",
            MAIN_ASSOCIATED_ACTOR_LABELS
        ))

    if "Stance_Support" in review_master.columns:
        representation_parts.append(build_distribution_table(
            review_master.assign(Stance_Support=review_master["Stance_Support"].fillna(0).astype(int)),
            "Stance_Support",
            STANCE_SUPPORT_LABELS
        ))

    if "Legitimation_Support" in review_master.columns:
        representation_parts.append(build_distribution_table(
            review_master.assign(Legitimation_Support=review_master["Legitimation_Support"].fillna(0).astype(int)),
            "Legitimation_Support",
            LEGITIMATION_SUPPORT_LABELS
        ))

    if "Dominant_Discourse_Support" in review_master.columns:
        representation_parts.append(build_distribution_table(
            review_master.assign(Dominant_Discourse_Support=review_master["Dominant_Discourse_Support"].fillna(0).astype(int)),
            "Dominant_Discourse_Support",
            DISCOURSE_SUPPORT_LABELS
        ))

    if representation_parts:
        representation_profile = pd.concat(representation_parts, axis=0, ignore_index=True)
    else:
        representation_profile = pd.DataFrame(columns=["variable", "code", "label", "count", "percent"])

    export_data(representation_profile, "representation_profile.csv")
    export_table(representation_profile, "representation_profile_table.csv")

    # =========================
    # FRAME PROFILE
    # =========================
    frame_rows = []
    rel2plus_denom = max(1, int(review_master["is_relevance_2plus"].sum()))
    rel3plus_denom = max(1, int(review_master["is_relevance_3plus"].sum()))
    rel4_denom = max(1, int(review_master["is_relevance_4"].sum()))

    for frame in FRAME_COLS:
        if frame in review_master.columns:
            count_total = (review_master[frame].fillna(0).astype(int) == 1).sum()
            count_rel2plus = ((review_master[frame].fillna(0).astype(int) == 1) & (review_master["is_relevance_2plus"] == 1)).sum()
            count_rel3plus = ((review_master[frame].fillna(0).astype(int) == 1) & (review_master["is_relevance_3plus"] == 1)).sum()
            count_rel4 = ((review_master[frame].fillna(0).astype(int) == 1) & (review_master["is_relevance_4"] == 1)).sum()

            frame_rows.append({
                "frame": frame,
                "count_total": count_total,
                "percent_total": count_total / total_rows * 100 if total_rows else 0,
                "count_relevance_2plus": count_rel2plus,
                "percent_relevance_2plus": count_rel2plus / rel2plus_denom * 100,
                "count_relevance_3plus": count_rel3plus,
                "percent_relevance_3plus": count_rel3plus / rel3plus_denom * 100,
                "count_relevance_4": count_rel4,
                "percent_relevance_4": count_rel4 / rel4_denom * 100,
            })

    frame_profile = pd.DataFrame(frame_rows)
    export_data(frame_profile, "frame_profile.csv")
    export_table(frame_profile, "frame_profile_table.csv")

    # =========================
    # REVIEW PROFILE
    # =========================
    review_profile_rows = []

    if "Any_Review_Flag" in review_master.columns:
        review_profile_rows.append({
            "metric": "Any_Review_Flag_1",
            "count": (review_master["Any_Review_Flag"].fillna(0).astype(int) == 1).sum(),
            "percent": (review_master["Any_Review_Flag"].fillna(0).astype(int) == 1).sum() / total_rows * 100 if total_rows else 0
        })

    if "Review_Flag_Count" in review_master.columns:
        rfc_dist = review_master["Review_Flag_Count"].fillna(0).astype(int).value_counts().reset_index()
        rfc_dist.columns = ["review_flag_count", "count"]
        rfc_dist["percent"] = rfc_dist["count"] / total_rows * 100 if total_rows else 0
        export_data(rfc_dist, "review_flag_count_distribution.csv")
        export_table(rfc_dist, "review_flag_count_distribution_table.csv")

    for step_col in [
        "Step2_Manual_Review",
        "Step3_Manual_Review",
        "Step4_Manual_Review",
        "Step5_Manual_Review",
        "Step6_Manual_Review",
        "Step7_Manual_Review",
        "Step8_Manual_Review",
    ]:
        if step_col in review_master.columns:
            count_1 = (review_master[step_col].fillna(0).astype(int) == 1).sum()
            review_profile_rows.append({
                "metric": step_col,
                "count": count_1,
                "percent": count_1 / total_rows * 100 if total_rows else 0
            })

    review_profile = pd.DataFrame(review_profile_rows)
    export_data(review_profile, "review_profile.csv")
    export_table(review_profile, "review_profile_table.csv")

    # =========================
    # SOURCE / REPUBLICATION PROFILE
    # =========================
    source_rows = []

    for col in [
        "explicit_external_source_flag",
        "author_external_source_flag",
        "explicit_malian_media_reference_flag",
        "republication_phrase_flag",
        "attribution_phrase_flag",
        "source_attributed_flag",
        "likely_republished_flag",
        "near_duplicate_flag",
        "near_duplicate_cross_outlet_flag",
    ]:
        if col in review_master.columns:
            count_1 = (review_master[col].fillna(0).astype(int) == 1).sum()
            source_rows.append({
                "metric": col,
                "count": count_1,
                "percent": count_1 / total_rows * 100 if total_rows else 0
            })

    if "republication_confidence" in review_master.columns:
        repconf = review_master["republication_confidence"].fillna("missing").value_counts().reset_index()
        repconf.columns = ["metric_value", "count"]
        repconf["metric"] = "republication_confidence"
        repconf["percent"] = repconf["count"] / total_rows * 100 if total_rows else 0
        repconf = repconf[["metric", "metric_value", "count", "percent"]]
    else:
        repconf = pd.DataFrame()

    if "likely_republished_basis" in review_master.columns:
        repbasis = review_master["likely_republished_basis"].fillna("missing").value_counts().reset_index()
        repbasis.columns = ["metric_value", "count"]
        repbasis["metric"] = "likely_republished_basis"
        repbasis["percent"] = repbasis["count"] / total_rows * 100 if total_rows else 0
        repbasis = repbasis[["metric", "metric_value", "count", "percent"]]
    else:
        repbasis = pd.DataFrame()

    source_profile = pd.DataFrame(source_rows)
    export_data(source_profile, "source_republication_profile.csv")
    export_table(source_profile, "source_republication_profile_table.csv")

    if not repconf.empty:
        export_data(repconf, "republication_confidence_profile.csv")
        export_table(repconf, "republication_confidence_profile_table.csv")

    if not repbasis.empty:
        export_data(repbasis, "republication_basis_profile.csv")
        export_table(repbasis, "republication_basis_profile_table.csv")

    # =========================
    # ADJUDICATION PROFILE
    # =========================
    adjudication_rows = []

    for col in [
        "has_gemini_working",
        "has_gemini_conservative",
        "has_gemini_authoritative",
        "has_gemini_manual_verification",
        "has_gemini_high_confidence",
        "has_gemini_parsed",
    ]:
        if col in review_master.columns:
            count_1 = (review_master[col].fillna(0).astype(int) == 1).sum()
            adjudication_rows.append({
                "metric": col,
                "count": count_1,
                "percent": count_1 / total_rows * 100 if total_rows else 0
            })

    adjudication_profile = pd.DataFrame(adjudication_rows)
    export_data(adjudication_profile, "adjudication_profile.csv")
    export_table(adjudication_profile, "adjudication_profile_table.csv")

    # =========================
    # FIGURES
    # =========================
    # Relevance
    if "Relevance" in review_master.columns:
        rel_plot = review_master["Relevance"].fillna(0).astype(int).value_counts().sort_index()
        rel_plot.index = [f"{int(x)}" for x in rel_plot.index]
        save_bar_chart(rel_plot, "Relevance distribution", "Relevance code", "Count", "relevance_distribution.png")

    # Outlet
    if not outlet_profile.empty:
        outlet_series = outlet_profile.set_index("Outlet_Label")["n_total"]
        save_bar_chart(outlet_series, "Outlet distribution", "Outlet", "Count", "outlet_distribution.png", rotate=True)

    # Actor mention
    if "Actor_Mention" in review_master.columns:
        am = review_master["Actor_Mention"].fillna(0).astype(int).value_counts().sort_index()
        am.index = [ACTOR_MENTION_LABELS.get(int(x), str(x)) for x in am.index]
        save_bar_chart(am, "Actor mention distribution", "Actor mention", "Count", "actor_mention_distribution.png", rotate=True)

    # Dominant label
    if "Dominant_Label" in review_master.columns:
        dl = review_master["Dominant_Label"].fillna(0).astype(int).value_counts().sort_index()
        dl.index = [DOMINANT_LABEL_LABELS.get(int(x), str(x)) for x in dl.index]
        save_bar_chart(dl, "Dominant label distribution", "Dominant label", "Count", "dominant_label_distribution.png", rotate=True)

    # Dominant location
    if "Dominant_Location" in review_master.columns:
        dloc = review_master["Dominant_Location"].fillna(0).astype(int).value_counts().sort_index()
        dloc.index = [DOMINANT_LOCATION_LABELS.get(int(x), str(x)) for x in dloc.index]
        save_bar_chart(dloc, "Dominant location distribution", "Dominant location", "Count", "dominant_location_distribution.png", rotate=True)

    # Main associated actor
    if "Main_Associated_Actor" in review_master.columns:
        maa = review_master["Main_Associated_Actor"].fillna(0).astype(int).value_counts().sort_index()
        maa.index = [MAIN_ASSOCIATED_ACTOR_LABELS.get(int(x), str(x)) for x in maa.index]
        save_bar_chart(maa, "Main associated actor distribution", "Main associated actor", "Count", "main_associated_actor_distribution.png", rotate=True)

    # Frame prevalence
    if not frame_profile.empty:
        frame_series = frame_profile.set_index("frame")["count_total"]
        save_bar_chart(frame_series, "Frame prevalence", "Frame", "Count", "frame_prevalence.png", rotate=True)

    # Discourse
    if "Dominant_Discourse_Support" in review_master.columns:
        disc = review_master["Dominant_Discourse_Support"].fillna(0).astype(int).value_counts().sort_index()
        disc.index = [DISCOURSE_SUPPORT_LABELS.get(int(x), str(x)) for x in disc.index]
        save_bar_chart(disc, "Dominant discourse support distribution", "Discourse", "Count", "discourse_support_distribution.png", rotate=True)

    # Review flags
    if "Review_Flag_Count" in review_master.columns:
        rfc_series = review_master["Review_Flag_Count"].fillna(0).astype(int).value_counts().sort_index()
        rfc_series.index = [str(int(x)) for x in rfc_series.index]
        save_bar_chart(rfc_series, "Review flag count distribution", "Review flag count", "Count", "review_flag_distribution.png")

    # Source profile
    if not source_profile.empty:
        src_series = source_profile.set_index("metric")["count"]
        save_bar_chart(src_series, "Source / republication profile", "Metric", "Count", "source_republication_profile.png", rotate=True)

    # =========================
    # SCHEMATA (MERMAID)
    # =========================
    pipeline_overview = f"""
flowchart TD
    A[Retrieved corpus: {total_rows}] --> B[Relevance 1: {relevance1}]
    A --> C[Relevance 2: {relevance2}]
    A --> D[Relevance 3: {relevance3}]
    A --> E[Relevance 4: {relevance4}]
    C --> F[Relevance 2+ adjudication pool: {relevance2 + relevance3 + relevance4}]
    D --> G[Relevance 3+ analytical subset: {relevance3 + relevance4}]
    E --> G
    A --> H[Review master]
    H --> I[Tables]
    H --> J[Figures]
    H --> K[Chapter 5.2 review workspace]
""".strip()

    adjudication_layers = f"""
flowchart TD
    A[Pipeline consolidated corpus] --> B[Gemini payload batch]
    B --> C[Gemini parsed outputs]
    C --> D[LLM authoritative table]
    C --> E[Conservative adjudicated table]
    C --> F[Manual verification table]
    C --> G[High-confidence subset]
""".strip()

    write_text(CORPUS_SCHEMATA_DIR / "pipeline_overview.mmd", pipeline_overview)
    write_text(CORPUS_SCHEMATA_DIR / "adjudication_layers.mmd", adjudication_layers)

    # =========================
    # SUMMARY TEXT
    # =========================
    top_outlet = outlet_profile.iloc[0]["Outlet_Label"] if not outlet_profile.empty else "N/A"

    def top_label_for_var(df_dist, variable_name):
        sub = df_dist[df_dist["variable"] == variable_name]
        if sub.empty:
            return "N/A"
        sub = sub.sort_values("count", ascending=False)
        return safe_str(sub.iloc[0]["label"])

    summary_text = f"""
CORPUS1 SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

1. Corpus size
- Retrieved corpus: {total_rows}
- Relevance 1: {relevance1}
- Relevance 2: {relevance2}
- Relevance 3: {relevance3}
- Relevance 4: {relevance4}
- Relevance 2+: {relevance2 + relevance3 + relevance4}
- Relevance 3+: {relevance3 + relevance4}

2. Outlet profile
- Number of outlets: {review_master["Outlet"].nunique() if "Outlet" in review_master.columns else 0}
- Largest outlet in corpus: {top_outlet}

3. Representation profile
- Most common Actor_Mention: {top_label_for_var(representation_profile, "Actor_Mention")}
- Most common Dominant_Label: {top_label_for_var(representation_profile, "Dominant_Label")}
- Most common Dominant_Location: {top_label_for_var(representation_profile, "Dominant_Location")}
- Most common Main_Associated_Actor: {top_label_for_var(representation_profile, "Main_Associated_Actor")}
- Most common Stance_Support: {top_label_for_var(representation_profile, "Stance_Support")}
- Most common Legitimation_Support: {top_label_for_var(representation_profile, "Legitimation_Support")}
- Most common Dominant_Discourse_Support: {top_label_for_var(representation_profile, "Dominant_Discourse_Support")}

4. Review profile
- Any_Review_Flag = 1: {(review_master["Any_Review_Flag"].fillna(0).astype(int) == 1).sum() if "Any_Review_Flag" in review_master.columns else 0}
- Chapter 5.2 priority cases: {review_master["chapter52_priority_flag"].sum()}

5. StepB source / republication profile
- StepB file present: {"yes" if post_stepb is not None else "no"}
- Source-attributed articles: {(review_master["source_attributed_flag"].fillna(0).astype(int) == 1).sum() if "source_attributed_flag" in review_master.columns else 0}
- Likely republished articles: {(review_master["likely_republished_flag"].fillna(0).astype(int) == 1).sum() if "likely_republished_flag" in review_master.columns else 0}
- Near-duplicate articles: {(review_master["near_duplicate_flag"].fillna(0).astype(int) == 1).sum() if "near_duplicate_flag" in review_master.columns else 0}

6. Gemini adjudication profile
- Gemini working file present: {"yes" if gemini_working is not None else "no"}
- Gemini conservative file present: {"yes" if gemini_conservative is not None else "no"}
- Gemini authoritative file present: {"yes" if gemini_authoritative is not None else "no"}
- Gemini high-confidence file present: {"yes" if gemini_high_conf is not None else "no"}
- Gemini valid parsed outputs: {len(gemini_parsed) if gemini_parsed is not None else 0}
- Gemini parse errors: {len(gemini_parse_errors) if gemini_parse_errors is not None else 0}
- Gemini merge issues: {len(gemini_merge_issues) if gemini_merge_issues is not None else 0}

7. Outputs generated
- review_master.csv
- corpus_overview.csv
- relevance_profile.csv
- outlet_profile.csv
- representation_profile.csv
- frame_profile.csv
- review_profile.csv
- source_republication_profile.csv
- adjudication_profile.csv
- tables/*.csv
- figures/*.png
- schemata/*.mmd
""".strip()

    write_text(CORPUS_SUMMARY_DIR / "corpus1_summary.txt", summary_text)
    write_text(CORPUS_SUMMARY_DIR / "corpus1_summary.md", summary_text)

    # =========================
    # CONSOLE DIAGNOSTICS
    # =========================
    print("\n=== CORPUS1 DIAGNOSTICS ===\n")
    print(f"Review master rows: {len(review_master)}")
    print(f"Relevance 1: {relevance1}")
    print(f"Relevance 2: {relevance2}")
    print(f"Relevance 3: {relevance3}")
    print(f"Relevance 4: {relevance4}")
    print(f"Relevance 2+: {relevance2 + relevance3 + relevance4}")
    print(f"Relevance 3+: {relevance3 + relevance4}")
    print(f"Chapter 5.2 priority cases: {review_master['chapter52_priority_flag'].sum()}")
    print(f"StepB present: {'yes' if post_stepb is not None else 'no'}")
    print(f"Gemini working present: {'yes' if gemini_working is not None else 'no'}")
    print(f"Gemini conservative present: {'yes' if gemini_conservative is not None else 'no'}")
    print(f"Gemini authoritative present: {'yes' if gemini_authoritative is not None else 'no'}")
    print(f"Gemini high-confidence present: {'yes' if gemini_high_conf is not None else 'no'}")
    print("\nSaved outputs to:")
    print(f"- {CORPUS_DATA_DIR}")
    print(f"- {CORPUS_TABLES_DIR}")
    print(f"- {CORPUS_FIGURES_DIR}")
    print(f"- {CORPUS_SCHEMATA_DIR}")
    print(f"- {CORPUS_SUMMARY_DIR}")


if __name__ == "__main__":
    main()