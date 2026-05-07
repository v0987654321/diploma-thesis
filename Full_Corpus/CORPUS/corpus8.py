import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

# Core inputs
INPUT_REVIEW_MASTER = SCRIPT_DIR / "data" / "review_master.csv"
INPUT_CORPUS_OVERVIEW = SCRIPT_DIR / "data" / "corpus_overview.csv"
INPUT_REPRESENTATION_PROFILE = SCRIPT_DIR / "data" / "representation_profile.csv"
INPUT_FRAME_PROFILE = SCRIPT_DIR / "data" / "frame_profile.csv"
INPUT_REVIEW_PROFILE = SCRIPT_DIR / "data" / "review_profile.csv"
INPUT_SOURCE_REPUBLICATION_PROFILE = SCRIPT_DIR / "data" / "source_republication_profile.csv"
INPUT_ADJUDICATION_PROFILE = SCRIPT_DIR / "data" / "adjudication_profile.csv"

# corpus2 evidence packs
INPUT_EVIDENCE_AC = SCRIPT_DIR / "evidence" / "evidence_africa_corps.csv"
INPUT_EVIDENCE_SUCCESSOR = SCRIPT_DIR / "evidence" / "evidence_successor_cases.csv"
INPUT_EVIDENCE_HRA = SCRIPT_DIR / "evidence" / "evidence_human_rights_abuse.csv"
INPUT_EVIDENCE_SEC = SCRIPT_DIR / "evidence" / "evidence_security_effectiveness.csv"
INPUT_EVIDENCE_GEO = SCRIPT_DIR / "evidence" / "evidence_geopolitical_rivalry.csv"
INPUT_EVIDENCE_SOURCE = SCRIPT_DIR / "evidence" / "evidence_source_attributed.csv"
INPUT_EVIDENCE_REP = SCRIPT_DIR / "evidence" / "evidence_likely_republished.csv"
INPUT_EVIDENCE_REVIEW = SCRIPT_DIR / "evidence" / "evidence_review_flagged.csv"
INPUT_EVIDENCE_CH52 = SCRIPT_DIR / "evidence" / "evidence_chapter52_priority.csv"

# corpus3
INPUT_KWIC_KEYWORD_SUMMARY = SCRIPT_DIR / "concordance" / "kwic_keyword_summary.csv"
INPUT_KWIC_GROUP_SUMMARY = SCRIPT_DIR / "concordance" / "kwic_group_summary.csv"
INPUT_KEYWORD_OUTLET_PROFILE = SCRIPT_DIR / "concordance" / "keyword_outlet_profile.csv"
INPUT_KEYWORD_SUBSET_PROFILE = SCRIPT_DIR / "concordance" / "keyword_subset_profile.csv"

# corpus4
INPUT_KEYWORD_GROUP_PROFILE = SCRIPT_DIR / "lexical" / "keyword_group_profile.csv"
INPUT_TOP_KEYWORDS_RELEVANCE4 = SCRIPT_DIR / "lexical" / "top_keywords_relevance_4.csv"
INPUT_TOP_KEYWORDS_AC = SCRIPT_DIR / "lexical" / "top_keywords_africa_corps_cases.csv"
INPUT_TOP_KEYWORDS_SUCCESSOR = SCRIPT_DIR / "lexical" / "top_keywords_successor_cases.csv"
INPUT_TOP_KEYWORDS_SOURCE = SCRIPT_DIR / "lexical" / "top_keywords_source_attributed.csv"
INPUT_REPEATED_SENTENCE_PATTERNS = SCRIPT_DIR / "lexical" / "repeated_sentence_patterns.csv"

# corpus5
INPUT_MODE_COMPARISON = SCRIPT_DIR / "lexical_norm" / "mode_comparison_summary.csv"
INPUT_ANALYTIC_GROUP_PROFILE = SCRIPT_DIR / "lexical_norm" / "analytic_group_profile.csv"
INPUT_TOP_TOKENS_ANALYTIC_REL4 = SCRIPT_DIR / "lexical_norm" / "top_tokens_analytic_relevance_4.csv"
INPUT_TOP_TOKENS_ANALYTIC_AC = SCRIPT_DIR / "lexical_norm" / "top_tokens_analytic_africa_corps_cases.csv"
INPUT_TOP_TOKENS_ANALYTIC_SUCCESSOR = SCRIPT_DIR / "lexical_norm" / "top_tokens_analytic_successor_cases.csv"
INPUT_TOP_TOKENS_ANALYTIC_CH52 = SCRIPT_DIR / "lexical_norm" / "top_tokens_analytic_chapter52_priority.csv"

# corpus6
INPUT_TOPIC_REL4_INFO = SCRIPT_DIR / "topics" / "relevance4" / "bertopic_topic_info.csv"
INPUT_TOPIC_AC_INFO = SCRIPT_DIR / "topics" / "africa_corps_or_both" / "bertopic_topic_info.csv"
INPUT_TOPIC_REL4_SUMMARY = SCRIPT_DIR / "topics" / "relevance4" / "topic_model_summary.txt"
INPUT_TOPIC_AC_SUMMARY = SCRIPT_DIR / "topics" / "africa_corps_or_both" / "topic_model_summary.txt"

# corpus7
INPUT_TOP_ENTITIES_OVERALL = SCRIPT_DIR / "ner" / "top_entities_overall.csv"
INPUT_TOP_ENTITIES_REL4 = SCRIPT_DIR / "ner" / "top_entities_relevance_4.csv"
INPUT_TOP_ENTITIES_AC = SCRIPT_DIR / "ner" / "top_entities_africa_corps_cases.csv"
INPUT_TOP_ENTITIES_SUCCESSOR = SCRIPT_DIR / "ner" / "top_entities_successor_cases.csv"
INPUT_TOP_ENTITIES_SOURCE = SCRIPT_DIR / "ner" / "top_entities_source_attributed.csv"
INPUT_TOP_ENTITIES_BY_OUTLET = SCRIPT_DIR / "ner" / "top_entities_by_outlet.csv"
INPUT_TARGET_ENTITY_PROFILE = SCRIPT_DIR / "ner" / "target_entity_profile.csv"
INPUT_TARGET_ENTITY_SENT_PROFILE = SCRIPT_DIR / "ner" / "target_entity_profile_sentence_level.csv"
INPUT_SUBSET_ENTITY_SUMMARY = SCRIPT_DIR / "ner" / "subset_entity_summary.csv"

# Outputs
SYNTHESIS_DIR = SCRIPT_DIR / "synthesis"
SUMMARY_DIR = SCRIPT_DIR / "summary"

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
    SYNTHESIS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def load_csv_optional(path, dtype=None):
    if path.exists():
        return pd.read_csv(path, dtype=dtype, low_memory=False)
    return None

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def top_value_label(df, col, mapping=None):
    if col not in df.columns or df.empty:
        return "N/A"
    vc = df[col].fillna(0).astype(int).value_counts()
    if vc.empty:
        return "N/A"
    code = int(vc.index[0])
    if mapping:
        return mapping.get(code, str(code))
    return str(code)

def percent(n, d):
    return (n / d * 100) if d else 0.0

def top_token_from_file(path):
    df = load_csv_optional(path)
    if df is None or df.empty or "token" not in df.columns:
        return "N/A"
    return safe_str(df.iloc[0]["token"])

def top_entity_from_file(path):
    df = load_csv_optional(path)
    if df is None or df.empty or "entity_norm" not in df.columns:
        return "N/A"
    return safe_str(df.iloc[0]["entity_norm"])

def top_topic_name(path):
    df = load_csv_optional(path)
    if df is None or df.empty:
        return "N/A"
    if "Topic" in df.columns:
        df = df[df["Topic"] != -1].copy()
    if df.empty:
        return "N/A"
    df = df.sort_values("Count", ascending=False) if "Count" in df.columns else df
    if "Name" in df.columns:
        return safe_str(df.iloc[0]["Name"])
    if "Topic" in df.columns:
        return f"Topic {safe_int(df.iloc[0]['Topic'])}"
    return "N/A"

def subset_df(df, mask):
    return df[mask].copy()

def build_binary_rate(df, col):
    if col not in df.columns or df.empty:
        return 0, 0.0
    count_1 = (df[col].fillna(0).astype(int) == 1).sum()
    return count_1, percent(count_1, len(df))

def load_text_optional(path):
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

# Labels
ACTOR_MENTION_LABELS = {
    1: "Wagner Group",
    2: "Africa Corps",
    3: "Both explicitly",
    4: "Indirect Russian contractors/forces",
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
# MAIN
# =========================
def main():
    ensure_dirs()

    review_master = load_csv_optional(INPUT_REVIEW_MASTER, dtype={"Article_ID": str})
    if review_master is None:
        raise FileNotFoundError(f"Missing required input: {INPUT_REVIEW_MASTER}")

    review_master["Relevance"] = review_master["Relevance"].fillna(0).astype(int)
    if "Actor_Mention" in review_master.columns:
        review_master["Actor_Mention"] = review_master["Actor_Mention"].fillna(0).astype(int)
    if "Successor_Frame" in review_master.columns:
        review_master["Successor_Frame"] = review_master["Successor_Frame"].fillna(0).astype(int)

    # optional inputs
    corpus_overview = load_csv_optional(INPUT_CORPUS_OVERVIEW)
    representation_profile = load_csv_optional(INPUT_REPRESENTATION_PROFILE)
    frame_profile = load_csv_optional(INPUT_FRAME_PROFILE)
    review_profile = load_csv_optional(INPUT_REVIEW_PROFILE)
    source_rep_profile = load_csv_optional(INPUT_SOURCE_REPUBLICATION_PROFILE)
    adjudication_profile = load_csv_optional(INPUT_ADJUDICATION_PROFILE)

    kwic_keyword_summary = load_csv_optional(INPUT_KWIC_KEYWORD_SUMMARY)
    kwic_group_summary = load_csv_optional(INPUT_KWIC_GROUP_SUMMARY)
    keyword_outlet_profile = load_csv_optional(INPUT_KEYWORD_OUTLET_PROFILE)
    keyword_subset_profile = load_csv_optional(INPUT_KEYWORD_SUBSET_PROFILE)

    keyword_group_profile = load_csv_optional(INPUT_KEYWORD_GROUP_PROFILE)
    repeated_sentence_patterns = load_csv_optional(INPUT_REPEATED_SENTENCE_PATTERNS)

    mode_comparison = load_csv_optional(INPUT_MODE_COMPARISON)
    analytic_group_profile = load_csv_optional(INPUT_ANALYTIC_GROUP_PROFILE)

    top_entities_overall = load_csv_optional(INPUT_TOP_ENTITIES_OVERALL)
    top_entities_rel4 = load_csv_optional(INPUT_TOP_ENTITIES_REL4)
    top_entities_ac = load_csv_optional(INPUT_TOP_ENTITIES_AC)
    top_entities_successor = load_csv_optional(INPUT_TOP_ENTITIES_SUCCESSOR)
    top_entities_source = load_csv_optional(INPUT_TOP_ENTITIES_SOURCE)
    top_entities_by_outlet = load_csv_optional(INPUT_TOP_ENTITIES_BY_OUTLET)
    target_entity_profile = load_csv_optional(INPUT_TARGET_ENTITY_PROFILE)
    target_entity_sent_profile = load_csv_optional(INPUT_TARGET_ENTITY_SENT_PROFILE)
    subset_entity_summary = load_csv_optional(INPUT_SUBSET_ENTITY_SUMMARY)

    # topic text summaries
    topic_rel4_summary_txt = load_text_optional(INPUT_TOPIC_REL4_SUMMARY)
    topic_ac_summary_txt = load_text_optional(INPUT_TOPIC_AC_SUMMARY)

    # =========================
    # MAIN SUBSETS
    # =========================
    df_all = review_master.copy()
    df_rel4 = subset_df(review_master, review_master["Relevance"] == 4)
    df_rel3plus = subset_df(review_master, review_master["Relevance"].isin([3, 4]))
    df_wagner_only = subset_df(review_master, review_master["Actor_Mention"] == 1) if "Actor_Mention" in review_master.columns else pd.DataFrame()
    df_ac_or_both = subset_df(review_master, review_master["Actor_Mention"].isin([2, 3])) if "Actor_Mention" in review_master.columns else pd.DataFrame()
    df_source_attr = subset_df(review_master, review_master["source_attributed_flag"].fillna(0).astype(int) == 1) if "source_attributed_flag" in review_master.columns else pd.DataFrame()
    df_non_source_attr = subset_df(review_master, review_master["source_attributed_flag"].fillna(0).astype(int) == 0) if "source_attributed_flag" in review_master.columns else pd.DataFrame()
    df_high_conf = subset_df(review_master, review_master["has_gemini_high_confidence"].fillna(0).astype(int) == 1) if "has_gemini_high_confidence" in review_master.columns else pd.DataFrame()
    df_review_heavy = subset_df(review_master, review_master["Any_Review_Flag"].fillna(0).astype(int) == 1) if "Any_Review_Flag" in review_master.columns else pd.DataFrame()

    # =========================
    # A. CORPUS-WIDE SYNTHESIS
    # =========================
    corpus_wide_rows = [{
        "metric": "retrieved_articles",
        "value": len(df_all)
    },{
        "metric": "relevance_1",
        "value": int((df_all["Relevance"] == 1).sum())
    },{
        "metric": "relevance_2",
        "value": int((df_all["Relevance"] == 2).sum())
    },{
        "metric": "relevance_3",
        "value": int((df_all["Relevance"] == 3).sum())
    },{
        "metric": "relevance_4",
        "value": int((df_all["Relevance"] == 4).sum())
    },{
        "metric": "relevance_3plus",
        "value": len(df_rel3plus)
    },{
        "metric": "wagner_only_cases",
        "value": len(df_wagner_only)
    },{
        "metric": "africa_corps_or_both_cases",
        "value": len(df_ac_or_both)
    },{
        "metric": "source_attributed_cases",
        "value": len(df_source_attr)
    },{
        "metric": "review_flagged_cases",
        "value": len(df_review_heavy)
    }]

    corpus_wide_synthesis = pd.DataFrame(corpus_wide_rows)
    save_csv(corpus_wide_synthesis, SYNTHESIS_DIR / "corpus_wide_synthesis.csv")

    # =========================
    # B. ACTOR REPRESENTATION SYNTHESIS
    # =========================
    actor_rep_rows = []
    for label, df_sub in [
        ("all", df_all),
        ("relevance_4", df_rel4),
        ("relevance_3plus", df_rel3plus),
        ("wagner_only", df_wagner_only),
        ("ac_or_both", df_ac_or_both),
    ]:
        if df_sub.empty:
            continue

        actor_rep_rows.append({
            "subset": label,
            "n_articles": len(df_sub),
            "top_actor_mention": top_value_label(df_sub, "Actor_Mention", ACTOR_MENTION_LABELS),
            "top_dominant_label": top_value_label(df_sub, "Dominant_Label", DOMINANT_LABEL_LABELS),
            "top_dominant_location": top_value_label(df_sub, "Dominant_Location", DOMINANT_LOCATION_LABELS),
            "top_main_associated_actor": top_value_label(df_sub, "Main_Associated_Actor", MAIN_ASSOCIATED_ACTOR_LABELS),
            "top_dominant_discourse": top_value_label(df_sub, "Dominant_Discourse_Support", DISCOURSE_SUPPORT_LABELS),
        })

    actor_representation_synthesis = pd.DataFrame(actor_rep_rows)
    save_csv(actor_representation_synthesis, SYNTHESIS_DIR / "actor_representation_synthesis.csv")

    # =========================
    # C. FRAME / DISCOURSE SYNTHESIS
    # =========================
    frame_discourse_rows = []
    for label, df_sub in [
        ("all", df_all),
        ("relevance_4", df_rel4),
        ("relevance_3plus", df_rel3plus),
        ("ac_or_both", df_ac_or_both),
        ("source_attributed", df_source_attr),
    ]:
        if df_sub.empty:
            continue

        row = {"subset": label, "n_articles": len(df_sub)}
        for frame in FRAME_COLS:
            if frame in df_sub.columns:
                count_1 = (df_sub[frame].fillna(0).astype(int) == 1).sum()
                row[f"{frame}_count"] = int(count_1)
                row[f"{frame}_percent"] = percent(count_1, len(df_sub))
        row["top_dominant_discourse"] = top_value_label(df_sub, "Dominant_Discourse_Support", DISCOURSE_SUPPORT_LABELS)
        frame_discourse_rows.append(row)

    frame_discourse_synthesis = pd.DataFrame(frame_discourse_rows)
    save_csv(frame_discourse_synthesis, SYNTHESIS_DIR / "frame_discourse_synthesis.csv")

    # =========================
    # D. OUTLET DIFFERENCE SYNTHESIS
    # =========================
    outlet_rows = []
    if "Outlet" in review_master.columns:
        for outlet, grp in review_master.groupby("Outlet", dropna=False):
            outlet_rows.append({
                "Outlet": outlet,
                "n_articles": len(grp),
                "n_relevance_3plus": int(grp["Relevance"].isin([3, 4]).sum()),
                "top_label": top_value_label(grp, "Dominant_Label", DOMINANT_LABEL_LABELS),
                "top_location": top_value_label(grp, "Dominant_Location", DOMINANT_LOCATION_LABELS),
                "top_associated_actor": top_value_label(grp, "Main_Associated_Actor", MAIN_ASSOCIATED_ACTOR_LABELS),
                "top_discourse": top_value_label(grp, "Dominant_Discourse_Support", DISCOURSE_SUPPORT_LABELS),
                "source_attributed_percent": percent((grp["source_attributed_flag"].fillna(0).astype(int) == 1).sum(), len(grp)) if "source_attributed_flag" in grp.columns else 0,
            })
    outlet_difference_synthesis = pd.DataFrame(outlet_rows)
    save_csv(outlet_difference_synthesis, SYNTHESIS_DIR / "outlet_difference_synthesis.csv")

    # =========================
    # E. SOURCE / REPUBLICATION SYNTHESIS
    # =========================
    source_rows = []
    if "source_attributed_flag" in review_master.columns:
        for label, df_sub in [("source_attributed", df_source_attr), ("non_source_attributed", df_non_source_attr)]:
            if df_sub.empty:
                continue
            source_rows.append({
                "subset": label,
                "n_articles": len(df_sub),
                "top_label": top_value_label(df_sub, "Dominant_Label", DOMINANT_LABEL_LABELS),
                "top_discourse": top_value_label(df_sub, "Dominant_Discourse_Support", DISCOURSE_SUPPORT_LABELS),
                "top_associated_actor": top_value_label(df_sub, "Main_Associated_Actor", MAIN_ASSOCIATED_ACTOR_LABELS),
                "hra_percent": percent((df_sub["Human_Rights_Abuse"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Human_Rights_Abuse" in df_sub.columns else 0,
                "security_effectiveness_percent": percent((df_sub["Security_Effectiveness"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Security_Effectiveness" in df_sub.columns else 0,
                "geopolitics_percent": percent((df_sub["Geopolitical_Rivalry"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Geopolitical_Rivalry" in df_sub.columns else 0,
                "ac_or_both_percent": percent((df_sub["Actor_Mention"].fillna(0).astype(int).isin([2, 3])).sum(), len(df_sub)) if "Actor_Mention" in df_sub.columns else 0,
            })
    source_republication_synthesis = pd.DataFrame(source_rows)
    save_csv(source_republication_synthesis, SYNTHESIS_DIR / "source_republication_synthesis.csv")

    # =========================
    # F. WAGNER VS AC INTEGRATED COMPARISON
    # IMPORTANT FIX:
    # - do not misuse relevance_4 lexical/entity/topic signals as Wagner-only signals
    # - use available AC-specific layers for AC
    # - use WAGNER-only structured metrics + generic overall lexical/entity context where explicit Wagner-only files do not exist
    # =========================
    wagner_vs_ac_rows = []

    # What is actually available:
    # - AC-specific lexical/entity/topic files exist
    # - Wagner-only dedicated lexical/entity/topic files do NOT exist in current pipeline
    # Therefore:
    #   Wagner-only lexical/entity/topic signals must be explicitly labelled as proxy/overall where used

    ac_top_token = top_token_from_file(INPUT_TOP_TOKENS_ANALYTIC_AC)
    ac_top_entity = top_entity_from_file(INPUT_TOP_ENTITIES_AC)
    ac_top_topic = top_topic_name(INPUT_TOPIC_AC_INFO)

    wagner_proxy_top_token = top_token_from_file(INPUT_TOP_TOKENS_ANALYTIC_REL4)
    wagner_proxy_top_entity = top_entity_from_file(INPUT_TOP_ENTITIES_OVERALL)
    wagner_proxy_top_topic = top_topic_name(INPUT_TOPIC_REL4_INFO)

    for label, df_sub in [("wagner_only", df_wagner_only), ("ac_or_both", df_ac_or_both)]:
        if df_sub.empty:
            continue

        caution_small_n = 1 if len(df_sub) < 50 else 0
        caution_high_review = 1 if "Any_Review_Flag" in df_sub.columns and percent((df_sub["Any_Review_Flag"].fillna(0).astype(int) == 1).sum(), len(df_sub)) > 50 else 0

        if label == "wagner_only":
            lexical_signal = wagner_proxy_top_token
            entity_signal = wagner_proxy_top_entity
            topic_signal = wagner_proxy_top_topic
            topic_note = "proxy_from_relevance4"
            lexical_note = "proxy_from_relevance4_analytic"
            entity_note = "proxy_from_overall_entities"
            topic_pointer = "topics/relevance4/"
            ner_pointer = "top_entities_overall.csv"
        else:
            lexical_signal = ac_top_token
            entity_signal = ac_top_entity
            topic_signal = ac_top_topic
            topic_note = "direct_ac_subset"
            lexical_note = "direct_ac_subset"
            entity_note = "direct_ac_subset"
            topic_pointer = "topics/africa_corps_or_both/"
            ner_pointer = "top_entities_africa_corps_cases.csv"

        wagner_vs_ac_rows.append({
            "comparison_group": label,
            "n_articles": len(df_sub),
            "relevance_4_percent": percent((df_sub["Relevance"] == 4).sum(), len(df_sub)),
            "successor_frame_percent": percent((df_sub["Successor_Frame"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Successor_Frame" in df_sub.columns else 0,
            "top_label": top_value_label(df_sub, "Dominant_Label", DOMINANT_LABEL_LABELS),
            "top_location": top_value_label(df_sub, "Dominant_Location", DOMINANT_LOCATION_LABELS),
            "top_associated_actor": top_value_label(df_sub, "Main_Associated_Actor", MAIN_ASSOCIATED_ACTOR_LABELS),
            "top_discourse": top_value_label(df_sub, "Dominant_Discourse_Support", DISCOURSE_SUPPORT_LABELS),
            "hra_percent": percent((df_sub["Human_Rights_Abuse"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Human_Rights_Abuse" in df_sub.columns else 0,
            "security_effectiveness_percent": percent((df_sub["Security_Effectiveness"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Security_Effectiveness" in df_sub.columns else 0,
            "geopolitics_percent": percent((df_sub["Geopolitical_Rivalry"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Geopolitical_Rivalry" in df_sub.columns else 0,
            "source_attributed_percent": percent((df_sub["source_attributed_flag"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "source_attributed_flag" in df_sub.columns else 0,
            "likely_republished_percent": percent((df_sub["likely_republished_flag"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "likely_republished_flag" in df_sub.columns else 0,
            "top_lexical_signal": lexical_signal,
            "top_lexical_signal_note": lexical_note,
            "top_entity_signal": entity_signal,
            "top_entity_signal_note": entity_note,
            "top_topic_signal": topic_signal,
            "top_topic_signal_note": topic_note,
            "small_n_flag": caution_small_n,
            "high_review_flag": caution_high_review,
            "evidence_pointer": "evidence_africa_corps.csv" if label == "ac_or_both" else "subset_wagner_only_cases.csv",
            "kwic_pointer": "kwic_group_core_actor_terms.csv",
            "topic_pointer": topic_pointer,
            "ner_pointer": ner_pointer,
        })

    wagner_vs_ac_integrated = pd.DataFrame(wagner_vs_ac_rows)
    save_csv(wagner_vs_ac_integrated, SYNTHESIS_DIR / "wagner_vs_ac_integrated_comparison.csv")

    # =========================
    # G. SOURCE ATTRIBUTED VS NON-ATTRIBUTED
    # =========================
    source_compare_rows = []
    for label, df_sub in [("source_attributed", df_source_attr), ("non_source_attributed", df_non_source_attr)]:
        if df_sub.empty:
            continue

        source_compare_rows.append({
            "comparison_group": label,
            "n_articles": len(df_sub),
            "relevance_4_percent": percent((df_sub["Relevance"] == 4).sum(), len(df_sub)),
            "top_label": top_value_label(df_sub, "Dominant_Label", DOMINANT_LABEL_LABELS),
            "top_discourse": top_value_label(df_sub, "Dominant_Discourse_Support", DISCOURSE_SUPPORT_LABELS),
            "top_associated_actor": top_value_label(df_sub, "Main_Associated_Actor", MAIN_ASSOCIATED_ACTOR_LABELS),
            "hra_percent": percent((df_sub["Human_Rights_Abuse"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Human_Rights_Abuse" in df_sub.columns else 0,
            "security_effectiveness_percent": percent((df_sub["Security_Effectiveness"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Security_Effectiveness" in df_sub.columns else 0,
            "geopolitics_percent": percent((df_sub["Geopolitical_Rivalry"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Geopolitical_Rivalry" in df_sub.columns else 0,
            "ac_or_both_percent": percent((df_sub["Actor_Mention"].fillna(0).astype(int).isin([2, 3])).sum(), len(df_sub)) if "Actor_Mention" in df_sub.columns else 0,
            "top_lexical_signal": top_token_from_file(INPUT_TOP_TOKENS_ANALYTIC_CH52) if label == "source_attributed" else top_token_from_file(INPUT_TOP_TOKENS_ANALYTIC_REL4),
            "top_entity_signal": top_entity_from_file(INPUT_TOP_ENTITIES_SOURCE) if label == "source_attributed" else top_entity_from_file(INPUT_TOP_ENTITIES_OVERALL),
            "small_n_flag": 1 if len(df_sub) < 50 else 0,
            "high_review_flag": 1 if "Any_Review_Flag" in df_sub.columns and percent((df_sub["Any_Review_Flag"].fillna(0).astype(int) == 1).sum(), len(df_sub)) > 50 else 0,
            "evidence_pointer": "evidence_source_attributed.csv" if label == "source_attributed" else "review_master.csv",
            "kwic_pointer": "kwic_group_source_attribution_terms.csv",
            "ner_pointer": "top_entities_source_attributed.csv" if label == "source_attributed" else "top_entities_overall.csv",
        })

    source_vs_nonattributed_integrated = pd.DataFrame(source_compare_rows)
    save_csv(source_vs_nonattributed_integrated, SYNTHESIS_DIR / "source_attributed_vs_nonattributed_integrated_comparison.csv")

    # =========================
    # H. REVIEW VS HIGH-CONFIDENCE
    # =========================
    review_vs_hc_rows = []
    for label, df_sub in [("review_flagged", df_review_heavy), ("high_confidence", df_high_conf)]:
        if df_sub.empty:
            continue

        review_vs_hc_rows.append({
            "comparison_group": label,
            "n_articles": len(df_sub),
            "top_label": top_value_label(df_sub, "Dominant_Label", DOMINANT_LABEL_LABELS),
            "top_discourse": top_value_label(df_sub, "Dominant_Discourse_Support", DISCOURSE_SUPPORT_LABELS),
            "top_associated_actor": top_value_label(df_sub, "Main_Associated_Actor", MAIN_ASSOCIATED_ACTOR_LABELS),
            "ac_or_both_percent": percent((df_sub["Actor_Mention"].fillna(0).astype(int).isin([2, 3])).sum(), len(df_sub)) if "Actor_Mention" in df_sub.columns else 0,
            "successor_frame_percent": percent((df_sub["Successor_Frame"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "Successor_Frame" in df_sub.columns else 0,
            "source_attributed_percent": percent((df_sub["source_attributed_flag"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "source_attributed_flag" in df_sub.columns else 0,
            "likely_republished_percent": percent((df_sub["likely_republished_flag"].fillna(0).astype(int) == 1).sum(), len(df_sub)) if "likely_republished_flag" in df_sub.columns else 0,
            "top_lexical_signal": top_token_from_file(INPUT_TOP_TOKENS_ANALYTIC_CH52) if label == "review_flagged" else top_token_from_file(INPUT_TOP_TOKENS_ANALYTIC_REL4),
            "top_entity_signal": top_entity_from_file(INPUT_TOP_ENTITIES_OVERALL),
            "small_n_flag": 1 if len(df_sub) < 50 else 0,
            "evidence_pointer": "evidence_review_flagged.csv" if label == "review_flagged" else "final_high_confidence_coding_table.csv",
        })

    review_vs_highconfidence_integrated = pd.DataFrame(review_vs_hc_rows)
    save_csv(review_vs_highconfidence_integrated, SYNTHESIS_DIR / "review_vs_highconfidence_integrated_comparison.csv")

    # =========================
    # I. KEY FINDINGS TABLES
    # =========================
    key_findings_ch52_rows = [
        {
            "finding_id": "KF01",
            "analytical_block": "corpus_composition",
            "candidate_claim": "The analytically substantive corpus is substantially smaller than the retrieved corpus, indicating that many retrieved texts mention Wagner/Africa Corps only marginally.",
            "structured_support_summary": f"Relevance 3+ = {len(df_rel3plus)} of {len(df_all)} retrieved articles.",
            "lexical_support_summary": "See corpus4/5 outputs for relevance_3plus lexical concentration.",
            "entity_support_summary": "See corpus7 subset_entity_summary for relevance_3plus.",
            "topic_support_summary": "See corpus6 relevance4 exploratory topics.",
            "caveat": "Relevance coding remains availability-based and depends on keyword retrieval logic.",
            "evidence_pointer": "corpus_overview.csv; relevance_profile.csv",
        },
        {
            "finding_id": "KF02",
            "analytical_block": "actor_transition",
            "candidate_claim": "Africa Corps coverage appears in a narrower and more transition-sensitive subset than Wagner-only coverage.",
            "structured_support_summary": f"Africa Corps or both cases = {len(df_ac_or_both)}; Wagner-only cases = {len(df_wagner_only)}.",
            "lexical_support_summary": f"Top AC lexical signal = {ac_top_token}; Wagner-only lexical signal is only available via broader proxy outputs.",
            "entity_support_summary": f"Top AC entity signal = {ac_top_entity}; Wagner-only entity signal currently uses overall proxy output.",
            "topic_support_summary": f"AC-specific topic hint = {ac_top_topic}; relevance4 topic output remains a broader proxy for Wagner-only context.",
            "caveat": "Dedicated topic and lexical clustering was only run for relevance4 and AC/both, not for Wagner-only as a standalone topic subset.",
            "evidence_pointer": "wagner_vs_ac_integrated_comparison.csv; evidence_africa_corps.csv",
        },
        {
            "finding_id": "KF03",
            "analytical_block": "framing_discourse",
            "candidate_claim": "Security/stabilization and violence/abuse appear more prominent than sovereignty/emancipation in the codebook-based support layer.",
            "structured_support_summary": "See frame_discourse_synthesis.csv and representation_profile.csv.",
            "lexical_support_summary": "See corpus4 lexical summaries and corpus5 analytic normalization outputs.",
            "entity_support_summary": "Entity environments may help distinguish military/security and abuse-oriented subsets.",
            "topic_support_summary": "Relevance 4 BERTopic output may reinforce conflict/security thematic bundling.",
            "caveat": "Support-layer discourse coding is intentionally conservative.",
            "evidence_pointer": "frame_discourse_synthesis.csv; corpus4_summary.txt; corpus5_summary.txt",
        },
        {
            "finding_id": "KF04",
            "analytical_block": "source_republication",
            "candidate_claim": "Source-attributed and likely republished articles form a meaningful stratum of the corpus and may partially shape observed outlet differences.",
            "structured_support_summary": f"Source-attributed cases = {len(df_source_attr)}.",
            "lexical_support_summary": "See source-attributed keyword subset outputs and source-attribution KWIC group.",
            "entity_support_summary": f"Top source-attributed entity = {top_entity_from_file(INPUT_TOP_ENTITIES_SOURCE)}.",
            "topic_support_summary": "No dedicated BERTopic clustering was run specifically for the source-attributed subset.",
            "caveat": "StepB remains an enrichment layer rather than a primary coding instrument.",
            "evidence_pointer": "source_republication_synthesis.csv; source_republication_profile.csv; evidence_source_attributed.csv",
        },
        {
            "finding_id": "KF05",
            "analytical_block": "media_differences",
            "candidate_claim": "Outlet-level differences should be interpreted not only through coding distributions but also through lexical, entity, and source-attribution environments.",
            "structured_support_summary": "See outlet_difference_synthesis.csv.",
            "lexical_support_summary": "See keyword_outlet_profile.csv and lexical outlet outputs.",
            "entity_support_summary": "See top_entities_by_outlet.csv and outlet_entity_summary.csv.",
            "topic_support_summary": "Topic clustering is exploratory and not outlet-specific in this stage.",
            "caveat": "Outlet asymmetries may partly reflect archive availability and republication patterns.",
            "evidence_pointer": "outlet_difference_synthesis.csv; keyword_outlet_profile.csv; top_entities_by_outlet.csv",
        },
    ]

    key_findings_discussion_rows = [
        {
            "finding_id": "KD01",
            "analytical_block": "triangulation",
            "candidate_claim": "Supplementary lexical, entity, and topic-cluster layers broadly reinforce rather than replace the structured coding architecture.",
            "structured_support_summary": "Primary results remain codebook-based.",
            "lexical_support_summary": "Corpus4 and corpus5 add lexical corroboration.",
            "entity_support_summary": "Corpus7 adds recurrent actor environments.",
            "topic_support_summary": "Corpus6 adds exploratory thematic bundles for selected subsets.",
            "caveat": "These layers are supplementary and exploratory.",
            "evidence_pointer": "corpus4_summary.txt; corpus5_summary.txt; corpus6_summary.txt; corpus7_summary.txt",
        },
        {
            "finding_id": "KD02",
            "analytical_block": "ac_transition",
            "candidate_claim": "Africa Corps coverage may be read as a more regionally dispersed and transition-linked representational field than Wagner-only coverage.",
            "structured_support_summary": "See integrated AC/Wagner comparison.",
            "lexical_support_summary": "AC lexical outputs are direct; Wagner-only lexical hints are more indirect.",
            "entity_support_summary": "AC subset entity patterns include stronger regional external actors.",
            "topic_support_summary": "AC BERTopic cluster output supports exploratory thematic differentiation.",
            "caveat": "AC subset is smaller, and direct topic modelling was not run separately for Wagner-only cases.",
            "evidence_pointer": "wagner_vs_ac_integrated_comparison.csv; topics/africa_corps_or_both/",
        },
        {
            "finding_id": "KD03",
            "analytical_block": "methodological_uncertainty",
            "candidate_claim": "Review-heavy cases likely concentrate in more ambiguous, transition-sensitive, or context-thin parts of the corpus.",
            "structured_support_summary": "See review_vs_highconfidence_integrated_comparison.csv.",
            "lexical_support_summary": "Review-flagged lexical summaries may indicate unstable wording environments.",
            "entity_support_summary": "Review-heavy cases can be compared with broader entity environments.",
            "topic_support_summary": "No dedicated BERTopic comparison for review-heavy subset at this stage.",
            "caveat": "Review flags indicate caution, not necessarily invalid coding.",
            "evidence_pointer": "review_vs_highconfidence_integrated_comparison.csv; evidence_review_flagged.csv",
        },
    ]

    key_findings_ch52 = pd.DataFrame(key_findings_ch52_rows)
    key_findings_discussion = pd.DataFrame(key_findings_discussion_rows)

    save_csv(key_findings_ch52, SYNTHESIS_DIR / "key_findings_ch52.csv")
    save_csv(key_findings_discussion, SYNTHESIS_DIR / "key_findings_discussion.csv")

    # =========================
    # J. WORKING NOTES
    # =========================
    chapter52_notes = f"""
CHAPTER 5.2 WORKING NOTES
Generated: {datetime.now().isoformat(timespec="seconds")}

1. Corpus composition
- Retrieved corpus: {len(df_all)}
- Relevance 3+ core analytical subset: {len(df_rel3plus)}
- Relevance 4 subset: {len(df_rel4)}

2. Actor representation
- Wagner-only cases: {len(df_wagner_only)}
- Africa Corps or both cases: {len(df_ac_or_both)}
- Successor framing should be read alongside the smaller AC subset and transition-sensitive outputs.

3. Framing/discourse
- Security/stabilization and violence/abuse appear more prominent than sovereignty/emancipation.
- The discourse support layer remains conservative and should be interpreted accordingly.

4. Outlet/media differences
- Use outlet_difference_synthesis.csv together with StepB source/republication outputs.
- Compare outlet coding patterns with lexical and entity outputs before making stronger claims.

5. Source/republication
- Source-attributed cases: {len(df_source_attr)}
- These cases likely matter for interpreting outlet variation and should not be ignored.

6. Suggested anchor files
- corpus_wide_synthesis.csv
- actor_representation_synthesis.csv
- frame_discourse_synthesis.csv
- outlet_difference_synthesis.csv
- source_republication_synthesis.csv
- wagner_vs_ac_integrated_comparison.csv
- key_findings_ch52.csv
""".strip()

    discussion_notes = f"""
DISCUSSION WORKING NOTES
Generated: {datetime.now().isoformat(timespec="seconds")}

1. Role of supplementary layers
- corpus4 adds lexical pattern interpretation
- corpus5 adds normalization-sensitive lexical aggregation
- corpus6 adds exploratory BERTopic clustering on selected high-salience subsets
- corpus7 adds NER and entity co-occurrence environments

2. Wagner vs Africa Corps
- Compare structured coding, lexical signals, entity environments, and topic hints.
- Note clearly that AC subset has direct topic/lexical/entity support, whereas Wagner-only topic/lexical support remains more indirect in the current exploratory setup.

3. Source-attributed vs non-attributed
- Use StepB + lexical/entity profiles to discuss whether outlet differences may be partly shaped by republication patterns.

4. Methodological caution
- Treat corpus6 topic modelling as exploratory, supplementary, and discussion-level.
- Treat corpus7 NER as enrichment rather than authoritative actor coding.

5. Bridge to external literature
- The new Mali/France article can be discussed here as a parallel corpus-assisted discourse study.
- Main comparison point: their topic-modelling/sentiment architecture vs this project’s structured codebook + adjudication + CDA design.

6. Suggested anchor files
- source_attributed_vs_nonattributed_integrated_comparison.csv
- review_vs_highconfidence_integrated_comparison.csv
- key_findings_discussion.csv
- corpus4_summary.txt
- corpus5_summary.txt
- corpus6_summary.txt
- corpus7_summary.txt
""".strip()

    corpus8_summary_text = f"""
CORPUS8 SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Role:
- Integrated comparative and writing-support synthesis layer.

Outputs generated:
- corpus_wide_synthesis.csv
- actor_representation_synthesis.csv
- frame_discourse_synthesis.csv
- outlet_difference_synthesis.csv
- source_republication_synthesis.csv
- wagner_vs_ac_integrated_comparison.csv
- source_attributed_vs_nonattributed_integrated_comparison.csv
- review_vs_highconfidence_integrated_comparison.csv
- key_findings_ch52.csv
- key_findings_discussion.csv
- chapter52_working_notes.txt
- discussion_working_notes.txt

Methodological note:
- This layer synthesizes previously generated structured, lexical, entity, and exploratory topic outputs.
- It does not produce new coding, but reorganizes existing analytical outputs into more writer-friendly and discussion-ready formats.
""".strip()

    with open(SUMMARY_DIR / "chapter52_working_notes.txt", "w", encoding="utf-8") as f:
        f.write(chapter52_notes)

    with open(SUMMARY_DIR / "discussion_working_notes.txt", "w", encoding="utf-8") as f:
        f.write(discussion_notes)

    with open(SUMMARY_DIR / "corpus8_summary.txt", "w", encoding="utf-8") as f:
        f.write(corpus8_summary_text)

    with open(SUMMARY_DIR / "corpus8_summary.md", "w", encoding="utf-8") as f:
        f.write(corpus8_summary_text)

    # =========================
    # CONSOLE DIAGNOSTICS
    # =========================
    print("\n=== CORPUS8 DIAGNOSTICS ===\n")
    print(f"Review master rows: {len(df_all)}")
    print(f"Relevance 4 rows: {len(df_rel4)}")
    print(f"Relevance 3+ rows: {len(df_rel3plus)}")
    print(f"Wagner-only rows: {len(df_wagner_only)}")
    print(f"Africa Corps / both rows: {len(df_ac_or_both)}")
    print(f"Source-attributed rows: {len(df_source_attr)}")
    print(f"Review-heavy rows: {len(df_review_heavy)}")
    print(f"High-confidence rows: {len(df_high_conf)}")
    print("\nSaved outputs to:")
    print(f"- {SYNTHESIS_DIR}")
    print(f"- {SUMMARY_DIR}")

if __name__ == "__main__":
    main()