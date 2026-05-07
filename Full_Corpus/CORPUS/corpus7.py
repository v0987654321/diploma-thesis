import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd

# Optional spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_REVIEW_MASTER = SCRIPT_DIR / "data" / "review_master.csv"

NER_DIR = SCRIPT_DIR / "ner"
SUMMARY_DIR = SCRIPT_DIR / "summary"

SPACY_FRENCH_MODELS = [
    "fr_core_news_md",
    "fr_core_news_sm",
]

TEXT_COLS = [
    "Headline",
    "Lead",
    "Body_Postclean",
]

META_COLS = [
    "Article_ID",
    "Outlet",
    "Date",
    "Headline",
    "Relevance",
    "Actor_Mention",
    "Successor_Frame",
    "Dominant_Label",
    "Dominant_Location",
    "Main_Associated_Actor",
    "Dominant_Discourse_Support",
    "Any_Review_Flag",
    "source_attributed_flag",
    "likely_republished_flag",
    "chapter52_priority_flag",
]

# Relevant NER types to keep
KEEP_ENTITY_LABELS = {
    "PER", "PERSON",
    "ORG",
    "GPE", "LOC",
    "NORP",
}

SUBSET_DEFINITIONS = {
    "relevance_4": lambda df: df["Relevance"].fillna(0).astype(int) == 4,
    "relevance_3plus": lambda df: df["Relevance"].fillna(0).astype(int).isin([3, 4]),
    "africa_corps_cases": lambda df: df["Actor_Mention"].fillna(0).astype(int).isin([2, 3]),
    "successor_cases": lambda df: df["Successor_Frame"].fillna(0).astype(int) == 1,
    "hra_cases": lambda df: df["Human_Rights_Abuse"].fillna(0).astype(int) == 1 if "Human_Rights_Abuse" in df.columns else pd.Series(False, index=df.index),
    "review_flagged": lambda df: df["Any_Review_Flag"].fillna(0).astype(int) == 1,
    "source_attributed": lambda df: df["source_attributed_flag"].fillna(0).astype(int) == 1 if "source_attributed_flag" in df.columns else pd.Series(False, index=df.index),
    "chapter52_priority": lambda df: df["chapter52_priority_flag"].fillna(0).astype(int) == 1 if "chapter52_priority_flag" in df.columns else pd.Series(False, index=df.index),
}

TARGET_PATTERNS = {
    "wagner": [
        r"\bwagner\b",
        r"\bgroupe\s+wagner\b",
        r"\bwagner\s+group\b",
    ],
    "africa_corps": [
        r"\bafrica\s+corps\b",
        r"\bcorps\s+africain\b",
        r"\bcorps\s+africain\s+russe\b",
    ],
}

# =========================
# CUSTOM ENTITY NORMALIZATION
# =========================
ENTITY_NORMALIZATION_MAP = {
    # Russia cluster
    "russie": "Russia",
    "moscou": "Russia",
    "kremlin": "Russia",
    "fédération de russie": "Russia",
    "federation de russie": "Russia",
    "russian federation": "Russia",

    # France cluster
    "france": "France",
    "paris": "France",
    "barkhane": "France",
    "serval": "France",

    # Mali state/military
    "fama": "FAMa",
    "forces armées maliennes": "FAMa",
    "forces armees maliennes": "FAMa",
    "armée malienne": "FAMa",
    "armee malienne": "FAMa",

    "assimi goïta": "Assimi Goïta",
    "assimi goita": "Assimi Goïta",
    "colonel goïta": "Assimi Goïta",
    "colonel goita": "Assimi Goïta",

    # UN / MINUSMA
    "minusma": "MINUSMA",
    "onu": "UN / MINUSMA",
    "nations unies": "UN / MINUSMA",

    # ECOWAS
    "cedeao": "ECOWAS / Regional",
    "ecowas": "ECOWAS / Regional",
    "union africaine": "African Union / Regional",
    "ua": "African Union / Regional",

    # Places
    "bamako": "Bamako",
    "kidal": "Kidal",
    "gao": "Gao",
    "tombouctou": "Tombouctou",
    "ménaka": "Ménaka",
    "menaka": "Ménaka",
    "mopti": "Mopti",

    # Sahel regional
    "burkina faso": "Burkina Faso",
    "burkina": "Burkina Faso",
    "niger": "Niger",
    "niamey": "Niger",
    "mali": "Mali",
    "sénégal": "Senegal",
    "senegal": "Senegal",
    "centrafrique": "Central African Republic",
    "république centrafricaine": "Central African Republic",
    "republique centrafricaine": "Central African Republic",

    # Jihadist clusters
    "jnim": "JNIM / GSIM",
    "gsim": "JNIM / GSIM",
    "aqmi": "AQMI",
    "etat islamique": "Islamic State",
    "état islamique": "Islamic State",
    "eis": "Islamic State",
}

# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def ensure_dirs():
    NER_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def build_full_text(row):
    parts = [safe_str(row.get(col)) for col in TEXT_COLS]
    parts = [p for p in parts if p]
    return " ".join(parts).strip()

def split_sentences(text):
    text = safe_str(text)
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

def normalize_entity_text(text):
    text = safe_str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("’", "'")
    text = text.lower()
    return ENTITY_NORMALIZATION_MAP.get(text, text.title() if text else "")

def get_spacy_model():
    if not SPACY_AVAILABLE:
        return None, "spaCy_not_available"

    for model_name in SPACY_FRENCH_MODELS:
        try:
            nlp = spacy.load(model_name, disable=["tagger", "parser", "lemmatizer"])
            return nlp, model_name
        except Exception:
            continue

    return None, "no_french_model_found"

def detect_target_in_text(text):
    text = safe_str(text).lower()
    has_wagner = any(re.search(p, text, flags=re.IGNORECASE) for p in TARGET_PATTERNS["wagner"])
    has_ac = any(re.search(p, text, flags=re.IGNORECASE) for p in TARGET_PATTERNS["africa_corps"])
    return has_wagner, has_ac

def extract_entities_from_row(row, nlp):
    article_id = safe_str(row.get("Article_ID"))
    full_text = build_full_text(row)
    if not full_text:
        return []

    doc = nlp(full_text)
    entities = []

    for ent in doc.ents:
        ent_label = safe_str(ent.label_)
        if ent_label not in KEEP_ENTITY_LABELS:
            continue

        ent_text_raw = safe_str(ent.text)
        ent_text_norm = normalize_entity_text(ent_text_raw)

        if not ent_text_norm:
            continue

        record = {
            "entity_raw": ent_text_raw,
            "entity_norm": ent_text_norm,
            "entity_label": ent_label,
        }

        for col in META_COLS:
            record[col] = row.get(col)

        entities.append(record)

    return entities

def extract_entity_sentence_rows(row, nlp):
    article_id = safe_str(row.get("Article_ID"))
    full_text = build_full_text(row)
    if not full_text:
        return []

    sentences = split_sentences(full_text)
    rows = []

    for sent_idx, sent in enumerate(sentences, start=1):
        doc = nlp(sent)
        sent_has_wagner, sent_has_ac = detect_target_in_text(sent)

        sent_entities = []
        for ent in doc.ents:
            ent_label = safe_str(ent.label_)
            if ent_label not in KEEP_ENTITY_LABELS:
                continue

            ent_text_raw = safe_str(ent.text)
            ent_text_norm = normalize_entity_text(ent_text_raw)
            if not ent_text_norm:
                continue

            sent_entities.append((ent_text_raw, ent_text_norm, ent_label))

        for ent_text_raw, ent_text_norm, ent_label in sent_entities:
            rec = {
                "sentence_index": sent_idx,
                "sentence_full": safe_str(sent),
                "entity_raw": ent_text_raw,
                "entity_norm": ent_text_norm,
                "entity_label": ent_label,
                "sentence_has_wagner": int(sent_has_wagner),
                "sentence_has_africa_corps": int(sent_has_ac),
            }
            for col in META_COLS:
                rec[col] = row.get(col)
            rows.append(rec)

    return rows

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    if not INPUT_REVIEW_MASTER.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_REVIEW_MASTER}")

    review_master = pd.read_csv(INPUT_REVIEW_MASTER, dtype={"Article_ID": str}, low_memory=False)
    review_master = ensure_columns(
        review_master,
        META_COLS + TEXT_COLS + ["Human_Rights_Abuse"],
        fill_value=None
    )

    nlp, model_info = get_spacy_model()
    if nlp is None:
        raise ImportError(
            f"French spaCy model not available. Status: {model_info}. "
            f"Install spaCy and one of: {', '.join(SPACY_FRENCH_MODELS)}"
        )

    # =========================
    # ENTITY EXTRACTION
    # =========================
    entity_rows = []
    sentence_entity_rows = []

    for _, row in review_master.iterrows():
        entity_rows.extend(extract_entities_from_row(row, nlp))
        sentence_entity_rows.extend(extract_entity_sentence_rows(row, nlp))

    entities_df = pd.DataFrame(entity_rows)
    sentence_entities_df = pd.DataFrame(sentence_entity_rows)

    if entities_df.empty:
        entities_df = pd.DataFrame(columns=META_COLS + ["entity_raw", "entity_norm", "entity_label"])
    if sentence_entities_df.empty:
        sentence_entities_df = pd.DataFrame(columns=META_COLS + [
            "sentence_index", "sentence_full", "entity_raw", "entity_norm",
            "entity_label", "sentence_has_wagner", "sentence_has_africa_corps"
        ])

    save_csv(entities_df, NER_DIR / "entities_article_level.csv")
    save_csv(sentence_entities_df, NER_DIR / "entities_sentence_level.csv")

    # =========================
    # TOP ENTITIES OVERALL
    # =========================
    if not entities_df.empty:
        top_entities_overall = (
            entities_df.groupby(["entity_norm", "entity_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        top_entities_overall = pd.DataFrame(columns=["entity_norm", "entity_label", "count"])

    save_csv(top_entities_overall, NER_DIR / "top_entities_overall.csv")

    # =========================
    # TOP ENTITIES BY SUBSET
    # =========================
    subset_summary_rows = []

    for subset_name, subset_fn in SUBSET_DEFINITIONS.items():
        mask = subset_fn(review_master)
        subset_ids = set(review_master.loc[mask, "Article_ID"].dropna().astype(str).tolist())

        if not subset_ids:
            top_entities_subset = pd.DataFrame(columns=["entity_norm", "entity_label", "count"])
        else:
            top_entities_subset = (
                entities_df[entities_df["Article_ID"].astype(str).isin(subset_ids)]
                .groupby(["entity_norm", "entity_label"], dropna=False)
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .reset_index(drop=True)
            )

        save_csv(top_entities_subset, NER_DIR / f"top_entities_{subset_name}.csv")

        subset_summary_rows.append({
            "subset": subset_name,
            "articles_total": len(subset_ids),
            "entity_mentions_total": int(top_entities_subset["count"].sum()) if not top_entities_subset.empty else 0,
            "unique_entities": top_entities_subset["entity_norm"].nunique() if not top_entities_subset.empty else 0
        })

    subset_entity_summary = pd.DataFrame(subset_summary_rows)
    save_csv(subset_entity_summary, NER_DIR / "subset_entity_summary.csv")

    # =========================
    # TOP ENTITIES BY OUTLET
    # =========================
    if not entities_df.empty and "Outlet" in entities_df.columns:
        top_entities_by_outlet = (
            entities_df.groupby(["Outlet", "entity_norm", "entity_label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["Outlet", "count"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        top_entities_by_outlet = pd.DataFrame(columns=["Outlet", "entity_norm", "entity_label", "count"])

    save_csv(top_entities_by_outlet, NER_DIR / "top_entities_by_outlet.csv")

    # =========================
    # TOP ENTITIES BY LABEL
    # =========================
    if not entities_df.empty:
        top_entities_by_label = (
            entities_df.groupby(["entity_label", "entity_norm"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["entity_label", "count"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        top_entities_by_label = pd.DataFrame(columns=["entity_label", "entity_norm", "count"])

    save_csv(top_entities_by_label, NER_DIR / "top_entities_by_label.csv")

    # =========================
    # ENTITY CO-OCCURRENCE WITH TARGETS (ARTICLE LEVEL)
    # =========================
    article_text_map = review_master[["Article_ID"] + TEXT_COLS].copy()
    article_text_map["Full_Text"] = article_text_map.apply(build_full_text, axis=1)

    target_cooccurrence_rows = []

    for _, row in article_text_map.iterrows():
        article_id = safe_str(row.get("Article_ID"))
        text = safe_str(row.get("Full_Text"))
        has_wagner, has_ac = detect_target_in_text(text)

        if not has_wagner and not has_ac:
            continue

        sub_ents = entities_df[entities_df["Article_ID"].astype(str) == article_id].copy()
        unique_ents = sub_ents[["entity_norm", "entity_label"]].drop_duplicates()

        for _, ent_row in unique_ents.iterrows():
            target_cooccurrence_rows.append({
                "Article_ID": article_id,
                "target_wagner": int(has_wagner),
                "target_africa_corps": int(has_ac),
                "entity_norm": ent_row["entity_norm"],
                "entity_label": ent_row["entity_label"],
            })

    target_cooccurrence_df = pd.DataFrame(target_cooccurrence_rows)
    save_csv(target_cooccurrence_df, NER_DIR / "target_entity_cooccurrence_article_level.csv")

    if not target_cooccurrence_df.empty:
        target_entity_profile = (
            target_cooccurrence_df.groupby(
                ["target_wagner", "target_africa_corps", "entity_norm", "entity_label"],
                dropna=False
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        target_entity_profile = pd.DataFrame(columns=[
            "target_wagner", "target_africa_corps", "entity_norm", "entity_label", "count"
        ])

    save_csv(target_entity_profile, NER_DIR / "target_entity_profile.csv")

    # =========================
    # ENTITY CO-OCCURRENCE WITH TARGETS (SENTENCE LEVEL)
    # =========================
    if not sentence_entities_df.empty:
        sent_target_entity_profile = (
            sentence_entities_df[
                (sentence_entities_df["sentence_has_wagner"] == 1) |
                (sentence_entities_df["sentence_has_africa_corps"] == 1)
            ]
            .groupby(
                ["sentence_has_wagner", "sentence_has_africa_corps", "entity_norm", "entity_label"],
                dropna=False
            )
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
    else:
        sent_target_entity_profile = pd.DataFrame(columns=[
            "sentence_has_wagner", "sentence_has_africa_corps", "entity_norm", "entity_label", "count"
        ])

    save_csv(sent_target_entity_profile, NER_DIR / "target_entity_profile_sentence_level.csv")

    # =========================
    # ENTITY EVIDENCE PACKS
    # =========================
    # representative sentence contexts for top entities
    if not sentence_entities_df.empty and not top_entities_overall.empty:
        top_entity_names = top_entities_overall.head(30)["entity_norm"].tolist()
        entity_evidence = sentence_entities_df[
            sentence_entities_df["entity_norm"].isin(top_entity_names)
        ].copy()
        entity_evidence = entity_evidence.sort_values(
            ["entity_norm", "Article_ID", "sentence_index"]
        ).reset_index(drop=True)
    else:
        entity_evidence = pd.DataFrame(columns=sentence_entities_df.columns if not sentence_entities_df.empty else [])

    save_csv(entity_evidence, NER_DIR / "entity_evidence_top30.csv")

    # =========================
    # OUTLET × ENTITY SUMMARY
    # =========================
    if not entities_df.empty and "Outlet" in entities_df.columns:
        outlet_entity_summary = (
            entities_df.groupby(["Outlet", "entity_norm"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["Outlet", "count"], ascending=[True, False])
        )
    else:
        outlet_entity_summary = pd.DataFrame(columns=["Outlet", "entity_norm", "count"])

    save_csv(outlet_entity_summary, NER_DIR / "outlet_entity_summary.csv")

    # =========================
    # KEY ENTITY PACKS
    # =========================
    key_entity_groups = {
        "russia_context": ["Russia"],
        "france_context": ["France"],
        "malian_state_military": ["FAMa", "Assimi Goïta"],
        "un_minusma": ["MINUSMA", "UN / MINUSMA"],
        "regional_ecowas": ["ECOWAS / Regional", "African Union / Regional"],
        "burkina_niger": ["Burkina Faso", "Niger"],
    }

    for pack_name, targets in key_entity_groups.items():
        sub = sentence_entities_df[sentence_entities_df["entity_norm"].isin(targets)].copy()
        save_csv(sub, NER_DIR / f"entity_pack_{pack_name}.csv")

    # =========================
    # SUMMARY TEXTS
    # =========================
    total_articles = len(review_master)
    total_entity_mentions = len(entities_df)
    unique_entities_total = entities_df["entity_norm"].nunique() if not entities_df.empty else 0

    if not top_entities_overall.empty:
        top_entity_name = safe_str(top_entities_overall.iloc[0]["entity_norm"])
        top_entity_count = int(top_entities_overall.iloc[0]["count"])
    else:
        top_entity_name = "N/A"
        top_entity_count = 0

    summary_text = f"""
CORPUS7 SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Input:
- review_master rows: {total_articles}

NER setup:
- spaCy available: {"yes" if SPACY_AVAILABLE else "no"}
- French model used: {model_info}

Outputs:
- article-level entities
- sentence-level entities
- top entities overall
- top entities by subset
- top entities by outlet
- top entities by label
- article-level target/entity co-occurrence
- sentence-level target/entity co-occurrence
- evidence packs for top entities
- thematic entity packs

Headline findings:
- total entity mentions retained: {total_entity_mentions}
- unique normalized entities: {unique_entities_total}
- top normalized entity: {top_entity_name} ({top_entity_count} mentions)
""".strip()

    with open(SUMMARY_DIR / "corpus7_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open(SUMMARY_DIR / "corpus7_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    ner_notes_text = f"""
CORPUS7 NER NOTES
Generated: {datetime.now().isoformat(timespec="seconds")}

Methodological note:
- This script provides exploratory NER enrichment.
- Entity normalization is partly model-driven and partly rule-based.
- Results should be used as analytical support, not as definitive actor-coding replacements.

Analytical uses:
- identify recurrent actor environments around Wagner / Africa Corps
- compare entity salience by outlet
- inspect sentence-level co-occurrence with target actors
- support Q3/Q5 interpretation
- enrich later synthesis tables
""".strip()

    with open(SUMMARY_DIR / "corpus7_ner_notes.txt", "w", encoding="utf-8") as f:
        f.write(ner_notes_text)

    print("\n=== CORPUS7 DIAGNOSTICS ===\n")
    print(f"Review master rows: {total_articles}")
    print(f"spaCy available: {'yes' if SPACY_AVAILABLE else 'no'}")
    print(f"French model used: {model_info}")
    print(f"Entity mentions retained: {total_entity_mentions}")
    print(f"Unique normalized entities: {unique_entities_total}")
    print(f"Top normalized entity: {top_entity_name} ({top_entity_count} mentions)")
    print("\nSaved outputs to:")
    print(f"- {NER_DIR}")
    print(f"- {SUMMARY_DIR}")


if __name__ == "__main__":
    main()