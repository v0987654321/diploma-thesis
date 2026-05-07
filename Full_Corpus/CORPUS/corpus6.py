import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Optional BERTopic stack
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import umap
    import hdbscan
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    BERTOPIC_AVAILABLE = True
except Exception:
    BERTOPIC_AVAILABLE = False

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_REVIEW_MASTER = SCRIPT_DIR / "data" / "review_master.csv"

TOPICS_DIR = SCRIPT_DIR / "topics"
FIGURES_DIR = SCRIPT_DIR / "figures"
SUMMARY_DIR = SCRIPT_DIR / "summary"

TOPIC_SUBDIRS = [
    "relevance4",
    "africa_corps_or_both",
]

TEXT_COLS = [
    "Headline",
    "Lead",
    "Body_Postclean",
]

MIN_TEXT_LEN = 100
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Topic-family grouping config
MAX_TOPIC_FAMILIES_RELEVANCE4 = 6
MAX_TOPIC_FAMILIES_AC = 4

# =========================
# FRENCH + PROJECT STOPWORDS
# =========================
FRENCH_STOPWORDS = {
    "a","à","afin","ai","aie","aient","ainsi","alors","après","as","au","aucun","aucune",
    "aujourd","aujourdhui","aura","aurai","auraient","aurais","aurait","auras","aurez",
    "auriez","aurions","aurons","auront","aussi","autre","autres","aux","avaient","avais",
    "avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons",
    "beaucoup","bien","ce","ceci","cela","celle","celles","celui","cependant","certain",
    "certaine","certaines","certains","ces","cet","cette","ceux","chaque","chez","ci",
    "comme","comment","contre","d","dans","de","dedans","dehors","depuis","des","dessous",
    "dessus","deux","devant","doit","doivent","donc","dont","du","elle","elles","en","encore",
    "entre","est","et","étaient","étais","était","étant","été","être","eu","eue","eues",
    "eurent","eus","eut","eux","fait","faites","fois","font","hors","ici","il","ils","j",
    "je","jusqu","jusque","l","la","le","les","leur","leurs","lors","lui","ma","mais","me",
    "même","mêmes","mes","moi","moins","mon","ne","ni","nos","notre","nous","on","ont","ou",
    "où","par","parce","parfois","pas","pendant","peu","peut","peuvent","plus","pour","pourquoi",
    "qu","quand","que","quel","quelle","quelles","quels","qui","sa","sans","se","sera","serai",
    "seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses",
    "si","soi","soit","soient","sois","sommes","son","sont","sous","soyez","soyons","sur",
    "ta","te","tes","toi","ton","tous","tout","toute","toutes","très","tu","un","une","vos",
    "votre","vous","y"
}

CUSTOM_STOPWORDS = {
    "mali","maliweb","malien","malienne","maliens","maliennes",
    "bamako","article","articles","selon","déclaré","déclare","déclarent",
    "indiqué","indique","rapporté","rapporte","publié","publiée","auteur",
    "rubrique","encore","ainsi","notamment","également","egalement","cependant",
    "toutefois","plus","moins","fait","faits","faire","être","etre","avoir",
    "doit","doivent","sera","serait","seront","avait","avaient","sont","ont",
    "cette","celui","celle","ceux","celles","leurs","leur","dont","après","avant",
    "depuis","pendant","avec","sans","entre","dans","pour","vers","sur","sous",
    "face","suite","maliens","maliennes"
}

STOPWORDS_ALL = sorted({w.lower() for w in FRENCH_STOPWORDS.union(CUSTOM_STOPWORDS)})

# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def ensure_dirs():
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    for sub in TOPIC_SUBDIRS:
        (TOPICS_DIR / sub).mkdir(parents=True, exist_ok=True)

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def build_full_text(row):
    parts = [safe_str(row.get(col)) for col in TEXT_COLS]
    parts = [p for p in parts if p]
    return " ".join(parts).strip()

def clean_topic_text(text):
    text = safe_str(text).lower()
    if not text:
        return ""

    text = text.replace("’", "'")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿñæœ'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        tok for tok in tokens
        if len(tok) >= 3 and tok not in STOPWORDS_ALL and not tok.isdigit()
    ]

    return " ".join(tokens).strip()

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def save_bar_chart(df, label_col, value_col, title, path, rotate=True, figsize=(10, 6)):
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
    plt.savefig(path, dpi=200)
    plt.close()

def prepare_subset(df, subset_name):
    if subset_name == "relevance4":
        sub = df[df["Relevance"].fillna(0).astype(int) == 4].copy()
    elif subset_name == "africa_corps_or_both":
        sub = df[df["Actor_Mention"].fillna(0).astype(int).isin([2, 3])].copy()
    else:
        raise ValueError(f"Unknown subset name: {subset_name}")

    sub["Topic_Text_Raw"] = sub.apply(build_full_text, axis=1)
    sub["Topic_Text_Raw"] = sub["Topic_Text_Raw"].fillna("").astype(str)
    sub["Topic_Text"] = sub["Topic_Text_Raw"].apply(clean_topic_text)
    sub["Topic_Text_Len"] = sub["Topic_Text"].str.len()
    sub = sub[sub["Topic_Text_Len"] >= MIN_TEXT_LEN].copy()

    return sub

def build_topic_model(embedding_model):
    vectorizer_model = CountVectorizer(
        stop_words=STOPWORDS_ALL,
        ngram_range=(1, 2),
        min_df=2
    )

    return BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        ),
        hdbscan_model=hdbscan.HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        ),
        calculate_probabilities=True,
        verbose=True
    )

def build_topic_family_outputs(topic_model, topic_info, subset_name):
    """
    Build broader topic-family grouping from micro-topics.
    This is intentionally exploratory.
    """
    topic_info_non_outlier = topic_info.copy()
    if "Topic" in topic_info_non_outlier.columns:
        topic_info_non_outlier = topic_info_non_outlier[topic_info_non_outlier["Topic"] != -1].copy()

    if topic_info_non_outlier.empty or len(topic_info_non_outlier) <= 1:
        topic_families = pd.DataFrame(columns=[
            "topic_id", "topic_name", "topic_count",
            "topic_family_id", "topic_family_label"
        ])
        family_members = pd.DataFrame(columns=[
            "topic_family_id", "topic_family_label",
            "topic_id", "topic_name", "topic_count"
        ])
        return topic_families, family_members

    topic_text_rows = []
    for _, row in topic_info_non_outlier.iterrows():
        topic_id = int(row["Topic"])
        topic_name = safe_str(row["Name"]) if "Name" in row.index else f"Topic {topic_id}"
        topic_count = int(row["Count"]) if "Count" in row.index else 0
        topic_terms = topic_model.get_topic(topic_id)
        if topic_terms is None:
            continue
        topic_terms_only = [term for term, _weight in topic_terms[:10]]
        topic_text_rows.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "topic_count": topic_count,
            "topic_terms_text": " ".join(topic_terms_only)
        })

    topic_terms_df = pd.DataFrame(topic_text_rows)
    if topic_terms_df.empty or len(topic_terms_df) <= 1:
        topic_families = topic_terms_df.copy()
        topic_families["topic_family_id"] = 0
        topic_families["topic_family_label"] = topic_families["topic_name"]
        family_members = topic_families[[
            "topic_family_id", "topic_family_label", "topic_id", "topic_name", "topic_count"
        ]].copy()
        return topic_families, family_members

    # Family count heuristic
    if subset_name == "relevance4":
        n_families = min(MAX_TOPIC_FAMILIES_RELEVANCE4, len(topic_terms_df))
    else:
        n_families = min(MAX_TOPIC_FAMILIES_AC, len(topic_terms_df))

    if n_families < 2:
        n_families = 1

    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_ALL, ngram_range=(1, 2))
    X = vectorizer.fit_transform(topic_terms_df["topic_terms_text"])

    if n_families >= len(topic_terms_df):
        # one family per topic if very small
        labels = np.arange(len(topic_terms_df))
    else:
        clustering = AgglomerativeClustering(n_clusters=n_families)
        labels = clustering.fit_predict(X.toarray())

    topic_terms_df["topic_family_id"] = labels

    # Build family labels by concatenating most common terms across members
    family_label_rows = []
    for fam_id, fam_grp in topic_terms_df.groupby("topic_family_id", dropna=False):
        combined_terms = " ".join(fam_grp["topic_terms_text"].tolist()).split()
        term_counts = Counter(combined_terms)
        top_terms = [term for term, _ in term_counts.most_common(4)]
        family_label = " / ".join(top_terms) if top_terms else f"family_{fam_id}"
        family_label_rows.append({
            "topic_family_id": fam_id,
            "topic_family_label": family_label
        })

    family_labels_df = pd.DataFrame(family_label_rows)
    topic_families = topic_terms_df.merge(family_labels_df, on="topic_family_id", how="left")

    family_members = topic_families[[
        "topic_family_id", "topic_family_label", "topic_id", "topic_name", "topic_count"
    ]].copy().sort_values(["topic_family_id", "topic_count"], ascending=[True, False])

    return topic_families, family_members

def export_topic_outputs(subset_name, sub_df, topic_model, topics, probs):
    subset_dir = TOPICS_DIR / subset_name

    # -------------------------
    # Document-topic assignments
    # -------------------------
    doc_topics = sub_df.copy()
    doc_topics["topic_id"] = topics

    if probs is not None:
        try:
            if hasattr(probs, "shape") and len(probs.shape) == 2:
                doc_topics["topic_probability"] = probs.max(axis=1)
            else:
                doc_topics["topic_probability"] = probs
        except Exception:
            doc_topics["topic_probability"] = None
    else:
        doc_topics["topic_probability"] = None

    doc_keep_cols = [
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
        "chapter52_priority_flag",
        "topic_id",
        "topic_probability",
        "Topic_Text",
    ]
    doc_keep_cols = [c for c in doc_keep_cols if c in doc_topics.columns]
    save_csv(doc_topics[doc_keep_cols], subset_dir / "bertopic_document_topics.csv")

    # -------------------------
    # Topic info
    # -------------------------
    topic_info = topic_model.get_topic_info()
    save_csv(topic_info, subset_dir / "bertopic_topic_info.csv")

    # -------------------------
    # Topic terms
    # -------------------------
    topic_terms_rows = []
    if "Topic" in topic_info.columns:
        topic_ids = topic_info["Topic"].tolist()
    else:
        topic_ids = []

    for topic_id in topic_ids:
        if topic_id == -1:
            continue
        topic_terms = topic_model.get_topic(topic_id)
        if topic_terms is None:
            continue
        for rank, (term, weight) in enumerate(topic_terms, start=1):
            topic_terms_rows.append({
                "topic_id": topic_id,
                "rank": rank,
                "term": term,
                "weight": weight
            })

    topic_terms_df = pd.DataFrame(topic_terms_rows)
    save_csv(topic_terms_df, subset_dir / "bertopic_topic_terms.csv")

    # -------------------------
    # Topic families (NEW)
    # -------------------------
    topic_families_df, family_members_df = build_topic_family_outputs(topic_model, topic_info, subset_name)
    save_csv(topic_families_df, subset_dir / "bertopic_topic_families.csv")
    save_csv(family_members_df, subset_dir / "bertopic_topic_family_members.csv")

    # -------------------------
    # Representative docs
    # -------------------------
    rep_docs_rows = []
    if "topic_id" in doc_topics.columns:
        for topic_id, grp in doc_topics.groupby("topic_id", dropna=False):
            if topic_id == -1:
                continue
            grp2 = grp.sort_values("topic_probability", ascending=False).head(10)
            for _, row in grp2.iterrows():
                rep_docs_rows.append({
                    "topic_id": topic_id,
                    "Article_ID": row.get("Article_ID"),
                    "Outlet": row.get("Outlet"),
                    "Date": row.get("Date"),
                    "Headline": row.get("Headline"),
                    "topic_probability": row.get("topic_probability"),
                    "Topic_Text": row.get("Topic_Text"),
                })

    rep_docs_df = pd.DataFrame(rep_docs_rows)
    save_csv(rep_docs_df, subset_dir / "bertopic_representative_docs.csv")

    # -------------------------
    # Topic distribution by outlet
    # -------------------------
    if "Outlet" in doc_topics.columns:
        topic_by_outlet = (
            doc_topics.groupby(["Outlet", "topic_id"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        outlet_totals = topic_by_outlet.groupby("Outlet")["count"].sum().reset_index(name="outlet_total")
        topic_by_outlet = topic_by_outlet.merge(outlet_totals, on="Outlet", how="left")
        topic_by_outlet["percent_within_outlet"] = topic_by_outlet["count"] / topic_by_outlet["outlet_total"] * 100
    else:
        topic_by_outlet = pd.DataFrame(columns=["Outlet", "topic_id", "count", "outlet_total", "percent_within_outlet"])

    save_csv(topic_by_outlet, subset_dir / "bertopic_topic_distribution_by_outlet.csv")

    # -------------------------
    # Topic distribution by year
    # -------------------------
    if "Date" in doc_topics.columns:
        doc_topics["Year"] = pd.to_datetime(doc_topics["Date"], errors="coerce").dt.year
        topic_by_year = (
            doc_topics.groupby(["Year", "topic_id"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        year_totals = topic_by_year.groupby("Year")["count"].sum().reset_index(name="year_total")
        topic_by_year = topic_by_year.merge(year_totals, on="Year", how="left")
        topic_by_year["percent_within_year"] = topic_by_year["count"] / topic_by_year["year_total"] * 100
    else:
        topic_by_year = pd.DataFrame(columns=["Year", "topic_id", "count", "year_total", "percent_within_year"])

    save_csv(topic_by_year, subset_dir / "bertopic_topic_distribution_by_year.csv")

    # -------------------------
    # Figures
    # -------------------------
    topic_info_plot = topic_info[topic_info["Topic"] != -1].copy() if "Topic" in topic_info.columns else pd.DataFrame()
    if not topic_info_plot.empty:
        topic_info_plot = topic_info_plot.sort_values("Count", ascending=False)
        label_col = "Name" if "Name" in topic_info_plot.columns else "Topic"
        save_bar_chart(
            topic_info_plot.head(20),
            label_col=label_col,
            value_col="Count",
            title=f"BERTopic topic sizes: {subset_name}",
            path=FIGURES_DIR / f"topic_{subset_name}_barchart.png"
        )

    if not family_members_df.empty:
        family_sizes = family_members_df.groupby(["topic_family_id", "topic_family_label"], dropna=False)["topic_count"].sum().reset_index()
        family_sizes = family_sizes.sort_values("topic_count", ascending=False)
        save_bar_chart(
            family_sizes,
            label_col="topic_family_label",
            value_col="topic_count",
            title=f"BERTopic topic families: {subset_name}",
            path=FIGURES_DIR / f"topic_{subset_name}_families_barchart.png"
        )

    # -------------------------
    # Summary text
    # -------------------------
    n_docs = len(sub_df)
    n_topic_assignments = len(doc_topics)
    n_topics_non_outlier = topic_info[topic_info["Topic"] != -1].shape[0] if "Topic" in topic_info.columns else 0
    outlier_count = topic_info.loc[topic_info["Topic"] == -1, "Count"].sum() if ("Topic" in topic_info.columns and (topic_info["Topic"] == -1).any()) else 0

    if not topic_info_plot.empty:
        top_topic_row = topic_info_plot.iloc[0]
        top_topic_name = safe_str(top_topic_row["Name"]) if "Name" in top_topic_row.index else safe_str(top_topic_row["Topic"])
        top_topic_count = int(top_topic_row["Count"])
    else:
        top_topic_name = "N/A"
        top_topic_count = 0

    if not family_members_df.empty:
        family_sizes = family_members_df.groupby(["topic_family_id", "topic_family_label"], dropna=False)["topic_count"].sum().reset_index()
        family_sizes = family_sizes.sort_values("topic_count", ascending=False)
        top_family_label = safe_str(family_sizes.iloc[0]["topic_family_label"])
        top_family_count = int(family_sizes.iloc[0]["topic_count"])
        n_families = len(family_sizes)
    else:
        top_family_label = "N/A"
        top_family_count = 0
        n_families = 0

    summary_text = f"""
CORPUS6 TOPIC CLUSTERING SUMMARY
Subset: {subset_name}
Generated: {datetime.now().isoformat(timespec="seconds")}

Methodological note:
- This BERTopic analysis is exploratory and supplementary.
- It is intended for later discussion/triangulation, not as a primary coding layer.
- Topic texts were pre-cleaned and vectorized with French/custom stopword control.
- The script now distinguishes between finer-grained micro-topics and broader topic families.

Input:
- documents in subset after text-length filtering: {n_docs}

BERTopic micro-topic results:
- assigned documents: {n_topic_assignments}
- non-outlier micro-topics: {n_topics_non_outlier}
- outlier cluster size (topic -1): {outlier_count}
- largest micro-topic: {top_topic_name} ({top_topic_count} documents)

Topic-family results:
- broader topic families: {n_families}
- largest topic family: {top_family_label} ({top_family_count} topic-member documents)

Outputs:
- bertopic_document_topics.csv
- bertopic_topic_info.csv
- bertopic_topic_terms.csv
- bertopic_topic_families.csv
- bertopic_topic_family_members.csv
- bertopic_representative_docs.csv
- bertopic_topic_distribution_by_outlet.csv
- bertopic_topic_distribution_by_year.csv
- topic size and topic family figures (.png)
""".strip()

    with open(subset_dir / "topic_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open(subset_dir / "topic_model_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    return {
        "subset": subset_name,
        "documents_after_filtering": n_docs,
        "non_outlier_topics": n_topics_non_outlier,
        "outlier_count": int(outlier_count) if pd.notna(outlier_count) else 0,
        "largest_topic_name": top_topic_name,
        "largest_topic_count": top_topic_count,
        "topic_families": n_families,
        "largest_topic_family": top_family_label,
        "largest_topic_family_count": top_family_count,
    }

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    if not INPUT_REVIEW_MASTER.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_REVIEW_MASTER}")

    review_master = pd.read_csv(INPUT_REVIEW_MASTER, dtype={"Article_ID": str}, low_memory=False)
    review_master = ensure_columns(review_master, TEXT_COLS + [
        "Article_ID", "Outlet", "Date", "Relevance", "Actor_Mention",
        "Successor_Frame", "Dominant_Label", "Dominant_Location",
        "Main_Associated_Actor", "Dominant_Discourse_Support",
        "Any_Review_Flag", "chapter52_priority_flag"
    ], fill_value=None)

    if not BERTOPIC_AVAILABLE:
        raise ImportError(
            "BERTopic stack is not available. Please install: bertopic, sentence-transformers, umap-learn, hdbscan, scikit-learn"
        )

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    corpus6_summary_rows = []

    for subset_name in TOPIC_SUBDIRS:
        print(f"\n=== Running BERTopic for subset: {subset_name} ===")
        sub_df = prepare_subset(review_master, subset_name)

        subset_dir = TOPICS_DIR / subset_name

        if sub_df.empty:
            empty_summary = f"""
CORPUS6 TOPIC CLUSTERING SUMMARY
Subset: {subset_name}
Generated: {datetime.now().isoformat(timespec="seconds")}

No documents available in this subset after text-length filtering.
""".strip()

            with open(subset_dir / "topic_model_summary.txt", "w", encoding="utf-8") as f:
                f.write(empty_summary)

            with open(subset_dir / "topic_model_summary.md", "w", encoding="utf-8") as f:
                f.write(empty_summary)

            corpus6_summary_rows.append({
                "subset": subset_name,
                "documents_after_filtering": 0,
                "non_outlier_topics": 0,
                "outlier_count": 0,
                "largest_topic_name": "N/A",
                "largest_topic_count": 0,
                "topic_families": 0,
                "largest_topic_family": "N/A",
                "largest_topic_family_count": 0,
            })
            continue

        texts = sub_df["Topic_Text"].tolist()
        print(f"Documents in subset after filtering: {len(texts)}")

        topic_model = build_topic_model(embedding_model)
        topics, probs = topic_model.fit_transform(texts)

        subset_summary = export_topic_outputs(
            subset_name=subset_name,
            sub_df=sub_df,
            topic_model=topic_model,
            topics=topics,
            probs=probs
        )
        corpus6_summary_rows.append(subset_summary)

    corpus6_summary_df = pd.DataFrame(corpus6_summary_rows)
    corpus6_summary_path = TOPICS_DIR / "corpus6_summary.csv"
    corpus6_summary_df.to_csv(corpus6_summary_path, index=False, encoding="utf-8-sig")

    overall_summary_text = f"""
CORPUS6 OVERALL SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Methodological positioning:
- BERTopic was used only as a supplementary exploratory clustering layer.
- Results from this script are intended for later discussion and triangulation, not as a primary coding procedure.
- Topic texts were pre-cleaned and vectorized with French/custom stopword control.
- The output now includes finer-grained micro-topics and broader topic-family groupings.

Subsets processed:
- relevance4
- africa_corps_or_both

Output:
- topic-specific files in topics/relevance4/
- topic-specific files in topics/africa_corps_or_both/
- corpus6_summary.csv
""".strip()

    with open(SUMMARY_DIR / "corpus6_summary.txt", "w", encoding="utf-8") as f:
        f.write(overall_summary_text)

    with open(SUMMARY_DIR / "corpus6_summary.md", "w", encoding="utf-8") as f:
        f.write(overall_summary_text)

    print("\n=== CORPUS6 DIAGNOSTICS ===\n")
    print(corpus6_summary_df)
    print("\nSaved outputs to:")
    print(f"- {TOPICS_DIR}")
    print(f"- {FIGURES_DIR}")
    print(f"- {SUMMARY_DIR}")

if __name__ == "__main__":
    main()