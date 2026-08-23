import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Optional spaCy import
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

LEXICAL_NORM_DIR = SCRIPT_DIR / "lexical_norm"
WORDCLOUD_DIR = SCRIPT_DIR / "figures" / "wordclouds"
SUMMARY_DIR = SCRIPT_DIR / "summary"

MIN_TOKEN_LEN = 3
TOP_N = 50
WORDCLOUD_WIDTH = 1400
WORDCLOUD_HEIGHT = 800
WORDCLOUD_BG = "white"

# spaCy French model preference order
SPACY_FRENCH_MODELS = [
    "fr_core_news_md",
    "fr_core_news_sm",
]

TEXT_COLS = [
    "Headline",
    "Lead",
    "Body_Postclean",
]

SUBSET_DEFINITIONS = {
    "relevance_4": lambda df: df["Relevance"].fillna(0).astype(int) == 4,
    "relevance_3plus": lambda df: df["Relevance"].fillna(0).astype(int).isin([3, 4]),
    "africa_corps_cases": lambda df: df["Actor_Mention"].fillna(0).astype(int).isin([2, 3]),
    "successor_cases": lambda df: df["Successor_Frame"].fillna(0).astype(int) == 1,
    "hra_cases": lambda df: df["Human_Rights_Abuse"].fillna(0).astype(int) == 1,
    "security_effectiveness_cases": lambda df: df["Security_Effectiveness"].fillna(0).astype(int) == 1,
    "geopolitical_rivalry_cases": lambda df: df["Geopolitical_Rivalry"].fillna(0).astype(int) == 1,
    "review_flagged": lambda df: df["Any_Review_Flag"].fillna(0).astype(int) == 1,
    "source_attributed": lambda df: df["source_attributed_flag"].fillna(0).astype(int) == 1,
    "likely_republished": lambda df: df["likely_republished_flag"].fillna(0).astype(int) == 1,
    "chapter52_priority": lambda df: df["chapter52_priority_flag"].fillna(0).astype(int) == 1,
}

# =========================
# STOPWORDS
# =========================
# Light French/general stopwords
FRENCH_STOPWORDS = {
    "a", "à", "afin", "ai", "aie", "aient", "ainsi", "alors", "après", "as", "au", "aucun",
    "aucune", "aujourd", "aujourdhui", "aura", "aurai", "auraient", "aurais", "aurait", "auras",
    "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autre", "autres", "aux", "avaient",
    "avais", "avait", "avant", "avec", "avez", "aviez", "avions", "avoir", "avons", "ayant", "ayez",
    "ayons", "beaucoup", "bien", "ce", "ceci", "cela", "celle", "celles", "celui", "cependant",
    "certain", "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "chaque", "chez",
    "ci", "comme", "comment", "contre", "d", "da", "dans", "de", "debout", "dedans", "dehors",
    "depuis", "des", "dessous", "dessus", "deux", "devant", "doit", "doivent", "donc", "dont", "du",
    "elle", "elles", "en", "encore", "entre", "est", "et", "étaient", "étais", "était", "étant",
    "été", "être", "eu", "eue", "eues", "eurent", "eus", "eusse", "eussent", "eusses", "eussiez",
    "eussions", "eut", "eux", "fait", "faites", "fois", "font", "hors", "ici", "il", "ils", "j",
    "je", "jusqu", "jusque", "l", "la", "le", "les", "leur", "leurs", "lors", "lui", "ma", "mais",
    "me", "même", "mêmes", "mes", "moi", "moins", "mon", "ne", "ni", "nos", "notre", "nous", "on",
    "ont", "ou", "où", "par", "parce", "parfois", "pas", "pendant", "peu", "peut", "peuvent", "plus",
    "plutôt", "pour", "pourquoi", "qu", "quand", "que", "quel", "quelle", "quelles", "quels", "qui",
    "sa", "sans", "se", "sera", "serai", "seraient", "serais", "serait", "seras", "serez", "seriez",
    "serions", "serons", "seront", "ses", "si", "sien", "sienne", "siennes", "siens", "soi", "soit",
    "soient", "sois", "sommes", "son", "sont", "sous", "soyez", "soyons", "sur", "ta", "te", "tes",
    "toi", "ton", "tous", "tout", "toute", "toutes", "très", "tu", "un", "une", "vos", "votre",
    "vous", "y", "été", "étées",
}

# Project-specific / corpus-specific cleanup stopwords
CUSTOM_STOPWORDS = {
    "mali", "mali", "malien", "malienne", "maliens", "maliennes",
    "bamako", "article", "articles", "selon", "déclaré", "indiqué",
    "rapporté", "publié", "publiée", "auteur", "rubrique", "ainsi",
    "fait", "été", "être", "plus", "moins", "encore", "contre",
    "après", "avant", "durant", "depuis", "autour", "cela", "celui",
    "celle", "ceux", "celles", "leurs", "leur", "dont", "cette",
    "ainsi", "notamment", "également", "ainsi", "parmi", "toutefois",
    "cependant", "encore", "auprès", "après", "avant", "entre",
    "via", "selon", "face", "suite", "ainsi", "été", "fait", "faits",
    "faire", "avoir", "être", "doit", "doivent", "peut", "peuvent",
    "sera", "serait", "seront", "avait", "avaient", "ont", "sont",
    "cest", "quest", "dans", "pour", "avec", "sans", "entre", "sous",
    "sur", "chez", "vers", "pendant", "depuis", "tandis", "alors",
}

COMBINED_STOPWORDS = {w.lower() for w in FRENCH_STOPWORDS.union(CUSTOM_STOPWORDS)}

# =========================
# CUSTOM NORMALIZATION MAP
# after lemmatization (or after surface cleanup if no lemma)
# =========================
CUSTOM_NORMALIZATION_MAP = {
    # actor naming
    "mercenaires": "mercenaire",
    "mercenaire": "mercenaire",
    "paramilitaires": "paramilitaire",
    "paramilitaire": "paramilitaire",
    "instructeurs": "instructeur",
    "instructeur": "instructeur",
    "formateurs": "formateur",
    "formateur": "formateur",
    "conseillers": "conseiller",
    "conseiller": "conseiller",
    "coopérants": "cooperant",
    "coopérant": "cooperant",
    "alliés": "allie",
    "allié": "allie",
    "partenaires": "partenaire",
    "partenaire": "partenaire",

    # grammar/number variation
    "russes": "russe",
    "russe": "russe",
    "français": "francais",
    "française": "francais",
    "françaises": "francais",
    "occidentaux": "occidental",
    "occidental": "occidental",
    "terroristes": "terroriste",
    "terroriste": "terroriste",
    "jihadistes": "jihadiste",
    "jihadiste": "jihadiste",
    "insurgés": "insurge",
    "insurgé": "insurge",

    # rights / violence
    "exactions": "exaction",
    "exaction": "exaction",
    "massacres": "massacre",
    "massacre": "massacre",
    "exécutions": "execution",
    "exécution": "execution",
    "violations": "violation",
    "violation": "violation",
    "atrocités": "atrocite",
    "atrocité": "atrocite",
    "crimes": "crime",
    "crime": "crime",

    # sovereignty
    "souverainetés": "souverainete",
    "souveraineté": "souverainete",
    "partenariat": "partenariat",
    "partenariats": "partenariat",

    # geopolitical
    "compétitions": "competition",
    "compétition": "competition",
    "rivalités": "rivalite",
    "rivalité": "rivalite",
    "influences": "influence",
    "influence": "influence",

    # economic
    "mines": "mine",
    "mine": "mine",
    "miniers": "minier",
    "minier": "minier",
    "gisements": "gisement",
    "gisement": "gisement",
    "ressources": "ressource",
    "ressource": "ressource",

    # states / places
    "fama": "fama",
    "forces": "force",
    "armées": "armee",
    "armée": "armee",
    "autorités": "autorite",
    "autorité": "autorite",
}

# =========================
# ANALYTICAL NORMALIZATION MAP
# maps normalized tokens into broader analytical buckets
# =========================
ANALYTICAL_NORMALIZATION_MAP = {
    # actor naming / labels
    "mercenaire": "label_mercenary",
    "paramilitaire": "label_paramilitary",
    "instructeur": "label_instructor",
    "formateur": "label_instructor",
    "conseiller": "label_instructor",
    "cooperant": "label_instructor",
    "partenaire": "label_partner",
    "allie": "label_partner",

    # security
    "terroriste": "armed_islamist_adversary",
    "jihadiste": "armed_islamist_adversary",
    "insurge": "armed_islamist_adversary",
    "aqmi": "armed_islamist_adversary",
    "jnim": "armed_islamist_adversary",
    "gsim": "armed_islamist_adversary",
    "eis": "armed_islamist_adversary",

    # violence / abuse
    "exaction": "abuse_violence_language",
    "massacre": "abuse_violence_language",
    "execution": "abuse_violence_language",
    "torture": "abuse_violence_language",
    "violation": "abuse_violence_language",
    "atrocite": "abuse_violence_language",
    "crime": "abuse_violence_language",

    # sovereignty
    "souverainete": "sovereignty_language",
    "autonomie": "sovereignty_language",
    "partenariat": "partnership_cooperation_language",
    "cooperation": "partnership_cooperation_language",
    "partenaire": "partnership_cooperation_language",
    "allie": "partnership_cooperation_language",

    # effectiveness
    "efficacite": "security_effectiveness_language",
    "efficace": "security_effectiveness_language",
    "stabilisation": "security_effectiveness_language",
    "stabiliser": "security_effectiveness_language",
    "resultat": "security_effectiveness_language",
    "succes": "security_effectiveness_language",

    # geopolitical
    "rivalite": "geopolitical_competition_language",
    "competition": "geopolitical_competition_language",
    "influence": "geopolitical_competition_language",
    "multipolarite": "geopolitical_competition_language",
    "geopolitique": "geopolitical_competition_language",

    # anti/neocolonial
    "colonial": "anti_neocolonial_language",
    "neocolonial": "anti_neocolonial_language",
    "imperialisme": "anti_neocolonial_language",
    "francafrique": "anti_neocolonial_language",
    "emancipation": "anti_neocolonial_language",

    # economic
    "mine": "resource_extraction_language",
    "minier": "resource_extraction_language",
    "gisement": "resource_extraction_language",
    "ressource": "resource_extraction_language",
    "or": "resource_extraction_language",
    "gold": "resource_extraction_language",

    # external actor context
    "russe": "russian_actor_context",
    "russie": "russian_actor_context",
    "moscou": "russian_actor_context",
    "kremlin": "russian_actor_context",
    "poutine": "russian_actor_context",

    "francais": "french_actor_context",
    "france": "french_actor_context",
    "paris": "french_actor_context",
    "barkhane": "french_actor_context",
    "serval": "french_actor_context",

    # mali state / military
    "fama": "malian_state_military_context",
    "armee": "malian_state_military_context",
    "force": "malian_state_military_context",
    "autorite": "malian_state_military_context",
    "gouvernement": "malian_state_military_context",
    "transition": "malian_state_military_context",
}

# =========================
# TOKENIZER REGEX
# =========================
TOKEN_REGEX = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœÆæÇç'-]+")

# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def ensure_dirs():
    LEXICAL_NORM_DIR.mkdir(parents=True, exist_ok=True)
    WORDCLOUD_DIR.mkdir(parents=True, exist_ok=True)
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

def get_spacy_model():
    if not SPACY_AVAILABLE:
        return None, "spaCy_not_available"

    for model_name in SPACY_FRENCH_MODELS:
        try:
            nlp = spacy.load(model_name, disable=["ner", "parser"])
            return nlp, model_name
        except Exception:
            continue

    return None, "no_french_model_found"

def simple_tokenize(text):
    return TOKEN_REGEX.findall(safe_str(text))

def clean_surface_token(tok):
    tok = safe_str(tok).lower().strip()
    tok = tok.replace("’", "'")
    tok = tok.strip("-'")
    if not tok:
        return ""
    if tok.isdigit():
        return ""
    if len(tok) < MIN_TOKEN_LEN:
        return ""
    if tok in COMBINED_STOPWORDS:
        return ""
    return tok

def normalize_custom(tok):
    tok = safe_str(tok).lower()
    return CUSTOM_NORMALIZATION_MAP.get(tok, tok)

def normalize_analytic(tok):
    tok = safe_str(tok).lower()
    return ANALYTICAL_NORMALIZATION_MAP.get(tok, tok)

def extract_surface_clean_tokens(text):
    raw_tokens = simple_tokenize(text)
    cleaned = []
    for tok in raw_tokens:
        t = clean_surface_token(tok)
        if t:
            cleaned.append(t)
    return cleaned

def extract_lemma_tokens(text, nlp):
    if nlp is None:
        # fallback to cleaned surface
        return extract_surface_clean_tokens(text)

    doc = nlp(safe_str(text))
    lemmas = []

    for token in doc:
        lemma = safe_str(token.lemma_).lower().strip()
        lemma = lemma.replace("’", "'")
        lemma = lemma.strip("-'")

        if not lemma:
            continue
        if lemma.isdigit():
            continue
        if len(lemma) < MIN_TOKEN_LEN:
            continue
        if lemma in COMBINED_STOPWORDS:
            continue

        lemmas.append(lemma)

    return lemmas

def token_counter_for_subset(df, text_col="Full_Text", mode="surface", nlp=None):
    counter = Counter()

    for _, row in df.iterrows():
        text = safe_str(row.get(text_col))
        if not text:
            continue

        if mode == "surface":
            toks = extract_surface_clean_tokens(text)
        elif mode == "lemma":
            toks = extract_lemma_tokens(text, nlp)
        elif mode == "custom":
            toks = [normalize_custom(t) for t in extract_lemma_tokens(text, nlp)]
        elif mode == "analytic":
            toks = [normalize_analytic(normalize_custom(t)) for t in extract_lemma_tokens(text, nlp)]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        toks = [t for t in toks if t and len(t) >= MIN_TOKEN_LEN and t not in COMBINED_STOPWORDS]
        counter.update(toks)

    return counter

def counter_to_df(counter, subset_name, mode_name):
    rows = []
    total = sum(counter.values())

    for token, count in counter.most_common():
        rows.append({
            "subset": subset_name,
            "mode": mode_name,
            "token": token,
            "count": count,
            "percent_of_tokens": (count / total * 100) if total else 0.0
        })

    return pd.DataFrame(rows)

def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def save_wordcloud(counter, title, filename):
    if not counter:
        return

    wc = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color=WORDCLOUD_BG,
        collocations=False
    ).generate_from_frequencies(counter)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(WORDCLOUD_DIR / filename, dpi=200)
    plt.close()

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    if not INPUT_REVIEW_MASTER.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_REVIEW_MASTER}")

    review_master = pd.read_csv(INPUT_REVIEW_MASTER, dtype={"Article_ID": str}, low_memory=False)
    review_master = ensure_columns(review_master, TEXT_COLS + [
        "Article_ID", "Relevance", "Actor_Mention", "Successor_Frame",
        "Human_Rights_Abuse", "Security_Effectiveness",
        "Geopolitical_Rivalry", "Any_Review_Flag",
        "source_attributed_flag", "likely_republished_flag",
        "chapter52_priority_flag"
    ], fill_value=None)

    review_master["Full_Text"] = review_master.apply(build_full_text, axis=1)

    nlp, model_info = get_spacy_model()

    mode_descriptions = {
        "surface": "surface cleaned tokens",
        "lemma": "lemmatized tokens",
        "custom": "lemmatized + custom normalization",
        "analytic": "lemmatized + custom normalization + analytical grouping",
    }

    all_token_tables = []
    subset_summary_rows = []
    sample_top_rows = []

    for subset_name, subset_fn in SUBSET_DEFINITIONS.items():
        mask = subset_fn(review_master)
        subset_df = review_master[mask].copy()

        subset_summary_rows.append({
            "subset": subset_name,
            "rows": len(subset_df)
        })

        if subset_df.empty:
            continue

        for mode_name in ["surface", "lemma", "custom", "analytic"]:
            counter = token_counter_for_subset(
                subset_df,
                text_col="Full_Text",
                mode=mode_name,
                nlp=nlp
            )

            token_df = counter_to_df(counter, subset_name, mode_name)

            if token_df.empty:
                token_df = pd.DataFrame(columns=[
                    "subset", "mode", "token", "count", "percent_of_tokens"
                ])

            save_csv(
                token_df.head(TOP_N),
                LEXICAL_NORM_DIR / f"top_tokens_{mode_name}_{subset_name}.csv"
            )

            all_token_tables.append(token_df)

            top10 = token_df.head(10).copy()
            if not top10.empty:
                sample_top_rows.append(top10)

            save_wordcloud(
                counter,
                title=f"{subset_name} | {mode_name}",
                filename=f"wordcloud_{mode_name}_{subset_name}.png"
            )

    if all_token_tables:
        all_tokens_df = pd.concat(all_token_tables, axis=0, ignore_index=True)
    else:
        all_tokens_df = pd.DataFrame(columns=[
            "subset", "mode", "token", "count", "percent_of_tokens"
        ])

    save_csv(all_tokens_df, LEXICAL_NORM_DIR / "all_token_profiles.csv")

    subset_summary_df = pd.DataFrame(subset_summary_rows)
    save_csv(subset_summary_df, LEXICAL_NORM_DIR / "subset_row_summary.csv")

    # =========================
    # MODE COMPARISON TABLES
    # =========================
    mode_comparison_rows = []

    for subset_name in subset_summary_df["subset"].tolist():
        for mode_name in ["surface", "lemma", "custom", "analytic"]:
            sub = all_tokens_df[
                (all_tokens_df["subset"] == subset_name) &
                (all_tokens_df["mode"] == mode_name)
            ].copy()

            mode_comparison_rows.append({
                "subset": subset_name,
                "mode": mode_name,
                "unique_tokens": sub["token"].nunique(),
                "token_instances_total": sub["count"].sum(),
                "top_token": safe_str(sub.iloc[0]["token"]) if not sub.empty else "",
                "top_token_count": int(sub.iloc[0]["count"]) if not sub.empty else 0,
            })

    mode_comparison_df = pd.DataFrame(mode_comparison_rows)
    save_csv(mode_comparison_df, LEXICAL_NORM_DIR / "mode_comparison_summary.csv")

    # =========================
    # ANALYTICAL GROUP PROFILE
    # =========================
    analytic_only = all_tokens_df[all_tokens_df["mode"] == "analytic"].copy()
    if not analytic_only.empty:
        analytic_group_profile = (
            analytic_only.groupby(["subset", "token"], dropna=False)["count"]
            .sum()
            .reset_index()
            .sort_values(["subset", "count"], ascending=[True, False])
        )
    else:
        analytic_group_profile = pd.DataFrame(columns=["subset", "token", "count"])

    save_csv(analytic_group_profile, LEXICAL_NORM_DIR / "analytic_group_profile.csv")

    # =========================
    # TOP TOKENS ACROSS MODES (SELECTED SUBSETS)
    # =========================
    selected_subsets = [
        "relevance_4",
        "relevance_3plus",
        "africa_corps_cases",
        "successor_cases",
        "hra_cases",
        "security_effectiveness_cases",
        "geopolitical_rivalry_cases",
        "chapter52_priority",
    ]

    cross_mode_top_rows = []

    for subset_name in selected_subsets:
        for mode_name in ["surface", "lemma", "custom", "analytic"]:
            sub = all_tokens_df[
                (all_tokens_df["subset"] == subset_name) &
                (all_tokens_df["mode"] == mode_name)
            ].copy().head(20)

            if sub.empty:
                continue

            for _, row in sub.iterrows():
                cross_mode_top_rows.append({
                    "subset": subset_name,
                    "mode": mode_name,
                    "token": row["token"],
                    "count": row["count"],
                    "percent_of_tokens": row["percent_of_tokens"]
                })

    cross_mode_top_df = pd.DataFrame(cross_mode_top_rows)
    save_csv(cross_mode_top_df, LEXICAL_NORM_DIR / "cross_mode_top_tokens_selected_subsets.csv")

    # =========================
    # SUMMARY
    # =========================
    total_articles = len(review_master)

    if not mode_comparison_df.empty:
        top_relevance4_analytic = mode_comparison_df[
            (mode_comparison_df["subset"] == "relevance_4") &
            (mode_comparison_df["mode"] == "analytic")
        ]
        if not top_relevance4_analytic.empty:
            top_relevance4_analytic_token = safe_str(top_relevance4_analytic.iloc[0]["top_token"])
        else:
            top_relevance4_analytic_token = "N/A"
    else:
        top_relevance4_analytic_token = "N/A"

    summary_text = f"""
CORPUS5 SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Input:
- review_master rows: {total_articles}

Normalization pipeline:
- spaCy available: {"yes" if SPACY_AVAILABLE else "no"}
- spaCy model used: {model_info}
- token modes generated:
  - surface
  - lemma
  - custom
  - analytic

Outputs:
- top token tables per subset and mode
- all_token_profiles.csv
- subset_row_summary.csv
- mode_comparison_summary.csv
- analytic_group_profile.csv
- cross_mode_top_tokens_selected_subsets.csv
- wordclouds per subset and normalization mode

Selected observation:
- top analytic token/group in relevance_4 subset: {top_relevance4_analytic_token}
""".strip()

    with open(SUMMARY_DIR / "corpus5_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open(SUMMARY_DIR / "corpus5_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    lexical_findings_text = f"""
CORPUS5 NORMALIZATION NOTES
Generated: {datetime.now().isoformat(timespec="seconds")}

1. Surface mode
- Uses cleaned surface tokens after stopword removal.
- Useful as a baseline and for checking visible lexical forms.

2. Lemma mode
- Uses French lemmatization where spaCy is available.
- Intended to reduce inflectional noise.

3. Custom mode
- Uses lemma mode plus custom normalization mapping.
- Intended to merge domain-relevant near-equivalent forms.

4. Analytic mode
- Uses custom-normalized tokens plus broader analytical grouping.
- Intended as a high-level interpretive lexical aid.

Important caution:
- Analytic normalization is not a replacement for close reading.
- It is an exploratory aggregation layer for lexical review and visualization.
""".strip()

    with open(SUMMARY_DIR / "corpus5_normalization_notes.txt", "w", encoding="utf-8") as f:
        f.write(lexical_findings_text)

    print("\n=== CORPUS5 DIAGNOSTICS ===\n")
    print(f"Review master rows: {total_articles}")
    print(f"spaCy available: {'yes' if SPACY_AVAILABLE else 'no'}")
    print(f"spaCy model used: {model_info}")
    print(f"Subset profiles created: {len(subset_summary_df)}")
    print(f"Combined token-profile rows: {len(all_tokens_df)}")
    print("\nSaved outputs to:")
    print(f"- {LEXICAL_NORM_DIR}")
    print(f"- {WORDCLOUD_DIR}")
    print(f"- {SUMMARY_DIR}")


if __name__ == "__main__":
    main()