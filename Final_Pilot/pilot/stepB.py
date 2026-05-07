import os
import re
import itertools
import pandas as pd
from rapidfuzz import fuzz

# =========================
# CONFIG
# =========================
INPUT_STEP1_XLSX = "data/postStep1.xlsx"
INPUT_STEP2_XLSX = "data/postStep2.xlsx"

OUTPUT_XLSX = "data/postStepB.xlsx"
OUTPUT_CSV = "data/postStepB.csv"
OUTPUT_DUP_ANALYSIS_XLSX = "data/postStepB_duplicate_analysis.xlsx"

# Relaxed similarity thresholds
HEADLINE_STRONG_THRESHOLD = 85
BODY_STRONG_THRESHOLD = 82
FULLTEXT_STRONG_THRESHOLD = 84

HEADLINE_MODERATE_THRESHOLD = 70
BODY_MODERATE_THRESHOLD = 70
FULLTEXT_MODERATE_THRESHOLD = 72

MIN_BODY_LEN_FOR_DUP = 150

# =========================
# OUTLET REFERENCE MAP
# =========================
MALIAN_MEDIA_PATTERNS = {
    "malijet.com": [
        r"\bmalijet\b",
        r"\bmalijet\.com\b",
    ],
    "maliweb.net": [
        r"\bmaliweb\b",
        r"\bmaliweb\.net\b",
    ],
    "bamada.net": [
        r"\bbamada\b",
        r"\bbamada\.net\b",
    ],
    "mali24.info": [
        r"\bmali\s*24\b",
        r"\bmali24\b",
        r"\bmali24\.info\b",
    ],
    "studiotamani.org": [
        r"\bstudio\s+tamani\b",
        r"\bstudiotamani\b",
        r"\bstudiotamani\.org\b",
    ],
    "journaldumali.com": [
        r"\bjournal\s+du\s+mali\b",
        r"\bjournaldumali\b",
        r"\bjournaldumali\.com\b",
    ],
    "malitribune.com": [
        r"\bmali\s+tribune\b",
        r"\bmalitribune\b",
        r"\bmalitribune\.com\b",
    ],
    "info-matin.ml": [
        r"\binfo[\s\-]?matin\b",
        r"\binfo[\-]?matin\.ml\b",
    ],
    "lejalon.com": [
        r"\ble\s+jalon\b",
        r"\blejalon\b",
        r"\blejalon\.com\b",
    ],
    "lessor.ml": [
        r"\bl['’]essor\b",
        r"\blessor\b",
        r"\blessor\.ml\b",
    ],
    "malikonews.com": [
        r"\bmaliko\s+news\b",
        r"\bmalikonews\b",
        r"\bmalikonews\.com\b",
    ],
}

EXTERNAL_SOURCE_PATTERNS = {
    "AFP": {
        "type": "news_agency",
        "patterns": [
            r"\bafp\b",
            r"\bavec\s+afp\b",
            r"\bselon\s+l['’]?afp\b",
            r"\bsource\s*:\s*afp\b",
            r"\bpubli[ée]\s+par\s+l['’]?afp\b",
            r"\bpar\s+afp\b",
            r"\br[ée]daction\s*/\s*afp\b",
        ]
    },
    "Reuters": {
        "type": "news_agency",
        "patterns": [
            r"\breuters\b",
            r"\bavec\s+reuters\b",
            r"\bselon\s+reuters\b",
            r"\bsource\s*:\s*reuters\b",
            r"\bpar\s+reuters\b",
        ]
    },
    "AP": {
        "type": "news_agency",
        "patterns": [
            r"\bassociated\s+press\b",
            r"\bap\b",
        ]
    },
    "Anadolu": {
        "type": "news_agency",
        "patterns": [
            r"\banadolu\b",
            r"\banadolu\s+agency\b",
        ]
    },
    "RFI": {
        "type": "foreign_media",
        "patterns": [
            r"\brfi\b",
            r"\bradio\s+france\s+internationale\b",
        ]
    },
    "France24": {
        "type": "foreign_media",
        "patterns": [
            r"\bfrance\s*24\b",
        ]
    },
    "Jeune Afrique": {
        "type": "foreign_media",
        "patterns": [
            r"\bjeune\s+afrique\b",
        ]
    },
    "BBC": {
        "type": "foreign_media",
        "patterns": [
            r"\bbbc\b",
            r"\bbbc\s+news\b",
        ]
    },
    "Le Monde": {
        "type": "foreign_media",
        "patterns": [
            r"\ble\s+monde\b",
        ]
    },
    "TV5Monde": {
        "type": "foreign_media",
        "patterns": [
            r"\btv5monde\b",
            r"\btv5\s+monde\b",
        ]
    },
    "Sputnik": {
        "type": "foreign_media",
        "patterns": [
            r"\bsputnik\b",
        ]
    },
    "TASS": {
        "type": "news_agency",
        "patterns": [
            r"\btass\b",
        ]
    },
}

# Strong republication markers
REPUBLICATION_PATTERNS = [
    r"\brepris\s+de\b",
    r"\breprise\s+de\b",
    r"\brelay[ée]\s+par\b",
    r"\bpubli[ée]\s+initialement\s+sur\b",
    r"\bpubli[ée]\s+par\b",
    r"\bsource\s*:",
    r"\bextrait\s+de\b",
]

# General attribution markers
ATTRIBUTION_PATTERNS = [
    r"\bselon\b",
    r"\bd['’]apr[eè]s\b",
    r"\ba\s+rapport[ée]?\b",
    r"\ba\s+indiqu[ée]?\b",
    r"\ba\s+d[ée]clar[ée]?\b",
    r"\binterview\s+au\b",
    r"\bdans\s+une\s+interview\s+[àa]\b",
    r"\bcit[ée]?\s+par\b",
    r"\bavec\b",
]

# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

def normalize_article_id(x):
    if pd.isna(x) or x is None:
        return None
    digits = re.sub(r"\D", "", str(x))
    if not digits:
        return None
    return digits.zfill(6)

def collapse_ws(text):
    text = safe_str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lowercase_text(text):
    return collapse_ws(text).lower()

def text_for_detection(row):
    parts = [
        safe_str(row.get("headline_clean", "")),
        safe_str(row.get("lead_clean", "")),
        safe_str(row.get("body_postclean", "")),
        safe_str(row.get("author_clean", "")),
        safe_str(row.get("rubrique_clean", "")),
        safe_str(row.get("url", "")),
    ]
    return "\n".join([p for p in parts if p]).strip()

def count_matches(text, patterns):
    total = 0
    matched_patterns = []
    if not text:
        return total, matched_patterns
    for pat in patterns:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            total += len(found)
            matched_patterns.append(pat)
    return total, matched_patterns

def find_named_pattern_hits(text, pattern_map):
    hits = []
    if not text:
        return hits
    for name, meta in pattern_map.items():
        patterns = meta["patterns"] if isinstance(meta, dict) else meta
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(name)
                break
    return sorted(set(hits))

def classify_external_source_type(names):
    if not names:
        return "none"
    types = []
    for name in names:
        if name in EXTERNAL_SOURCE_PATTERNS:
            types.append(EXTERNAL_SOURCE_PATTERNS[name]["type"])
    types = sorted(set(types))
    if not types:
        return "other_external"
    if len(types) == 1:
        return types[0]
    return "mixed_external"

def detect_explicit_external_sources(row):
    text = text_for_detection(row)
    names = find_named_pattern_hits(text, EXTERNAL_SOURCE_PATTERNS)

    return {
        "explicit_external_source_flag": 1 if names else 0,
        "explicit_external_source_type": classify_external_source_type(names),
        "explicit_external_source_name": "; ".join(names) if names else "",
        "explicit_external_source_count": len(names),
        "explicit_external_source_note": "explicit external source markers found" if names else ""
    }

def detect_author_external_source(row):
    author = safe_str(row.get("author_clean", ""))
    names = find_named_pattern_hits(author, EXTERNAL_SOURCE_PATTERNS)

    return {
        "author_external_source_flag": 1 if names else 0,
        "author_external_source_name": "; ".join(names) if names else "",
        "author_external_source_type": classify_external_source_type(names),
        "author_external_source_note": "external source detected in author field" if names else ""
    }

def detect_malian_media_references(row):
    text = text_for_detection(row)
    current_outlet = safe_str(row.get("outlet", ""))

    names = []
    for outlet_name, patterns in MALIAN_MEDIA_PATTERNS.items():
        if outlet_name == current_outlet:
            continue
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                names.append(outlet_name)
                break

    names = sorted(set(names))

    ref_type = "none"
    note = ""
    if names:
        low = lowercase_text(text)
        if re.search(r"\brepris\s+de\b|\breprise\s+de\b|\bpubli[ée]\s+sur\b|\bpubli[ée]\s+par\b", low):
            ref_type = "reprinted"
        elif re.search(r"\bselon\b|\bd['’]apr[eè]s\b|\binterview\s+au\b", low):
            ref_type = "quoted"
        else:
            ref_type = "mentioned"
        note = "explicit Malian media reference detected"

    return {
        "explicit_malian_media_reference_flag": 1 if names else 0,
        "explicit_malian_media_reference_names": "; ".join(names) if names else "",
        "explicit_malian_media_reference_count": len(names),
        "explicit_malian_media_reference_type": ref_type,
        "explicit_malian_media_reference_note": note
    }

def detect_republication_phrases(row):
    text = text_for_detection(row)
    n, pats = count_matches(text, REPUBLICATION_PATTERNS)

    return {
        "republication_phrase_flag": 1 if n > 0 else 0,
        "republication_phrase_count": n,
        "republication_phrase_note": "; ".join(sorted(set(pats))) if pats else ""
    }

def detect_attribution_phrases(row):
    text = text_for_detection(row)
    n, pats = count_matches(text, ATTRIBUTION_PATTERNS)

    return {
        "attribution_phrase_flag": 1 if n > 0 else 0,
        "attribution_phrase_count": n,
        "attribution_phrase_note": "; ".join(sorted(set(pats))) if pats else ""
    }

def normalize_for_similarity(text):
    text = lowercase_text(text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\sàâçéèêëîïôûùüÿñæœ'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_similarity_texts(df):
    df["headline_sim_text"] = df["headline_clean"].apply(normalize_for_similarity)
    df["lead_sim_text"] = df["lead_clean"].apply(normalize_for_similarity)
    df["body_sim_text"] = df["body_postclean"].apply(normalize_for_similarity)
    df["full_sim_text"] = (
        df["headline_clean"].apply(safe_str) + " " +
        df["lead_clean"].apply(safe_str) + " " +
        df["body_postclean"].apply(safe_str)
    ).apply(normalize_for_similarity)
    return df

def headline_score(a, b):
    return fuzz.token_sort_ratio(a, b)

def body_score(a, b):
    return fuzz.token_set_ratio(a, b)

def fulltext_score(a, b):
    return fuzz.token_set_ratio(a, b)

def candidate_duplicate_relation(row1, row2):
    if row1["outlet"] != row2["outlet"]:
        return "cross_outlet_possible_republication"
    return "same_outlet_possible_duplicate"

def duplicate_strength_label(hs, bs, fs):
    if hs >= HEADLINE_STRONG_THRESHOLD and bs >= BODY_STRONG_THRESHOLD and fs >= FULLTEXT_STRONG_THRESHOLD:
        return "strong"
    if hs >= HEADLINE_MODERATE_THRESHOLD and bs >= BODY_MODERATE_THRESHOLD and fs >= FULLTEXT_MODERATE_THRESHOLD:
        return "moderate"
    return "weak"

def should_compare(row1, row2):
    if row1["article_id"] == row2["article_id"]:
        return False

    if len(safe_str(row1.get("body_sim_text", ""))) < MIN_BODY_LEN_FOR_DUP and len(safe_str(row2.get("body_sim_text", ""))) < MIN_BODY_LEN_FOR_DUP:
        return False

    h1 = safe_str(row1.get("headline_sim_text", ""))
    h2 = safe_str(row2.get("headline_sim_text", ""))

    if h1 and h2:
        shared_head_terms = set(h1.split()) & set(h2.split())
        if len(shared_head_terms) >= 2:
            return True

    if (
        row1.get("explicit_external_source_flag", 0) == 1 or
        row2.get("explicit_external_source_flag", 0) == 1 or
        row1.get("author_external_source_flag", 0) == 1 or
        row2.get("author_external_source_flag", 0) == 1 or
        row1.get("explicit_malian_media_reference_flag", 0) == 1 or
        row2.get("explicit_malian_media_reference_flag", 0) == 1
    ):
        return True

    return True

def find_duplicate_pairs(df):
    pairs = []
    rows = df.to_dict("records")

    for row1, row2 in itertools.combinations(rows, 2):
        if not should_compare(row1, row2):
            continue

        hs = headline_score(row1["headline_sim_text"], row2["headline_sim_text"])
        bs = body_score(row1["body_sim_text"], row2["body_sim_text"])
        fs = fulltext_score(row1["full_sim_text"], row2["full_sim_text"])

        if hs >= HEADLINE_MODERATE_THRESHOLD and bs >= BODY_MODERATE_THRESHOLD and fs >= FULLTEXT_MODERATE_THRESHOLD:
            pairs.append({
                "article_id_1": row1["article_id"],
                "outlet_1": row1["outlet"],
                "date_1": row1.get("date_iso_full"),
                "article_id_2": row2["article_id"],
                "outlet_2": row2["outlet"],
                "date_2": row2.get("date_iso_full"),
                "same_outlet_flag": 1 if row1["outlet"] == row2["outlet"] else 0,
                "cross_outlet_flag": 1 if row1["outlet"] != row2["outlet"] else 0,
                "headline_score": hs,
                "body_score": bs,
                "fulltext_score": fs,
                "duplicate_strength": duplicate_strength_label(hs, bs, fs),
                "candidate_republication_relation": candidate_duplicate_relation(row1, row2),
            })

    return pd.DataFrame(pairs)

def build_duplicate_clusters(df_pairs):
    if df_pairs.empty:
        return {}, pd.DataFrame(columns=[
            "cluster_id", "cluster_size", "outlet_count",
            "cross_outlet_flag", "member_article_ids",
            "member_outlets", "representative_article_id"
        ])

    graph = {}
    for _, row in df_pairs.iterrows():
        a = row["article_id_1"]
        b = row["article_id_2"]
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    visited = set()
    clusters = []
    cluster_map = {}

    cluster_idx = 1
    for node in graph:
        if node in visited:
            continue

        stack = [node]
        members = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            members.append(cur)
            for nbr in graph.get(cur, []):
                if nbr not in visited:
                    stack.append(nbr)

        cluster_id = f"ND_{cluster_idx:04d}"
        for m in members:
            cluster_map[m] = cluster_id
        clusters.append((cluster_id, sorted(members)))
        cluster_idx += 1

    cluster_rows = []
    for cluster_id, members in clusters:
        df_sub = df_pairs[
            (df_pairs["article_id_1"].isin(members)) |
            (df_pairs["article_id_2"].isin(members))
        ]

        outlets = set(df_sub["outlet_1"].dropna().tolist()) | set(df_sub["outlet_2"].dropna().tolist())
        cross_outlet_flag = 1 if len(outlets) > 1 else 0

        cluster_rows.append({
            "cluster_id": cluster_id,
            "cluster_size": len(members),
            "outlet_count": len(outlets),
            "cross_outlet_flag": cross_outlet_flag,
            "member_article_ids": "; ".join(members),
            "member_outlets": "; ".join(sorted(outlets)),
            "representative_article_id": members[0]
        })

    df_clusters = pd.DataFrame(cluster_rows)
    return cluster_map, df_clusters

def build_cluster_members(df, cluster_map):
    rows = []
    for _, row in df.iterrows():
        aid = row["article_id"]
        if aid in cluster_map:
            rows.append({
                "cluster_id": cluster_map[aid],
                "article_id": aid,
                "outlet": row.get("outlet"),
                "date": row.get("date_iso_full"),
                "headline": row.get("headline_clean"),
            })
    return pd.DataFrame(rows)

def build_article_level_duplicate_features(df, df_pairs, cluster_map):
    df["near_duplicate_flag"] = 0
    df["near_duplicate_cluster_id"] = ""
    df["near_duplicate_match_count"] = 0
    df["near_duplicate_cross_outlet_flag"] = 0
    df["near_duplicate_same_outlet_flag"] = 0
    df["near_duplicate_top_match_id"] = ""
    df["near_duplicate_top_match_outlet"] = ""
    df["near_duplicate_top_match_score"] = 0
    df["near_duplicate_top_match_headline_score"] = 0
    df["near_duplicate_top_match_body_score"] = 0

    if df_pairs.empty:
        return df

    by_id = {}
    for _, row in df_pairs.iterrows():
        for src, tgt, out, hs, bs, fs, cross_flag, same_flag in [
            (
                row["article_id_1"], row["article_id_2"], row["outlet_2"],
                row["headline_score"], row["body_score"], row["fulltext_score"],
                row["cross_outlet_flag"], row["same_outlet_flag"]
            ),
            (
                row["article_id_2"], row["article_id_1"], row["outlet_1"],
                row["headline_score"], row["body_score"], row["fulltext_score"],
                row["cross_outlet_flag"], row["same_outlet_flag"]
            )
        ]:
            by_id.setdefault(src, []).append({
                "match_id": tgt,
                "match_outlet": out,
                "headline_score": hs,
                "body_score": bs,
                "fulltext_score": fs,
                "cross_outlet_flag": cross_flag,
                "same_outlet_flag": same_flag
            })

    for idx, row in df.iterrows():
        aid = row["article_id"]
        matches = by_id.get(aid, [])
        if not matches:
            continue

        matches_sorted = sorted(matches, key=lambda x: x["fulltext_score"], reverse=True)
        top = matches_sorted[0]

        df.at[idx, "near_duplicate_flag"] = 1
        df.at[idx, "near_duplicate_cluster_id"] = cluster_map.get(aid, "")
        df.at[idx, "near_duplicate_match_count"] = len(matches)
        df.at[idx, "near_duplicate_cross_outlet_flag"] = 1 if any(m["cross_outlet_flag"] == 1 for m in matches) else 0
        df.at[idx, "near_duplicate_same_outlet_flag"] = 1 if any(m["same_outlet_flag"] == 1 for m in matches) else 0
        df.at[idx, "near_duplicate_top_match_id"] = top["match_id"]
        df.at[idx, "near_duplicate_top_match_outlet"] = top["match_outlet"]
        df.at[idx, "near_duplicate_top_match_score"] = top["fulltext_score"]
        df.at[idx, "near_duplicate_top_match_headline_score"] = top["headline_score"]
        df.at[idx, "near_duplicate_top_match_body_score"] = top["body_score"]

    return df

def compute_source_attributed_flag(row):
    explicit_external = int(row.get("explicit_external_source_flag", 0) or 0)
    author_external = int(row.get("author_external_source_flag", 0) or 0)
    explicit_malian = int(row.get("explicit_malian_media_reference_flag", 0) or 0)

    if explicit_external == 1 or author_external == 1 or explicit_malian == 1:
        return 1

    return 0

def compute_republication_index(row):
    explicit_external = int(row.get("explicit_external_source_flag", 0) or 0)
    author_external = int(row.get("author_external_source_flag", 0) or 0)
    explicit_malian = int(row.get("explicit_malian_media_reference_flag", 0) or 0)
    republication_phrase = int(row.get("republication_phrase_flag", 0) or 0)
    attribution_phrase = int(row.get("attribution_phrase_flag", 0) or 0)
    near_dup = int(row.get("near_duplicate_flag", 0) or 0)
    cross_outlet = int(row.get("near_duplicate_cross_outlet_flag", 0) or 0)
    top_score = row.get("near_duplicate_top_match_score", 0) or 0

    if author_external == 1:
        return 3, "author_external_source", "high"

    if explicit_external == 1:
        return 3, "explicit_external_source", "high"

    if explicit_malian == 1 and republication_phrase == 1:
        return 3, "explicit_malian_media_reference+republication_phrase", "high"

    if near_dup == 1 and cross_outlet == 1 and top_score >= FULLTEXT_STRONG_THRESHOLD:
        return 3, "near_duplicate_cross_outlet_strong", "high"

    if explicit_malian == 1 and attribution_phrase == 1:
        return 2, "explicit_malian_reference+attribution", "medium"

    if near_dup == 1 and cross_outlet == 1:
        return 2, "near_duplicate_cross_outlet", "medium"

    if explicit_malian == 1 or republication_phrase == 1 or attribution_phrase == 1 or near_dup == 1:
        return 1, "weak_or_partial_signal", "low"

    return 0, "none", "none"

def apply_relevance_filter(df, mode):
    mode = safe_str(mode).lower()

    if mode == "all":
        return df.copy()

    if not os.path.exists(INPUT_STEP2_XLSX):
        raise FileNotFoundError(f"Relevance filter requires file: {INPUT_STEP2_XLSX}")

    df2 = pd.read_excel(INPUT_STEP2_XLSX, dtype={"article_id": str})
    if "article_id" not in df2.columns or "relevance_code" not in df2.columns:
        raise ValueError("postStep2.xlsx must contain 'article_id' and 'relevance_code' columns")

    df2["article_id"] = df2["article_id"].apply(normalize_article_id)

    if mode == "2+":
        keep_ids = set(df2[df2["relevance_code"] >= 2]["article_id"].dropna().tolist())
    elif mode == "3+":
        keep_ids = set(df2[df2["relevance_code"] >= 3]["article_id"].dropna().tolist())
    elif mode == "4":
        keep_ids = set(df2[df2["relevance_code"] == 4]["article_id"].dropna().tolist())
    else:
        raise ValueError("Invalid relevance filter mode. Use: all, 2+, 3+, or 4")

    return df[df["article_id"].isin(keep_ids)].copy()

def choose_relevance_mode():
    print("\nChoose relevance filter mode for stepB:")
    print("  1 = all articles")
    print("  2 = Relevance 2+")
    print("  3 = Relevance 3+")
    print("  4 = Relevance 4 only")

    mapping = {
        "1": "all",
        "2": "2+",
        "3": "3+",
        "4": "4",
        "all": "all",
        "2+": "2+",
        "3+": "3+",
        "4": "4",
    }

    while True:
        choice = input("Enter choice [1/2/3/4]: ").strip().lower()
        if choice in mapping:
            return mapping[choice]
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

# =========================
# MAIN
# =========================
def main():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(INPUT_STEP1_XLSX):
        raise FileNotFoundError(f"Missing input: {INPUT_STEP1_XLSX}")

    relevance_mode = choose_relevance_mode()

    df = pd.read_excel(INPUT_STEP1_XLSX, dtype={"article_id": str})
    df["article_id"] = df["article_id"].apply(normalize_article_id)

    expected_cols = [
        "article_id", "outlet", "url", "date_iso_full",
        "headline_clean", "lead_clean", "body_postclean",
        "author_clean", "rubrique_clean"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df = apply_relevance_filter(df, relevance_mode)
    df["stepB_relevance_filter_mode"] = relevance_mode

    # Explicit detections
    ext_df = df.apply(detect_explicit_external_sources, axis=1, result_type="expand")
    author_ext_df = df.apply(detect_author_external_source, axis=1, result_type="expand")
    malian_df = df.apply(detect_malian_media_references, axis=1, result_type="expand")
    repub_df = df.apply(detect_republication_phrases, axis=1, result_type="expand")
    attr_df = df.apply(detect_attribution_phrases, axis=1, result_type="expand")

    df = pd.concat([df, ext_df, author_ext_df, malian_df, repub_df, attr_df], axis=1)

    # Similarity texts
    df = make_similarity_texts(df)

    # Duplicate pairs
    df_pairs = find_duplicate_pairs(df)

    # Duplicate clusters
    cluster_map, df_clusters = build_duplicate_clusters(df_pairs)
    df_cluster_members = build_cluster_members(df, cluster_map)

    # Article-level duplicate features
    df = build_article_level_duplicate_features(df, df_pairs, cluster_map)

    # Synthesis
    df["source_attributed_flag"] = df.apply(compute_source_attributed_flag, axis=1)

    repub_index_data = df.apply(compute_republication_index, axis=1)
    df["republication_index"] = repub_index_data.apply(lambda x: x[0])
    df["likely_republished_basis"] = repub_index_data.apply(lambda x: x[1])
    df["republication_confidence"] = repub_index_data.apply(lambda x: x[2])

    # stricter likely republished flag: only strong evidence
    df["likely_republished_flag"] = df["republication_index"].apply(lambda x: 1 if x >= 3 else 0)

    # Export main output
    df.to_excel(OUTPUT_XLSX, index=False)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Export duplicate analysis workbook
    with pd.ExcelWriter(OUTPUT_DUP_ANALYSIS_XLSX, engine="openpyxl") as writer:
        df_pairs.to_excel(writer, sheet_name="duplicate_pairs", index=False)
        df_clusters.to_excel(writer, sheet_name="duplicate_clusters", index=False)
        df_cluster_members.to_excel(writer, sheet_name="cluster_members", index=False)

    # Summary
    print(f"\nRelevance filter mode: {relevance_mode}")
    print(f"Loaded articles after filter: {len(df)}")
    print(f"Explicit external source hits: {int(df['explicit_external_source_flag'].sum())}")
    print(f"Author-based external source hits: {int(df['author_external_source_flag'].sum())}")
    print(f"Explicit Malian media reference hits: {int(df['explicit_malian_media_reference_flag'].sum())}")
    print(f"Source-attributed articles: {int(df['source_attributed_flag'].sum())}")
    print(f"Near-duplicate flagged articles: {int(df['near_duplicate_flag'].sum())}")
    print(f"Likely republished articles (index >= 3): {int(df['likely_republished_flag'].sum())}")

    print(f"\nSaved article-level output to {OUTPUT_XLSX}")
    print(f"Saved article-level output to {OUTPUT_CSV}")
    print(f"Saved duplicate analysis workbook to {OUTPUT_DUP_ANALYSIS_XLSX}")

if __name__ == "__main__":
    main()