import re
from pathlib import Path
from datetime import datetime
import pandas as pd

# =========================
# CONFIG
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_REVIEW_MASTER = SCRIPT_DIR / "data" / "review_master.csv"

KWIC_DIR = SCRIPT_DIR / "kwic"
CONCORDANCE_DIR = SCRIPT_DIR / "concordance"
SUMMARY_DIR = SCRIPT_DIR / "summary"

# -------------------------
# KEYWORD GROUPS
# -------------------------
KEYWORD_GROUPS = {
    "core_actor_terms": {
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
        "russian_contractors": [
            r"\bmercenaires?\s+russes?\b",
            r"\bparamilitaires?\s+russes?\b",
            r"\binstructeurs?\s+russes?\b",
            r"\bformateurs?\s+russes?\b",
            r"\bcoop[ée]rants?\s+russes?\b",
        ],
    },

    "labels": {
        "mercenaires": [
            r"\bmercenaires?\b",
            r"\bmercenariat\b",
        ],
        "paramilitaires": [
            r"\bparamilitaires?\b",
        ],
        "instructeurs": [
            r"\binstructeurs?\b",
            r"\bformateurs?\b",
            r"\bconseillers?\b",
            r"\bcoop[ée]rants?\b",
        ],
        "partenaires_allies": [
            r"\bpartenaire\b",
            r"\bpartenaires?\b",
            r"\balli[ée]s?\b",
            r"\bpartenaire\s+historique\b",
            r"\bpartenaire\s+strat[ée]gique\b",
        ],
        "neutral_designations": [
            r"\bsoci[ée]t[ée]\s+de\s+s[ée]curit[ée]\s+priv[ée]e?\b",
            r"\bentreprise\s+militaire\s+priv[ée]e?\b",
            r"\bgroupe\s+paramilitaire\b",
            r"\bpmc\b",
        ],
        "occupation_foreign_force": [
            r"\bforces?\s+[ée]trang[èe]res?\b",
            r"\bforces?\s+d['’]occupation\b",
            r"\boccupants?\b",
            r"\bmilice\b",
            r"\bmilices\b",
        ],
    },

    "security_terms": {
        "terrorisme": [
            r"\bterrorisme\b",
            r"\bterroristes?\b",
            r"\banti-terroriste\b",
            r"\blutte\s+contre\s+le\s+terrorisme\b",
        ],
        "jihadisme": [
            r"\bjihadistes?\b",
            r"\bantijihadiste\b",
            r"\bgsim\b",
            r"\bjnim\b",
            r"\baqmi\b",
            r"\beis\b",
            r"\betat\s+islamique\b",
        ],
        "operational_terms": [
            r"\bopérations?\b",
            r"\bdéploiement\b",
            r"\bconvoi\b",
            r"\bs[ée]curisation\b",
            r"\bdispositif\b",
            r"\binter-forces\b",
        ],
    },

    "sovereignty_terms": {
        "souverainete": [
            r"\bsouverainet[ée]\b",
            r"\brespect\s+de\s+la\souverainet[ée]\b",
            r"\bpays\s+souverain\b",
            r"\betat\s+souverain\b",
        ],
        "autonomy_choice": [
            r"\bchoix\s+de\s+partenaires\b",
            r"\bchoix\s+strat[ée]giques\b",
            r"\bautonomie\s+strat[ée]gique\b",
            r"\bnon-ing[ée]rence\b",
            r"\bdroit\s+de\s+choisir\s+ses\s+partenaires\b",
        ],
    },

    "abuse_terms": {
        "exactions": [
            r"\bexactions?\b",
            r"\bviolations?\s+des?\s+droits\b",
            r"\bviolations?\s+des?\s+droits\s+de\s+l['’]homme\b",
        ],
        "mass_violence": [
            r"\bmassacres?\b",
            r"\bex[ée]cutions?\b",
            r"\btorture\b",
            r"\batrocit[ée]s?\b",
            r"\bcrimes?\b",
        ],
        "civilian_harm": [
            r"\bviolence\s+contre\s+les\s+civils\b",
            r"\babus\s+contre\s+les\s+civils\b",
            r"\bpopulations?\s+civiles?\b",
            r"\bdisparitions?\s+forc[ée]es?\b",
            r"\bd[ée]tentions?\s+arbitraires?\b",
            r"\benl[eè]vements?\b",
        ],
    },

    "anti_neocolonial_terms": {
        "colonial_terms": [
            r"\bcolonial\b",
            r"\bcoloniale?\b",
            r"\bpuissance\s+coloniale\b",
        ],
        "neocolonial_terms": [
            r"\bn[ée]ocolonial\b",
            r"\bn[ée]ocolonialisme\b",
            r"\bimp[ée]rialisme\b",
        ],
        "emancipation_terms": [
            r"\banticolonial\b",
            r"\bfran[çc]afrique\b",
            r"\blib[ée]ration\b",
            r"\b[ée]mancipation\b",
        ],
    },

    "western_failure_terms": {
        "failure_terms": [
            r"\b[ée]chec\b",
            r"\b[ée]chec\s+de\s+la\s+france\b",
            r"\babandon\b",
            r"\bhypocrisie\b",
            r"\binefficace\b",
            r"\bincapable\b",
            r"\bdouble\s+standard\b",
        ],
        "withdrawal_terms": [
            r"\bretrait\s+pr[ée]cipit[ée]\b",
            r"\bincompatible\s+avec\s+sa\s+pr[ée]sence\b",
            r"\bperte\s+de\s+souverainet[ée]\b",
        ],
    },

    "security_effectiveness_terms": {
        "effectiveness_terms": [
            r"\befficacit[ée]\b",
            r"\befficace\b",
            r"\br[ée]sultats?\b",
            r"\bstabiliser\b",
            r"\bstabilisation\b",
            r"\bsucc[èe]s\b",
        ],
        "operational_success_terms": [
            r"\bmont[ée]e\s+en\s+puissance\b",
            r"\bcapacit[ée]s?\s+op[ée]rationnelles?\b",
            r"\bcapacit[ée]\s+de\s+combat\b",
            r"\bmission\s+accomplie\b",
            r"\bsans\s+incident\s+majeur\b",
            r"\bcoordination\s+inter-forces\b",
            r"\bdispositif\s+de\s+s[ée]curisation\b",
            r"\btotalement\s+ma[iî]tris[ée]\b",
            r"\bsucc[èe]s\s+strat[ée]gique\b",
        ],
    },

    "economic_terms": {
        "resources_terms": [
            r"\bmine[s]?\b",
            r"\bminier[s]?\b",
            r"\bsite[s]?\s+miniers?\b",
            r"\bgisements?\b",
            r"\bressources?\s+naturelles?\b",
            r"\bcontrat[s]?\s+miniers?\b",
            r"\bor\b",
            r"\bgold\b",
            r"\bextraction\b",
            r"\bexploitation\b",
        ],
    },

    "mali_state_terms": {
        "fama_terms": [
            r"\bfama\b",
            r"\bforces?\s+arm[ée]es?\s+maliennes\b",
            r"\barmée\s+malienne\b",
        ],
        "state_terms": [
            r"\bautorit[ée]s?\s+maliennes\b",
            r"\bgouvernement\s+malien\b",
            r"\btransition\s+malienne\b",
            r"\bassimi\s+go[iï]ta\b",
            r"\bjunte\b",
            r"\bbamako\b",
        ],
        "mali_places": [
            r"\bkidal\b",
            r"\bgao\b",
            r"\btombouctou\b",
            r"\bm[ée]naka\b",
            r"\bmopti\b",
        ],
    },

    "external_actor_terms": {
        "russia_terms": [
            r"\brussie\b",
            r"\brusse[s]?\b",
            r"\bmoscou\b",
            r"\bkremlin\b",
            r"\bpoutine\b",
            r"\blavrov\b",
        ],
        "france_terms": [
            r"\bfrance\b",
            r"\bfran[çc]ais[e]?[s]?\b",
            r"\bparis\b",
            r"\bbarkhane\b",
            r"\bserval\b",
        ],
        "international_terms": [
            r"\bminusma\b",
            r"\bonu\b",
            r"\bnations\s+unies\b",
            r"\bcedeao\b",
            r"\becowas\b",
            r"\bunion\s+africaine\b",
            r"\boccident\b",
            r"\boccidentaux\b",
            r"\betats?-unis\b",
            r"\bunion\s+europ[ée]enne\b",
        ],
    },

    "geopolitical_terms": {
        "rivalry_terms": [
            r"\bg[ée]opolitique\b",
            r"\brivalit[ée]\b",
            r"\bcomp[ée]tition\b",
            r"\binfluence\b",
            r"\bbras\s+de\s+fer\b",
        ],
        "multipolar_terms": [
            r"\bmonde\s+multipolaire\b",
            r"\bmultipolar(?:it[ée]|ire)\b",
            r"\bd[ée]clin\s+fulgurant\b",
        ],
    },

    "source_attribution_terms": {
        "attribution_terms": [
            r"\bselon\b",
            r"\bd['’]apr[eè]s\b",
            r"\bcit[ée]?\s+par\b",
            r"\ba\s+rapport[ée]?\b",
            r"\ba\s+d[ée]clar[ée]?\b",
        ],
        "republication_terms": [
            r"\brepris\s+de\b",
            r"\breprise\s+de\b",
            r"\brelay[ée]\s+par\b",
            r"\bpubli[ée]\s+par\b",
            r"\bpubli[ée]\s+initialement\s+sur\b",
            r"\bsource\s*:\b",
        ],
        "external_sources": [
            r"\bafp\b",
            r"\breuters\b",
            r"\brfi\b",
            r"\bfrance\s*24\b",
            r"\bjeune\s+afrique\b",
            r"\bbbc\b",
            r"\ble\s+monde\b",
            r"\bsputnik\b",
            r"\btass\b",
        ],
    },
}

META_COLS = [
    "Article_ID",
    "Outlet",
    "Date",
    "Headline",
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
    "source_attributed_flag",
    "likely_republished_flag",
    "chapter52_priority_flag",
]

TEXT_COLS = [
    "Headline",
    "Lead",
    "Body_Postclean",
]

# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def ensure_dirs():
    KWIC_DIR.mkdir(parents=True, exist_ok=True)
    CONCORDANCE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

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

def build_full_text(row):
    parts = [safe_str(row.get(col)) for col in TEXT_COLS]
    parts = [p for p in parts if p]
    return " ".join(parts).strip()

def normalize_spaces(text):
    return re.sub(r"\s+", " ", safe_str(text)).strip()

def extract_kwic_rows(row, keyword_group, keyword_name, patterns, context_window_chars=70):
    full_text = build_full_text(row)
    if not full_text:
        return []

    sentences = split_sentences(full_text)
    rows = []

    for sent_idx, sent in enumerate(sentences, start=1):
        sent_norm = normalize_spaces(sent)

        for pat in patterns:
            for m in re.finditer(pat, sent_norm, flags=re.IGNORECASE):
                start, end = m.span()
                keyword_text = sent_norm[start:end]

                left_context = sent_norm[max(0, start - context_window_chars):start].strip()
                right_context = sent_norm[end:min(len(sent_norm), end + context_window_chars)].strip()

                out = {
                    "keyword_group": keyword_group,
                    "keyword_name": keyword_name,
                    "matched_pattern": pat,
                    "matched_text": keyword_text,
                    "sentence_index": sent_idx,
                    "left_context": left_context,
                    "keyword": keyword_text,
                    "right_context": right_context,
                    "sentence_full": sent_norm,
                }

                for col in META_COLS:
                    out[col] = row.get(col)

                rows.append(out)

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
    review_master = ensure_columns(review_master, META_COLS + TEXT_COLS, fill_value=None)

    all_kwic_rows = []
    keyword_summary_rows = []
    outlet_summary_rows = []
    subset_summary_rows = []
    group_summary_rows = []

    for keyword_group, keyword_items in KEYWORD_GROUPS.items():
        group_rows = []

        for keyword_name, patterns in keyword_items.items():
            rows = []

            for _, row in review_master.iterrows():
                rows.extend(extract_kwic_rows(row, keyword_group, keyword_name, patterns))

            kwic_df = pd.DataFrame(rows)

            if kwic_df.empty:
                kwic_df = pd.DataFrame(columns=[
                    *META_COLS,
                    "keyword_group",
                    "keyword_name",
                    "matched_pattern",
                    "matched_text",
                    "sentence_index",
                    "left_context",
                    "keyword",
                    "right_context",
                    "sentence_full",
                ])

            kwic_path = KWIC_DIR / f"kwic_{keyword_name}.csv"
            save_csv(kwic_df, kwic_path)

            all_kwic_rows.extend(rows)
            group_rows.extend(rows)

            keyword_summary_rows.append({
                "keyword_group": keyword_group,
                "keyword_name": keyword_name,
                "matches_total": len(kwic_df),
                "unique_articles": kwic_df["Article_ID"].nunique() if "Article_ID" in kwic_df.columns else 0,
                "unique_outlets": kwic_df["Outlet"].nunique() if "Outlet" in kwic_df.columns else 0,
            })

            if not kwic_df.empty and "Outlet" in kwic_df.columns:
                outlet_counts = (
                    kwic_df.groupby("Outlet", dropna=False)
                    .size()
                    .reset_index(name="matches_total")
                )
                outlet_counts["keyword_group"] = keyword_group
                outlet_counts["keyword_name"] = keyword_name
                outlet_summary_rows.append(outlet_counts)

            if not kwic_df.empty:
                subset_summary_rows.append(pd.DataFrame([
                    {
                        "keyword_group": keyword_group,
                        "keyword_name": keyword_name,
                        "subset": "relevance_3plus",
                        "matches_total": kwic_df[kwic_df["Relevance"].fillna(0).astype(int).isin([3, 4])].shape[0],
                        "unique_articles": kwic_df[kwic_df["Relevance"].fillna(0).astype(int).isin([3, 4])]["Article_ID"].nunique()
                    },
                    {
                        "keyword_group": keyword_group,
                        "keyword_name": keyword_name,
                        "subset": "relevance_4",
                        "matches_total": kwic_df[kwic_df["Relevance"].fillna(0).astype(int) == 4].shape[0],
                        "unique_articles": kwic_df[kwic_df["Relevance"].fillna(0).astype(int) == 4]["Article_ID"].nunique()
                    },
                    {
                        "keyword_group": keyword_group,
                        "keyword_name": keyword_name,
                        "subset": "africa_corps_cases",
                        "matches_total": kwic_df[kwic_df["Actor_Mention"].fillna(0).astype(int).isin([2, 3])].shape[0],
                        "unique_articles": kwic_df[kwic_df["Actor_Mention"].fillna(0).astype(int).isin([2, 3])]["Article_ID"].nunique()
                    },
                    {
                        "keyword_group": keyword_group,
                        "keyword_name": keyword_name,
                        "subset": "successor_cases",
                        "matches_total": kwic_df[kwic_df["Successor_Frame"].fillna(0).astype(int) == 1].shape[0],
                        "unique_articles": kwic_df[kwic_df["Successor_Frame"].fillna(0).astype(int) == 1]["Article_ID"].nunique()
                    },
                    {
                        "keyword_group": keyword_group,
                        "keyword_name": keyword_name,
                        "subset": "review_flagged",
                        "matches_total": kwic_df[kwic_df["Any_Review_Flag"].fillna(0).astype(int) == 1].shape[0],
                        "unique_articles": kwic_df[kwic_df["Any_Review_Flag"].fillna(0).astype(int) == 1]["Article_ID"].nunique()
                    },
                    {
                        "keyword_group": keyword_group,
                        "keyword_name": keyword_name,
                        "subset": "source_attributed",
                        "matches_total": kwic_df[kwic_df["source_attributed_flag"].fillna(0).astype(int) == 1].shape[0],
                        "unique_articles": kwic_df[kwic_df["source_attributed_flag"].fillna(0).astype(int) == 1]["Article_ID"].nunique()
                    },
                ]))

        group_df = pd.DataFrame(group_rows)
        if group_df.empty:
            group_df = pd.DataFrame(columns=[
                *META_COLS,
                "keyword_group",
                "keyword_name",
                "matched_pattern",
                "matched_text",
                "sentence_index",
                "left_context",
                "keyword",
                "right_context",
                "sentence_full",
            ])

        save_csv(group_df, KWIC_DIR / f"kwic_group_{keyword_group}.csv")

        group_summary_rows.append({
            "keyword_group": keyword_group,
            "matches_total": len(group_df),
            "unique_articles": group_df["Article_ID"].nunique() if "Article_ID" in group_df.columns else 0,
            "unique_outlets": group_df["Outlet"].nunique() if "Outlet" in group_df.columns else 0,
            "keywords_in_group": len(keyword_items)
        })

    all_kwic_df = pd.DataFrame(all_kwic_rows)
    if all_kwic_df.empty:
        all_kwic_df = pd.DataFrame(columns=[
            *META_COLS,
            "keyword_group",
            "keyword_name",
            "matched_pattern",
            "matched_text",
            "sentence_index",
            "left_context",
            "keyword",
            "right_context",
            "sentence_full",
        ])

    save_csv(all_kwic_df, CONCORDANCE_DIR / "kwic_all_keywords.csv")

    keyword_summary_df = pd.DataFrame(keyword_summary_rows)
    save_csv(keyword_summary_df, CONCORDANCE_DIR / "kwic_keyword_summary.csv")

    group_summary_df = pd.DataFrame(group_summary_rows)
    save_csv(group_summary_df, CONCORDANCE_DIR / "kwic_group_summary.csv")

    if outlet_summary_rows:
        keyword_outlet_profile = pd.concat(outlet_summary_rows, axis=0, ignore_index=True)
    else:
        keyword_outlet_profile = pd.DataFrame(columns=["Outlet", "matches_total", "keyword_group", "keyword_name"])
    save_csv(keyword_outlet_profile, CONCORDANCE_DIR / "keyword_outlet_profile.csv")

    if subset_summary_rows:
        keyword_subset_profile = pd.concat(subset_summary_rows, axis=0, ignore_index=True)
    else:
        keyword_subset_profile = pd.DataFrame(columns=[
            "keyword_group", "keyword_name", "subset", "matches_total", "unique_articles"
        ])
    save_csv(keyword_subset_profile, CONCORDANCE_DIR / "keyword_subset_profile.csv")

    total_matches = len(all_kwic_df)
    total_unique_articles = all_kwic_df["Article_ID"].nunique() if "Article_ID" in all_kwic_df.columns else 0

    if not keyword_summary_df.empty:
        top_keyword_row = keyword_summary_df.sort_values("matches_total", ascending=False).iloc[0]
        top_keyword_name = safe_str(top_keyword_row["keyword_name"])
        top_keyword_group = safe_str(top_keyword_row["keyword_group"])
        top_keyword_matches = int(top_keyword_row["matches_total"])
    else:
        top_keyword_name = "N/A"
        top_keyword_group = "N/A"
        top_keyword_matches = 0

    if not group_summary_df.empty:
        top_group_row = group_summary_df.sort_values("matches_total", ascending=False).iloc[0]
        top_group_name = safe_str(top_group_row["keyword_group"])
        top_group_matches = int(top_group_row["matches_total"])
    else:
        top_group_name = "N/A"
        top_group_matches = 0

    summary_text = f"""
CORPUS3 SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Input:
- review_master.csv rows: {len(review_master)}

KWIC outputs:
- keyword groups processed: {len(KEYWORD_GROUPS)}
- individual keyword sets processed: {len(keyword_summary_df)}
- total KWIC matches across all keywords: {total_matches}
- unique articles represented in KWIC outputs: {total_unique_articles}
- top keyword by matches: {top_keyword_name} (group: {top_keyword_group}, matches: {top_keyword_matches})
- top keyword group by matches: {top_group_name} ({top_group_matches} matches)

Output files:
- kwic/kwic_<keyword>.csv
- kwic/kwic_group_<group>.csv
- concordance/kwic_all_keywords.csv
- concordance/kwic_keyword_summary.csv
- concordance/kwic_group_summary.csv
- concordance/keyword_outlet_profile.csv
- concordance/keyword_subset_profile.csv
""".strip()

    with open(SUMMARY_DIR / "corpus3_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    with open(SUMMARY_DIR / "corpus3_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n=== CORPUS3 DIAGNOSTICS ===\n")
    print(f"Input review_master rows: {len(review_master)}")
    print(f"Keyword groups processed: {len(KEYWORD_GROUPS)}")
    print(f"Individual keyword sets processed: {len(keyword_summary_df)}")
    print(f"Total KWIC matches: {total_matches}")
    print(f"Unique articles with at least one KWIC hit: {total_unique_articles}")
    print(f"Top keyword: {top_keyword_name} ({top_keyword_matches} matches)")
    print(f"Top keyword group: {top_group_name} ({top_group_matches} matches)")
    print("\nSaved outputs to:")
    print(f"- {KWIC_DIR}")
    print(f"- {CONCORDANCE_DIR}")
    print(f"- {SUMMARY_DIR}")


if __name__ == "__main__":
    main()