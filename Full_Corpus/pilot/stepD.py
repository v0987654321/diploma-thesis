import os
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
INPUT_CONSOLIDATED = "data/postConsolidated.csv"
INPUT_CONSERVATIVE = "GEMINI/data/final_conservative_adjudicated_table.csv"

OUTPUT_FLAGS = "data/postStepD_prigozhin_mutiny_article_flags.csv"
OUTPUT_MONTHLY_LONG = "data/postStepD_prigozhin_mutiny_monthly_long.csv"
OUTPUT_MONTHLY_WIDE = "data/postStepD_prigozhin_mutiny_monthly_wide.csv"
OUTPUT_SUMMARY = "data/postStepD_prigozhin_mutiny_summary.txt"

FIGURE_DIR = "figures"

START_MONTH = "2021-07"
END_MONTH = "2025-12"

# The main graph uses strict mutiny-event detection.
# A broader Prigozhin-any flag is still exported for sensitivity checks.
GRAPH_FLAG_COL = "Mutiny_Strict_Flag"

ACTOR_GROUPS = [
    "Wagner_only",
    "AC_only",
    "Both_mentioned",
]

ACTOR_GROUP_COLORS = {
    "Wagner_only": "#0b3c5d",
    "AC_only": "#3288bd",
    "Both_mentioned": "#d95f02",
}


# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def safe_int(x, default=None):
    if pd.isna(x) or x is None:
        return default
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            if float(x).is_integer():
                return int(x)
            return default
        s = str(x).strip()
        if not s:
            return default
        f = float(s)
        if f.is_integer():
            return int(f)
        return default
    except Exception:
        return default


def normalize_article_id(x):
    raw = safe_str(x)
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return digits.zfill(6)


def normalize_text(text):
    text = safe_str(text).lower()
    text = text.replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text):
    text = safe_str(text)
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", text) if s.strip()]


def has_any(text_norm, patterns):
    if not text_norm:
        return False
    for pat in patterns:
        if re.search(pat, text_norm, flags=re.IGNORECASE):
            return True
    return False


def matched_patterns(text_norm, patterns):
    found = []
    if not text_norm:
        return found
    for pat in patterns:
        if re.search(pat, text_norm, flags=re.IGNORECASE):
            found.append(pat)
    return found


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


def build_full_text(row):
    parts = [
        safe_str(row.get("Headline")),
        safe_str(row.get("Lead")),
        safe_str(row.get("Body_Postclean")),
    ]
    return " ".join([p for p in parts if p]).strip()


# =========================
# PRIGOZHIN / MUTINY LEXICONS
# =========================

PRIGOZHIN_PATTERNS = [
    r"\bprigojine\b",
    r"\bprigozhin\b",
    r"\bprigojin\b",
    r"\bevgueni\s+prigo",
    r"\bye[v]?geny\s+prigo",
]

WAGNER_PATTERNS = [
    r"\bwagner\b",
    r"\bgroupe\s+wagner\b",
    r"\bwagner\s+group\b",
]

RUSSIA_CONTEXT_PATTERNS = [
    r"\brussie\b",
    r"\brusse[s]?\b",
    r"\bmoscou\b",
    r"\bkremlin\b",
    r"\bpoutine\b",
    r"\brostov\b",
    r"\brostov-sur-le-don\b",
]

# Strong event terms that are relatively specific to the June 2023 event
# when combined with Wagner / Prigozhin / Russian context.
STRONG_MUTINY_EVENT_PATTERNS = [
    r"\bmutinerie\b",
    r"\bmutin[s]?\b",
    r"\bmarche\s+sur\s+moscou\b",
    r"\bmarche\s+vers\s+moscou\b",
    r"\bprise\s+de\s+rostov\b",
    r"\brostov-sur-le-don\b",
    r"\brostov\b",
    r"\b24\s+juin\s+2023\b",
    r"\bjuin\s+2023\b",
    r"\brebellion\s+de\s+wagner\b",
    r"\bwagner.{0,60}\brebellion\b",
    r"\brebellion.{0,60}\bwagner\b",
    r"\bprigo(?:jine|zhin|jin).{0,80}\brebellion\b",
    r"\brebellion.{0,80}\bprigo(?:jine|zhin|jin)\b",
]

# Broader uprising/coup vocabulary.
# These terms are only counted when they occur with Prigozhin/Moscow/Rostov
# or with Wagner + Russian context, to avoid false positives from Malian coups.
GENERIC_UPRISING_PATTERNS = [
    r"\brebellion\b",
    r"\brevolte\b",
    r"\bsoulevement\b",
    r"\binsurrection\b",
    r"\bputsch\b",
    r"\btentative\s+de\s+coup\b",
    r"\bcoup\s+d[' ]etat\b",
    r"\btentative\s+de\s+putsch\b",
    r"\btrahison\b",
]

AFTERMATH_PATTERNS = [
    r"\bapres\s+la\s+mutinerie\b",
    r"\bapres\s+sa\s+mutinerie\b",
    r"\bau\s+lendemain\s+de\s+la\s+mutinerie\b",
    r"\bdepuis\s+la\s+mutinerie\b",
    r"\bsuite\s+a\s+la\s+mutinerie\b",
    r"\bapres\s+la\s+rebellion\b",
    r"\bsuite\s+a\s+la\s+rebellion\b",
]


# =========================
# DETECTION LOGIC
# =========================
def detect_prigozhin_mutiny(row):
    """
    Returns strict and broad flags.

    Strict flag:
    - article contains a plausible reference to the June 2023 Prigozhin/Wagner mutiny,
      not merely any Prigozhin mention.

    Broad Prigozhin flag:
    - article mentions Prigozhin in a WG/Russia context.
      Exported for sensitivity, not used for main figure by default.
    """
    full_text = build_full_text(row)
    full_norm = normalize_text(full_text)
    sentences = split_sentences(full_text)

    article_has_prigo = has_any(full_norm, PRIGOZHIN_PATTERNS)
    article_has_wagner = has_any(full_norm, WAGNER_PATTERNS)
    article_has_russia_context = has_any(full_norm, RUSSIA_CONTEXT_PATTERNS)
    article_has_strong_event = has_any(full_norm, STRONG_MUTINY_EVENT_PATTERNS)
    article_has_generic_event = has_any(full_norm, GENERIC_UPRISING_PATTERNS)
    article_has_aftermath = has_any(full_norm, AFTERMATH_PATTERNS)

    broad_prigozhin_flag = 1 if (
        article_has_prigo and (article_has_wagner or article_has_russia_context)
    ) else 0

    score = 0
    evidence_sentences = []
    all_matches = []

    # Sentence-level detection is the main guard against false positives.
    for sent in sentences:
        sent_norm = normalize_text(sent)

        sent_has_prigo = has_any(sent_norm, PRIGOZHIN_PATTERNS)
        sent_has_wagner = has_any(sent_norm, WAGNER_PATTERNS)
        sent_has_russia_context = has_any(sent_norm, RUSSIA_CONTEXT_PATTERNS)
        sent_has_strong_event = has_any(sent_norm, STRONG_MUTINY_EVENT_PATTERNS)
        sent_has_generic_event = has_any(sent_norm, GENERIC_UPRISING_PATTERNS)
        sent_has_aftermath = has_any(sent_norm, AFTERMATH_PATTERNS)

        sent_matches = []
        sent_matches.extend(matched_patterns(sent_norm, PRIGOZHIN_PATTERNS))
        sent_matches.extend(matched_patterns(sent_norm, STRONG_MUTINY_EVENT_PATTERNS))
        sent_matches.extend(matched_patterns(sent_norm, GENERIC_UPRISING_PATTERNS))
        sent_matches.extend(matched_patterns(sent_norm, AFTERMATH_PATTERNS))

        # Strong event with relevant context
        if sent_has_strong_event and (sent_has_prigo or sent_has_wagner or sent_has_russia_context):
            score += 3
            evidence_sentences.append(sent)
            all_matches.extend(sent_matches)
            continue

        # Generic uprising term: require tighter context
        if sent_has_generic_event and (
            sent_has_prigo or
            "moscou" in sent_norm or
            "rostov" in sent_norm or
            (sent_has_wagner and sent_has_russia_context)
        ):
            score += 2
            evidence_sentences.append(sent)
            all_matches.extend(sent_matches)
            continue

        # Aftermath phrases tied to Prigozhin/Wagner/Russia
        if sent_has_aftermath and (sent_has_prigo or sent_has_wagner or sent_has_russia_context):
            score += 2
            evidence_sentences.append(sent)
            all_matches.extend(sent_matches)
            continue

    # Article-level fallback:
    # if Prigozhin + Wagner + event terms occur in the same article,
    # accept as strict even if sentence splitting missed proximity.
    if score < 2:
        if article_has_prigo and article_has_wagner and (article_has_strong_event or article_has_aftermath):
            score = max(score, 2)
            all_matches.extend(matched_patterns(full_norm, PRIGOZHIN_PATTERNS))
            all_matches.extend(matched_patterns(full_norm, STRONG_MUTINY_EVENT_PATTERNS))
            all_matches.extend(matched_patterns(full_norm, AFTERMATH_PATTERNS))
            evidence_sentences = evidence_sentences[:3]

        elif article_has_prigo and article_has_russia_context and article_has_strong_event:
            score = max(score, 2)
            all_matches.extend(matched_patterns(full_norm, PRIGOZHIN_PATTERNS))
            all_matches.extend(matched_patterns(full_norm, STRONG_MUTINY_EVENT_PATTERNS))
            evidence_sentences = evidence_sentences[:3]

    strict_flag = 1 if score >= 2 else 0

    return {
        "Prigozhin_Any_Flag": broad_prigozhin_flag,
        "Mutiny_Strict_Flag": strict_flag,
        "Mutiny_Detection_Score": score,
        "Mutiny_Matched_Patterns": "; ".join(sorted(set(all_matches))),
        "Mutiny_Evidence_Sentences": " || ".join(evidence_sentences[:3]),
    }


# =========================
# FINAL CODING HARMONIZATION
# =========================
def derive_final_relevance(row):
    adj = safe_int(row.get("Adj_V04_Relevance"), None)
    if adj is not None:
        return adj
    return safe_int(row.get("Relevance"), None)


def derive_final_actor_mention(row):
    adj = safe_int(row.get("Adj_V05_Actor_Mention"), None)
    if adj is not None:
        return adj
    return safe_int(row.get("Actor_Mention"), None)


def actor_subset_from_code(code):
    code = safe_int(code, None)
    if code == 1:
        return "Wagner_only"
    if code == 2:
        return "AC_only"
    if code == 3:
        return "Both_mentioned"
    if code == 4:
        return "Indirect_only"
    if code == 5:
        return "Cannot_determine"
    return "Missing"


def derive_month(row):
    year = safe_int(row.get("Date_Year"), None)
    month = safe_int(row.get("Date_Month"), None)

    if year is not None and month is not None and 1 <= month <= 12:
        return f"{year:04d}-{month:02d}"

    date = safe_str(row.get("Date"))
    if date:
        dt = pd.to_datetime(date, errors="coerce")
        if pd.notna(dt):
            return dt.to_period("M").strftime("%Y-%m")

    return ""


# =========================
# LOAD + MERGE
# =========================
def load_inputs():
    if not os.path.exists(INPUT_CONSOLIDATED):
        raise FileNotFoundError(f"Missing required input: {INPUT_CONSOLIDATED}")

    if not os.path.exists(INPUT_CONSERVATIVE):
        raise FileNotFoundError(
            f"Missing required conservative adjudicated table: {INPUT_CONSERVATIVE}"
        )

    df_cons = pd.read_csv(INPUT_CONSOLIDATED, dtype={"Article_ID": str}, low_memory=False)
    df_adj = pd.read_csv(INPUT_CONSERVATIVE, dtype={"Article_ID": str}, low_memory=False)

    df_cons["Article_ID"] = df_cons["Article_ID"].apply(normalize_article_id)
    df_adj["Article_ID"] = df_adj["Article_ID"].apply(normalize_article_id)

    keep_adj_cols = [
        "Article_ID",
        "Adj_V04_Relevance",
        "Adj_V05_Actor_Mention",
        "Adj_V06_Successor_Frame",
        "Adj_V07_Dominant_Label",
        "Adj_V08_Stance",
        "Adj_V09_Dominant_Location",
        "Adj_V10_Ambivalence",
        "Adj_V11_Legitimation",
        "Adj_V12_Counterterrorism",
        "Adj_V13_Sovereignty",
        "Adj_V14_Human_Rights_Abuse",
        "Adj_V15_Anti_or_Neocolonialism",
        "Adj_V16_Western_Failure",
        "Adj_V17_Security_Effectiveness",
        "Adj_V18_Economic_Interests",
        "Adj_V19_Geopolitical_Rivalry",
        "Adj_V20_Main_Associated_Actor",
        "Adj_V21_Dominant_Discourse",
        "Manual_Check_Required",
        "Manual_Check_Reasons",
        "LLM_Review_Note",
        "Pro_Review_Candidate",
        "Pro_Review_Reason",
    ]
    keep_adj_cols = [c for c in keep_adj_cols if c in df_adj.columns]

    df = df_cons.merge(df_adj[keep_adj_cols], on="Article_ID", how="left")

    df["Final_Relevance"] = df.apply(derive_final_relevance, axis=1)
    df["Final_Actor_Mention"] = df.apply(derive_final_actor_mention, axis=1)
    df["Actor_Subset_D"] = df["Final_Actor_Mention"].apply(actor_subset_from_code)
    df["Month"] = df.apply(derive_month, axis=1)

    return df


# =========================
# MONTHLY AGGREGATION
# =========================
def build_monthly_tables(df_flags):
    month_range = pd.period_range(START_MONTH, END_MONTH, freq="M").astype(str).tolist()

    all_long_rows = []

    relevance_filters = {
        "relevance2plus": [2, 3, 4],
        "relevance3plus": [3, 4],
        "relevance4": [4],
    }

    for filter_name, rel_codes in relevance_filters.items():
        df_sub = df_flags[
            df_flags["Final_Relevance"].fillna(0).astype(int).isin(rel_codes) &
            df_flags["Actor_Subset_D"].isin(ACTOR_GROUPS) &
            df_flags["Month"].isin(month_range)
        ].copy()

        grouped = (
            df_sub
            .groupby(["Month", "Actor_Subset_D"], dropna=False)
            .agg(
                article_n=("Article_ID", "count"),
                mutiny_strict_n=("Mutiny_Strict_Flag", "sum"),
                prigozhin_any_n=("Prigozhin_Any_Flag", "sum"),
            )
            .reset_index()
        )

        full_index = pd.MultiIndex.from_product(
            [month_range, ACTOR_GROUPS],
            names=["Month", "Actor_Subset_D"]
        )
        grouped = (
            grouped
            .set_index(["Month", "Actor_Subset_D"])
            .reindex(full_index)
            .reset_index()
        )

        grouped["article_n"] = grouped["article_n"].fillna(0).astype(int)
        grouped["mutiny_strict_n"] = grouped["mutiny_strict_n"].fillna(0).astype(int)
        grouped["prigozhin_any_n"] = grouped["prigozhin_any_n"].fillna(0).astype(int)

        grouped["mutiny_strict_percent"] = grouped.apply(
            lambda row: (row["mutiny_strict_n"] / row["article_n"] * 100)
            if row["article_n"] > 0 else None,
            axis=1
        )

        grouped["prigozhin_any_percent"] = grouped.apply(
            lambda row: (row["prigozhin_any_n"] / row["article_n"] * 100)
            if row["article_n"] > 0 else None,
            axis=1
        )

        grouped["Relevance_Filter"] = filter_name

        all_long_rows.append(grouped)

    monthly_long = pd.concat(all_long_rows, axis=0, ignore_index=True)

    monthly_wide = monthly_long.pivot_table(
        index=["Relevance_Filter", "Month"],
        columns="Actor_Subset_D",
        values=[
            "article_n",
            "mutiny_strict_n",
            "mutiny_strict_percent",
            "prigozhin_any_n",
            "prigozhin_any_percent",
        ],
        aggfunc="first"
    )

    monthly_wide.columns = [
        f"{metric}_{actor_group}" for metric, actor_group in monthly_wide.columns
    ]
    monthly_wide = monthly_wide.reset_index()

    return monthly_long, monthly_wide


# =========================
# FIGURES
# =========================
def plot_timeline(monthly_long, filter_name):
    df_plot = monthly_long[monthly_long["Relevance_Filter"] == filter_name].copy()

    if GRAPH_FLAG_COL == "Mutiny_Strict_Flag":
        percent_col = "mutiny_strict_percent"
        n_col = "mutiny_strict_n"
        title_flag = "Prigozhin/Wagner mutiny-related articles"
    else:
        percent_col = "prigozhin_any_percent"
        n_col = "prigozhin_any_n"
        title_flag = "Prigozhin-mentioned articles"

    df_plot["Month_Date"] = pd.to_datetime(df_plot["Month"] + "-01", errors="coerce")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    ax1, ax2 = axes

    # Top panel: percent lines
    for actor_group in ACTOR_GROUPS:
        sub = df_plot[df_plot["Actor_Subset_D"] == actor_group].copy()
        ax1.plot(
            sub["Month_Date"],
            sub[percent_col],
            marker="o",
            linewidth=2,
            markersize=4,
            label=actor_group,
            color=ACTOR_GROUP_COLORS.get(actor_group)
        )

    ax1.axvline(
        pd.to_datetime("2023-06-01"),
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8
    )
    ax1.text(
        pd.to_datetime("2023-06-15"),
        95,
        "June 2023\nPrigozhin mutiny",
        fontsize=9,
        va="top"
    )

    ax1.set_title(
        f"{title_flag}, monthly percent within actor subset ({filter_name})"
    )
    ax1.set_ylabel("Percent within actor subset")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25)
    ax1.legend(title="Actor subset", loc="upper right")

    # Bottom panel: denominator counts
    bottom = None
    month_dates = sorted(df_plot["Month_Date"].dropna().unique())

    for actor_group in ACTOR_GROUPS:
        sub = (
            df_plot[df_plot["Actor_Subset_D"] == actor_group]
            .set_index("Month_Date")
            .reindex(month_dates)
        )
        counts = sub["article_n"].fillna(0).values

        if bottom is None:
            ax2.bar(
                month_dates,
                counts,
                width=20,
                label=actor_group,
                color=ACTOR_GROUP_COLORS.get(actor_group),
                alpha=0.75
            )
            bottom = counts
        else:
            ax2.bar(
                month_dates,
                counts,
                bottom=bottom,
                width=20,
                label=actor_group,
                color=ACTOR_GROUP_COLORS.get(actor_group),
                alpha=0.75
            )
            bottom = bottom + counts

    ax2.axvline(
        pd.to_datetime("2023-06-01"),
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8
    )

    ax2.set_ylabel("Article count\n(denominator)")
    ax2.set_xlabel("Month")
    ax2.grid(True, axis="y", alpha=0.25)

    # X ticks every 3 months
    tick_months = pd.date_range(
        start=pd.to_datetime(START_MONTH + "-01"),
        end=pd.to_datetime(END_MONTH + "-01"),
        freq="3MS"
    )
    ax2.set_xticks(tick_months)
    ax2.set_xticklabels([d.strftime("%Y-%m") for d in tick_months], rotation=45, ha="right")

    plt.tight_layout()

    out_path = Path(FIGURE_DIR) / f"stepD_prigozhin_mutiny_timeline_{filter_name}.png"
    plt.savefig(out_path, dpi=250)
    plt.close()

    return out_path


# =========================
# SUMMARY
# =========================
def build_summary_text(df_flags, monthly_long):
    rel2plus = df_flags[df_flags["Final_Relevance"].fillna(0).astype(int).isin([2, 3, 4])].copy()
    rel3plus = df_flags[df_flags["Final_Relevance"].fillna(0).astype(int).isin([3, 4])].copy()
    rel4 = df_flags[df_flags["Final_Relevance"].fillna(0).astype(int) == 4].copy()

    def subset_summary(df, label):
        df_actor = df[df["Actor_Subset_D"].isin(ACTOR_GROUPS)].copy()
        rows = [f"{label}:"]
        rows.append(f"  - actor-coded rows included: {len(df_actor)}")
        for actor_group in ACTOR_GROUPS:
            sub = df_actor[df_actor["Actor_Subset_D"] == actor_group]
            n = len(sub)
            strict_n = int(sub["Mutiny_Strict_Flag"].sum()) if n else 0
            prigo_n = int(sub["Prigozhin_Any_Flag"].sum()) if n else 0
            rows.append(
                f"  - {actor_group}: n={n}, strict_mutiny={strict_n} "
                f"({strict_n / n * 100 if n else 0:.2f}%), "
                f"prigozhin_any={prigo_n} ({prigo_n / n * 100 if n else 0:.2f}%)"
            )
        return "\n".join(rows)

    text = f"""
STEPD PRIGOZHIN MUTINY TIMELINE SUMMARY
Generated: {datetime.now().isoformat(timespec="seconds")}

Purpose:
- Identify monthly presence of articles related to the June 2023 Prigozhin/Wagner mutiny.
- Calculate percentages within three actor subsets:
  1. Wagner_only
  2. AC_only
  3. Both_mentioned

Inputs:
- Consolidated corpus: {INPUT_CONSOLIDATED}
- Preferred coding source: {INPUT_CONSERVATIVE}

Detection logic:
- Main figure uses Mutiny_Strict_Flag.
- Strict flag requires Prigozhin/Wagner/Russia context plus mutiny/rebellion/Rostov/Moscow/June 2023 event language.
- Prigozhin_Any_Flag is exported only as a broader sensitivity indicator.

Caution:
- Monthly percentages for AC_only and Both_mentioned may be unstable in months with small denominators.
- The figure therefore includes a lower denominator-count panel.

Corpus summaries:
{subset_summary(rel2plus, "Relevance 2+")}
{subset_summary(rel3plus, "Relevance 3+")}
{subset_summary(rel4, "Relevance 4")}

Outputs:
- {OUTPUT_FLAGS}
- {OUTPUT_MONTHLY_LONG}
- {OUTPUT_MONTHLY_WIDE}
- figures/stepD_prigozhin_mutiny_timeline_relevance2plus.png
- figures/stepD_prigozhin_mutiny_timeline_relevance3plus.png
- figures/stepD_prigozhin_mutiny_timeline_relevance4.png
""".strip()

    return text


# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    print("\nStepD: loading inputs...")
    df = load_inputs()

    print(f"Rows after merge: {len(df)}")
    print("Running Prigozhin/Wagner mutiny detection...")

    detection_df = df.apply(detect_prigozhin_mutiny, axis=1, result_type="expand")
    df_flags = pd.concat([df, detection_df], axis=1)

    # Export article-level flags
    flag_cols = [
        "Article_ID",
        "Outlet",
        "Date",
        "Date_Year",
        "Date_Month",
        "Month",
        "Headline",
        "Final_Relevance",
        "Final_Actor_Mention",
        "Actor_Subset_D",
        "Prigozhin_Any_Flag",
        "Mutiny_Strict_Flag",
        "Mutiny_Detection_Score",
        "Mutiny_Matched_Patterns",
        "Mutiny_Evidence_Sentences",
        "URL",
    ]
    flag_cols = [c for c in flag_cols if c in df_flags.columns]
    df_flags[flag_cols].to_csv(OUTPUT_FLAGS, index=False, encoding="utf-8-sig")

    print("Building monthly tables...")
    monthly_long, monthly_wide = build_monthly_tables(df_flags)

    monthly_long.to_csv(OUTPUT_MONTHLY_LONG, index=False, encoding="utf-8-sig")
    monthly_wide.to_csv(OUTPUT_MONTHLY_WIDE, index=False, encoding="utf-8-sig")

    print("Creating figures...")
    fig_paths = []
    for filter_name in ["relevance2plus", "relevance3plus", "relevance4"]:
        fig_paths.append(plot_timeline(monthly_long, filter_name))

    summary_text = build_summary_text(df_flags, monthly_long)
    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Console diagnostics
    print("\nStepD diagnostics:")
    print(f"Total rows: {len(df_flags)}")
    print(f"Rows with Final_Relevance 2+: {df_flags['Final_Relevance'].fillna(0).astype(int).isin([2,3,4]).sum()}")
    print(f"Rows in actor groups 1/2/3: {df_flags['Actor_Subset_D'].isin(ACTOR_GROUPS).sum()}")
    print(f"Strict mutiny flagged rows: {int(df_flags['Mutiny_Strict_Flag'].sum())}")
    print(f"Prigozhin-any flagged rows: {int(df_flags['Prigozhin_Any_Flag'].sum())}")

    print("\nActor subset strict mutiny counts, Relevance 2+:")
    rel2 = df_flags[
        df_flags["Final_Relevance"].fillna(0).astype(int).isin([2, 3, 4]) &
        df_flags["Actor_Subset_D"].isin(ACTOR_GROUPS)
    ].copy()

    for actor_group in ACTOR_GROUPS:
        sub = rel2[rel2["Actor_Subset_D"] == actor_group]
        n = len(sub)
        m = int(sub["Mutiny_Strict_Flag"].sum()) if n else 0
        pct = m / n * 100 if n else 0
        print(f"  {actor_group}: {m}/{n} = {pct:.2f}%")

    print("\nSaved outputs:")
    print(f"- {OUTPUT_FLAGS}")
    print(f"- {OUTPUT_MONTHLY_LONG}")
    print(f"- {OUTPUT_MONTHLY_WIDE}")
    print(f"- {OUTPUT_SUMMARY}")
    for p in fig_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()