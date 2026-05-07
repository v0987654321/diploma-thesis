import pandas as pd
import re
import unicodedata
from dateparser import parse as parse_date

# =========================
# 1. CONFIG
# =========================
INPUT_PATH = "data/pilot.xlsx"
OUTPUT_CSV = "data/postStep1.csv"

OUTLET_MAP = {
    "01": "malijet.com",
    "02": "maliweb.net",
    "03": "bamada.net",
    "04": "mali24.info",
    "05": "studiotamani.org",
    "06": "journaldumali.com",
    "07": "malitribune.com",
    "08": "info-matin.ml",
    "09": "lejalon.com",
    "10": "lessor.ml",
    "11": "malikonews.com",
}

COLUMN_MAP = {
    "Rubrique": "rubrique",
    "Lead": "lead",
    "number comments": "comments_n",
}

# =========================
# 2. BASIC HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

def normalize_unicode(text):
    text = safe_str(text)
    if not text:
        return None
    return unicodedata.normalize("NFKC", text)

def collapse_whitespace(text):
    text = safe_str(text)
    if not text:
        return None
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = text.strip()
    return text if text else None

def strip_html_residues(text):
    text = safe_str(text)
    if not text:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    return text

# =========================
# 3. GENERIC CLEANING
# =========================
def remove_generic_boilerplate(text):
    text = safe_str(text)
    if not text:
        return None

    patterns = [
        r"Partager sur Facebook.*?$",
        r"Partager sur X.*?$",
        r"Partager\s*:\s*.*?$",
        r"Articles similaires.*?$",
        r"En savoir plus.*?$",
        r"Subscribe to get the latest posts sent to your email.*?$",
        r"Saisissez votre adresse e-mail.*?$",
        r"Abonnez-vous.*?$",
        r"Voir toutes les publications.*?$",
        r"Auteur/Autrice.*?$",
        r"https?://\S+\.mp3",
        r"Télécharger.*?$",
        r"Boîte de commentaires Facebook.*?$",
        r"pic\.twitter\.com/\S+",
    ]

    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    return text

def clean_text_basic(text):
    if pd.isna(text) or text is None:
        return None

    text = normalize_unicode(text)
    text = strip_html_residues(text)
    text = remove_generic_boilerplate(text)
    text = collapse_whitespace(text)

    return text if text else None

# =========================
# 4. DATE PARSING WITH PRECISION
# =========================
MONTH_ONLY_PATTERN = re.compile(r"^\s*(\d{4})[-/](\d{1,2})\s*$")

def parse_date_with_precision(date_raw):
    if pd.isna(date_raw) or not safe_str(date_raw).strip():
        return {
            "date_iso_full": None,
            "date_year": None,
            "date_month": None,
            "date_day": None,
            "date_precision": "unknown"
        }

    raw = safe_str(date_raw).strip()

    m = MONTH_ONLY_PATTERN.match(raw)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        return {
            "date_iso_full": None,
            "date_year": year,
            "date_month": month,
            "date_day": None,
            "date_precision": "month"
        }

    dt = parse_date(raw, languages=["fr", "en"])
    if dt is not None:
        return {
            "date_iso_full": dt.date().isoformat(),
            "date_year": dt.year,
            "date_month": dt.month,
            "date_day": dt.day,
            "date_precision": "day"
        }

    return {
        "date_iso_full": None,
        "date_year": None,
        "date_month": None,
        "date_day": None,
        "date_precision": "unknown"
    }

# =========================
# 5. ARTICLE ID / OUTLET PARSING
# =========================
def clean_article_id(article_id):
    if pd.isna(article_id) or article_id is None:
        return None

    raw = safe_str(article_id).strip()
    digits = re.sub(r"\D", "", raw)

    if not digits:
        return None

    return digits.zfill(6)

def derive_outlet_code(article_id):
    article_id = safe_str(article_id)
    if len(article_id) < 2:
        return None
    return article_id[:2]

def derive_article_seq(article_id):
    article_id = safe_str(article_id)
    if len(article_id) < 6:
        return None
    return article_id[2:]

# =========================
# 6. LEAD / BODY DEDUPLICATION
# =========================
def normalize_compare(text):
    text = safe_str(text)
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def split_into_sentences(text):
    text = safe_str(text)
    if not text:
        return []

    text = collapse_whitespace(text)
    if not text:
        return []

    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

def first_n_sentences(text, n=3):
    sents = split_into_sentences(text)
    if not sents:
        return None
    return collapse_whitespace(" ".join(sents[:n]))

def body_starts_with_lead(body, lead, tolerance=250):
    body = safe_str(body)
    lead = safe_str(lead)

    if not body or not lead:
        return False

    lead_cmp = collapse_whitespace(lead)
    if not lead_cmp:
        return False

    prefix_len = max(tolerance, len(lead_cmp) + 50)
    body_prefix = body[:prefix_len]
    body_cmp = collapse_whitespace(body_prefix)

    if not body_cmp:
        return False

    lead_cmp = re.split(r"\[\.\.\.\]|\.\.\.", lead_cmp)[0].strip()

    lead_cmp = normalize_compare(lead_cmp)
    body_cmp = normalize_compare(body_cmp)

    if not lead_cmp:
        return False

    return body_cmp.startswith(lead_cmp)

def remove_prefix_from_body(body, prefix_text):
    body = safe_str(body)
    prefix_text = safe_str(prefix_text)

    if not body:
        return None
    if not prefix_text:
        return collapse_whitespace(body)

    body_sents = split_into_sentences(body)
    prefix_sents = split_into_sentences(prefix_text)

    if not body_sents:
        return None
    if not prefix_sents:
        return collapse_whitespace(body)

    n = len(prefix_sents)

    body_prefix = collapse_whitespace(" ".join(body_sents[:n]))
    prefix_norm = normalize_compare(prefix_text)
    body_prefix_norm = normalize_compare(body_prefix)

    if body_prefix_norm == prefix_norm:
        remaining = body_sents[n:]
        if not remaining:
            return None
        return collapse_whitespace(" ".join(remaining))

    return collapse_whitespace(body)

def repair_lead_and_body(lead, body):
    lead = safe_str(lead)
    body = safe_str(body)

    if not lead and not body:
        return None, None
    if not lead:
        return None, collapse_whitespace(body)
    if not body:
        return collapse_whitespace(lead), None

    lead = collapse_whitespace(lead)
    body = collapse_whitespace(body)

    if not lead and not body:
        return None, None
    if not lead:
        return None, body
    if not body:
        return lead, None

    if body_starts_with_lead(body, lead):
        lead_short = first_n_sentences(lead, 3)
        body_new = remove_prefix_from_body(body, lead_short)
        return lead_short if lead_short else lead, body_new if body_new else None

    return lead, body

# =========================
# 7. OUTLET-SPECIFIC POST-CLEANING
# =========================
def cleanup_studiotamani(text):
    text = safe_str(text)
    if not text:
        return None

    patterns = [
        r"Koulikoro, écoutez les programmes du Studio Tamani.*?$",
        r"https?://www\.studiotamani\.org/\S+\.mp3",
        r"Télécharger.*?$",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    return collapse_whitespace(text)

def cleanup_malikonews(text):
    text = safe_str(text)
    if not text:
        return None

    patterns = [
        r"Avec AFP.*?$",
        r"Auteur/Autrice.*?$",
        r"Voir toutes les publications.*?$",
        r"Maliko News.*?$",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    return collapse_whitespace(text)

def cleanup_lejalon(text):
    text = safe_str(text)
    if not text:
        return None

    patterns = [
        r"@CourPenaleInt.*?$",
        r"@FRANCE24.*?$",
        r"@RFI.*?$",
        r"@BCCI.*?$",
        r"pic\.twitter\.com/\S+",
        r"September \d{1,2}, \d{4}.*?$",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    return collapse_whitespace(text)

def cleanup_mali24(text):
    text = safe_str(text)
    if not text:
        return None

    patterns = [
        r"Boîte de commentaires Facebook.*?$",
        r"Partager\s*:\s*.*?$",
        r"Articles similaires.*?$",
        r"En savoir plus.*?$",
        r"Subscribe to get the latest posts sent to your email.*?$",
        r"Saisissez votre adresse e-mail.*?$",
        r"Abonnez-vous.*?$",
        r"Par\s+[A-ZÉÈÊËÀÂÎÏÔÖÙÛÜÇa-zéèêëàâîïôöùûüç\-\s]+l[’']essor.*?$",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    return collapse_whitespace(text)

def cleanup_lessor(text):
    text = safe_str(text)
    if not text:
        return None

    patterns = [
        r"Le Mali retire sa reconnaissance.*?$",
        r"Massa Makan Diabaté.*?$",
        r"Ce grand orchestre a fait la gloire.*?$",
        r"La salle de conférence du gouvernorat.*?$",
        r"La somme de 500 Fcfa.*?$",
        r"Le ministre des Maliens établis.*?$",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    return collapse_whitespace(text)

def cleanup_info_matin(text):
    text = safe_str(text)
    if not text:
        return None

    patterns = [
        r"PAR\s+[A-ZÉÈÊËÀÂÎÏÔÖÙÛÜÇa-zéèêëàâîïôöùûüç\-\s]+$",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    return collapse_whitespace(text)

def cleanup_by_outlet(text, outlet):
    text = safe_str(text)
    outlet = safe_str(outlet)

    if not text:
        return None

    text = collapse_whitespace(text)

    if outlet == "studiotamani.org":
        text = cleanup_studiotamani(text)
    elif outlet == "malikonews.com":
        text = cleanup_malikonews(text)
    elif outlet == "lejalon.com":
        text = cleanup_lejalon(text)
    elif outlet == "mali24.info":
        text = cleanup_mali24(text)
    elif outlet == "lessor.ml":
        text = cleanup_lessor(text)
    elif outlet == "info-matin.ml":
        text = cleanup_info_matin(text)

    return collapse_whitespace(text)

# =========================
# 8. FULL TEXT BUILDERS
# =========================
def join_nonempty_parts(parts, sep=" "):
    cleaned = []
    for p in parts:
        p = collapse_whitespace(p)
        if p:
            cleaned.append(p)
    if not cleaned:
        return None
    return sep.join(cleaned)

def build_full_text(headline, lead, body_dedup):
    return join_nonempty_parts([headline, lead, body_dedup], sep=" ")

def build_postclean_fulltext(row):
    return join_nonempty_parts(
        [row.get("headline_clean"), row.get("lead_clean"), row.get("body_postclean")],
        sep=" "
    )

# =========================
# 9. MAIN
# =========================
def main():
    df = pd.read_excel(INPUT_PATH, dtype={"article_id": str})
    df.columns = [safe_str(col).strip() for col in df.columns]

    df = df.rename(columns=COLUMN_MAP)

    expected_cols = [
        "article_id", "url", "headline", "date_raw",
        "author", "rubrique", "lead", "comments_n", "body"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    df["article_id"] = df["article_id"].apply(clean_article_id)
    df["outlet_code"] = df["article_id"].apply(derive_outlet_code)
    df["article_seq"] = df["article_id"].apply(derive_article_seq)
    df["outlet"] = df["outlet_code"].map(OUTLET_MAP)

    date_info = df["date_raw"].apply(parse_date_with_precision)
    df["date_iso_full"] = date_info.apply(lambda x: x["date_iso_full"])
    df["date_year"] = date_info.apply(lambda x: x["date_year"])
    df["date_month"] = date_info.apply(lambda x: x["date_month"])
    df["date_day"] = date_info.apply(lambda x: x["date_day"])
    df["date_precision"] = date_info.apply(lambda x: x["date_precision"])

    for col in ["headline", "lead", "body", "author", "rubrique"]:
        clean_col = f"{col}_clean"
        df[clean_col] = df[col].apply(clean_text_basic)

    repaired = df.apply(
        lambda row: repair_lead_and_body(row.get("lead_clean"), row.get("body_clean")),
        axis=1
    )
    df["lead_clean"] = repaired.apply(lambda x: x[0] if isinstance(x, tuple) and len(x) == 2 else None)
    df["body_dedup"] = repaired.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) == 2 else None)

    df["full_text"] = df.apply(
        lambda row: build_full_text(
            row.get("headline_clean"),
            row.get("lead_clean"),
            row.get("body_dedup")
        ),
        axis=1
    )
    df["full_text_clean"] = df["full_text"].apply(collapse_whitespace)
    df["full_text_lower"] = df["full_text_clean"].apply(lambda x: safe_str(x).lower() if pd.notna(x) else None)

    df["body_postclean"] = df.apply(
        lambda row: cleanup_by_outlet(row.get("body_dedup"), row.get("outlet")),
        axis=1
    )

    df["full_text_postclean"] = df.apply(build_postclean_fulltext, axis=1)
    df["full_text_postclean_lower"] = df["full_text_postclean"].apply(
        lambda x: safe_str(x).lower() if pd.notna(x) else None
    )

    # diagnostics
    print("\nBasic diagnostics:")
    print(f"Rows total: {len(df)}")
    print(f"Missing article_id: {df['article_id'].isna().sum()}")
    print(f"Missing headline_clean: {df['headline_clean'].isna().sum()}")
    print(f"Missing lead_clean: {df['lead_clean'].isna().sum()}")
    print(f"Missing body_clean: {df['body_clean'].isna().sum()}")
    print(f"Missing body_dedup: {df['body_dedup'].isna().sum()}")
    print(f"Missing body_postclean: {df['body_postclean'].isna().sum()}")
    print(f"Missing full_text_postclean: {df['full_text_postclean'].isna().sum()}")

    preview_cols = [
        "article_id", "outlet_code", "article_seq", "outlet",
        "date_raw", "date_iso_full", "date_precision",
        "headline_clean", "lead_clean", "body_postclean", "full_text_postclean"
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(10))

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()