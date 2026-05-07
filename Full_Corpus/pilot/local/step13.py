import os
import json
import re
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_RAW_JSONL = "data/local_raw_outputs.jsonl"
INPUT_PAYLOADS_JSONL = "data/gemini_batch_payloads.jsonl"
INPUT_CONSOLIDATED_CSV = "data/postConsolidated.csv"

OUTPUT_PARSED_CSV = "data/local_parsed_outputs.csv"
OUTPUT_PARSE_ERRORS_CSV = "data/local_parse_errors.csv"

OUTPUT_FINAL_WORKING_CSV = "data/final_local_coding_working.csv"
OUTPUT_MERGE_ISSUES_CSV = "data/final_local_merge_issues.csv"

OUTPUT_CONSERVATIVE_CSV = "data/final_local_conservative_adjudicated_table.csv"
OUTPUT_LLM_AUTH_CSV = "data/final_local_llm_authoritative_table.csv"
OUTPUT_MANUAL_VERIFY_CSV = "data/final_local_manual_verification_table.csv"
OUTPUT_HIGH_CONF_CSV = "data/final_local_high_confidence_coding_table.csv"

# Optional Excel exports disabled by default.
EXPORT_XLSX = False

OUTPUT_PARSED_XLSX = "data/local_parsed_outputs.xlsx"
OUTPUT_PARSE_ERRORS_XLSX = "data/local_parse_errors.xlsx"
OUTPUT_FINAL_WORKING_XLSX = "data/final_local_coding_working.xlsx"
OUTPUT_MERGE_ISSUES_XLSX = "data/final_local_merge_issues.xlsx"
OUTPUT_CONSERVATIVE_XLSX = "data/final_local_conservative_adjudicated_table.xlsx"
OUTPUT_LLM_AUTH_XLSX = "data/final_local_llm_authoritative_table.xlsx"
OUTPUT_MANUAL_VERIFY_XLSX = "data/final_local_manual_verification_table.xlsx"
OUTPUT_HIGH_CONF_XLSX = "data/final_local_high_confidence_coding_table.xlsx"

# =========================
# PATH LOGIC
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

LOCAL_INPUT_RAW_JSONL = SCRIPT_DIR / INPUT_RAW_JSONL
LOCAL_INPUT_PAYLOADS_JSONL = SCRIPT_DIR / INPUT_PAYLOADS_JSONL
LOCAL_INPUT_CONSOLIDATED_CSV = SCRIPT_DIR / INPUT_CONSOLIDATED_CSV

LOCAL_OUTPUT_PARSED_CSV = SCRIPT_DIR / OUTPUT_PARSED_CSV
LOCAL_OUTPUT_PARSE_ERRORS_CSV = SCRIPT_DIR / OUTPUT_PARSE_ERRORS_CSV
LOCAL_OUTPUT_FINAL_WORKING_CSV = SCRIPT_DIR / OUTPUT_FINAL_WORKING_CSV
LOCAL_OUTPUT_MERGE_ISSUES_CSV = SCRIPT_DIR / OUTPUT_MERGE_ISSUES_CSV
LOCAL_OUTPUT_CONSERVATIVE_CSV = SCRIPT_DIR / OUTPUT_CONSERVATIVE_CSV
LOCAL_OUTPUT_LLM_AUTH_CSV = SCRIPT_DIR / OUTPUT_LLM_AUTH_CSV
LOCAL_OUTPUT_MANUAL_VERIFY_CSV = SCRIPT_DIR / OUTPUT_MANUAL_VERIFY_CSV
LOCAL_OUTPUT_HIGH_CONF_CSV = SCRIPT_DIR / OUTPUT_HIGH_CONF_CSV

LOCAL_OUTPUT_PARSED_XLSX = SCRIPT_DIR / OUTPUT_PARSED_XLSX
LOCAL_OUTPUT_PARSE_ERRORS_XLSX = SCRIPT_DIR / OUTPUT_PARSE_ERRORS_XLSX
LOCAL_OUTPUT_FINAL_WORKING_XLSX = SCRIPT_DIR / OUTPUT_FINAL_WORKING_XLSX
LOCAL_OUTPUT_MERGE_ISSUES_XLSX = SCRIPT_DIR / OUTPUT_MERGE_ISSUES_XLSX
LOCAL_OUTPUT_CONSERVATIVE_XLSX = SCRIPT_DIR / OUTPUT_CONSERVATIVE_XLSX
LOCAL_OUTPUT_LLM_AUTH_XLSX = SCRIPT_DIR / OUTPUT_LLM_AUTH_XLSX
LOCAL_OUTPUT_MANUAL_VERIFY_XLSX = SCRIPT_DIR / OUTPUT_MANUAL_VERIFY_XLSX
LOCAL_OUTPUT_HIGH_CONF_XLSX = SCRIPT_DIR / OUTPUT_HIGH_CONF_XLSX

# =========================
# REQUIRED OUTPUT FIELDS
# =========================
REQUIRED_FIELDS = [
    "Article_ID",
    "V04_Relevance_Final",
    "V05_Actor_Mention_Final",
    "V06_Successor_Frame_Final",
    "V07_Dominant_Label_Final",
    "V08_Stance_Final",
    "V09_Dominant_Location_Final",
    "V10_Ambivalence_Final",
    "V11_Legitimation_Final",
    "V12_Counterterrorism_Final",
    "V13_Sovereignty_Final",
    "V14_Human_Rights_Abuse_Final",
    "V15_Anti_or_Neocolonialism_Final",
    "V16_Western_Failure_Final",
    "V17_Security_Effectiveness_Final",
    "V18_Economic_Interests_Final",
    "V19_Geopolitical_Rivalry_Final",
    "V20_Main_Associated_Actor_Final",
    "V21_Dominant_Discourse_Final",
    "LLM_Review_Note",
    "Pro_Review_Candidate",
    "Pro_Review_Reason",
]

ALLOWED_VALUES = {
    "V04_Relevance_Final": {1, 2, 3, 4},
    "V05_Actor_Mention_Final": {1, 2, 3, 4, 5},
    "V06_Successor_Frame_Final": {0, 1},
    "V07_Dominant_Label_Final": {1, 2, 3, 4, 5, 6},
    "V08_Stance_Final": {1, 2, 3, 4, 5},
    "V09_Dominant_Location_Final": {1, 2, 3, 4, 5},
    "V10_Ambivalence_Final": {0, 1},
    "V11_Legitimation_Final": {1, 2, 3, 4},
    "V12_Counterterrorism_Final": {0, 1},
    "V13_Sovereignty_Final": {0, 1},
    "V14_Human_Rights_Abuse_Final": {0, 1},
    "V15_Anti_or_Neocolonialism_Final": {0, 1},
    "V16_Western_Failure_Final": {0, 1},
    "V17_Security_Effectiveness_Final": {0, 1},
    "V18_Economic_Interests_Final": {0, 1},
    "V19_Geopolitical_Rivalry_Final": {0, 1},
    "V20_Main_Associated_Actor_Final": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
    "V21_Dominant_Discourse_Final": {1, 2, 3, 4, 5, 6},
    "Pro_Review_Candidate": {0, 1},
}

FINAL_CODE_COLS = [
    "V04_Relevance_Final",
    "V05_Actor_Mention_Final",
    "V06_Successor_Frame_Final",
    "V07_Dominant_Label_Final",
    "V08_Stance_Final",
    "V09_Dominant_Location_Final",
    "V10_Ambivalence_Final",
    "V11_Legitimation_Final",
    "V12_Counterterrorism_Final",
    "V13_Sovereignty_Final",
    "V14_Human_Rights_Abuse_Final",
    "V15_Anti_or_Neocolonialism_Final",
    "V16_Western_Failure_Final",
    "V17_Security_Effectiveness_Final",
    "V18_Economic_Interests_Final",
    "V19_Geopolitical_Rivalry_Final",
    "V20_Main_Associated_Actor_Final",
    "V21_Dominant_Discourse_Final",
]

TEXT_OUTPUT_COLS = [
    "LLM_Review_Note",
    "Pro_Review_Reason",
]

BROADER_FRAME_MAP = {
    "V13_Sovereignty_Final": "Sovereignty",
    "V15_Anti_or_Neocolonialism_Final": "Anti_or_Neocolonialism",
    "V16_Western_Failure_Final": "Western_Failure",
    "V17_Security_Effectiveness_Final": "Security_Effectiveness",
    "V19_Geopolitical_Rivalry_Final": "Geopolitical_Rivalry",
}

# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def safe_int(x, default=None):
    if x is None:
        return default
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if pd.isna(x):
            return default
        if float(x).is_integer():
            return int(x)
        return default

    s = safe_str(x)
    if s == "":
        return default

    try:
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

def ensure_output_dirs():
    (SCRIPT_DIR / "data").mkdir(parents=True, exist_ok=True)

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSONL file: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}")
    return records

def strip_code_fences(text):
    text = safe_str(text)
    if not text:
        return text

    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()

def extract_json_block(text):
    text = safe_str(text)
    if not text:
        return ""

    direct = text.strip()
    if direct.startswith("{") and direct.endswith("}"):
        return direct

    stripped = strip_code_fences(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    return stripped

def try_parse_response_json(response_text):
    candidate = extract_json_block(response_text)

    if not candidate:
        return None, "empty_response_text"

    try:
        parsed = json.loads(candidate)
        return parsed, ""
    except Exception as e:
        return None, f"json_parse_error: {e}"

def validate_schema(parsed_obj):
    missing = [field for field in REQUIRED_FIELDS if field not in parsed_obj]

    if missing:
        return 0, f"missing required fields: {missing}"

    return 1, ""

def validate_values(parsed_obj):
    errors = []

    for field, allowed in ALLOWED_VALUES.items():
        val = safe_int(parsed_obj.get(field), default=None)
        if val is None or val not in allowed:
            errors.append(f"{field} invalid value: {parsed_obj.get(field)}")

    for field in TEXT_OUTPUT_COLS:
        try:
            _ = safe_str(parsed_obj.get(field, ""))
        except Exception:
            errors.append(f"{field} invalid text value: {parsed_obj.get(field)}")

    article_id = normalize_article_id(parsed_obj.get("Article_ID"))
    if article_id is None:
        errors.append(f"Article_ID invalid value: {parsed_obj.get('Article_ID')}")

    if errors:
        return 0, "; ".join(errors)

    return 1, ""

def parsed_obj_to_row(parsed_obj):
    row = {}
    row["Article_ID"] = normalize_article_id(parsed_obj.get("Article_ID"))

    for col in FINAL_CODE_COLS:
        row[col] = safe_int(parsed_obj.get(col), default=None)

    row["LLM_Review_Note"] = safe_str(parsed_obj.get("LLM_Review_Note"))
    row["Pro_Review_Candidate"] = safe_int(parsed_obj.get("Pro_Review_Candidate"), default=None)
    row["Pro_Review_Reason"] = safe_str(parsed_obj.get("Pro_Review_Reason"))

    return row

def choose_best_raw_record(group):
    success_group = group[group["api_status"] == "success"]

    if not success_group.empty:
        return success_group.iloc[-1]

    return group.iloc[-1]

def build_merge_issue(article_id, issue_type, issue_detail, payload_meta=None, raw_meta=None):
    payload_meta = payload_meta or {}
    raw_meta = raw_meta or {}

    return {
        "Article_ID": article_id,
        "Issue_Type": issue_type,
        "Issue_Detail": issue_detail,
        "Prompt_Layer": payload_meta.get("prompt_layer", raw_meta.get("prompt_layer", "")),
        "Routing_Reason": payload_meta.get("routing_reason", raw_meta.get("routing_reason", "")),
        "API_Status": raw_meta.get("api_status", ""),
        "Parse_Status": raw_meta.get("Parse_Status", ""),
    }

def print_progress(i, total, every=100, prefix="Progress"):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"{prefix}: {i}/{total} ({pct}%)")

def maybe_export_xlsx(df, path):
    if EXPORT_XLSX:
        df.to_excel(path, index=False)

# =========================
# ADJUDICATION HELPERS
# =========================
def is_bulletin_case(row):
    headline = safe_str(row.get("Headline", "")).lower()
    lead = safe_str(row.get("Lead", "")).lower()
    body = safe_str(row.get("Body_Postclean", "")).lower()
    note = safe_str(row.get("Relevance_Note", "")).lower()

    if "les titres du" in headline:
        return 1
    if body.count("👉") >= 2 or lead.count("👉") >= 2:
        return 1
    if "bulletin-style article" in note:
        return 1
    if "ci-dessous, écoutez" in body:
        return 1

    return 0

def is_short_context_case(row):
    rel = safe_int(row.get("Relevance"), default=None)
    note = safe_str(row.get("Relevance_Note", "")).lower()

    if rel == 2:
        return 1
    if "single sentence" in note or "target sentence count=1" in note:
        return 1

    return 0

def has_multiple_labels_signal(row):
    pipeline_label = safe_int(row.get("Dominant_Label"), default=None)
    llm_label = safe_int(row.get("V07_Dominant_Label_Final"), default=None)
    note = safe_str(row.get("Dominant_Label_Note", "")).lower()

    if llm_label == 6:
        return 1
    if "multiple labels" in note:
        return 1
    if pipeline_label in [1, 5] and llm_label == 6:
        return 1

    return 0

# =========================
# ADJUDICATION RULES
# =========================
def adjudicate_v04(row):
    pipeline_val = safe_int(row.get("Relevance"), default=None)
    llm_val = safe_int(row.get("V04_Relevance_Final"), default=None)

    if llm_val is None:
        return pipeline_val

    if pipeline_val == 2 and llm_val == 3:
        if is_bulletin_case(row) == 1 or is_short_context_case(row) == 1:
            return 2

    return llm_val

def adjudicate_v05(row):
    llm_val = safe_int(row.get("V05_Actor_Mention_Final"), default=None)
    pipeline_val = safe_int(row.get("Actor_Mention"), default=None)
    return llm_val if llm_val is not None else pipeline_val

def adjudicate_v06(row):
    pipeline_val = safe_int(row.get("Successor_Frame"), default=None)
    llm_val = safe_int(row.get("V06_Successor_Frame_Final"), default=None)

    if llm_val is None:
        return pipeline_val

    if pipeline_val == 0 and llm_val == 1:
        return 0

    return llm_val

def adjudicate_v07(row):
    pipeline_val = safe_int(row.get("Dominant_Label"), default=None)
    llm_val = safe_int(row.get("V07_Dominant_Label_Final"), default=None)

    if llm_val is None:
        return pipeline_val

    if has_multiple_labels_signal(row) == 1:
        return llm_val

    return llm_val

def adjudicate_v08(row):
    pipeline_support = safe_int(row.get("Stance_Support"), default=None)
    llm_val = safe_int(row.get("V08_Stance_Final"), default=None)

    if llm_val is None:
        return pipeline_support

    if pipeline_support in [1, 2] and llm_val == 3:
        return pipeline_support

    return llm_val

def adjudicate_v09(row):
    llm_val = safe_int(row.get("V09_Dominant_Location_Final"), default=None)
    pipeline_val = safe_int(row.get("Dominant_Location"), default=None)
    return llm_val if llm_val is not None else pipeline_val

def adjudicate_v10(row):
    llm_val = safe_int(row.get("V10_Ambivalence_Final"), default=None)
    pipeline_val = safe_int(row.get("Ambivalence_Support"), default=None)
    return llm_val if llm_val is not None else pipeline_val

def adjudicate_v11(row):
    pipeline_support = safe_int(row.get("Legitimation_Support"), default=None)
    llm_val = safe_int(row.get("V11_Legitimation_Final"), default=None)

    if llm_val is None:
        return pipeline_support

    if pipeline_support in [1, 4] and llm_val in [2, 3]:
        return pipeline_support

    return llm_val

def adjudicate_broader_frame(row, llm_col, pipeline_col):
    llm_val = safe_int(row.get(llm_col), default=None)
    pipeline_val = safe_int(row.get(pipeline_col), default=None)

    if llm_val is None:
        return pipeline_val

    if is_short_context_case(row) == 1 and llm_val == 1:
        return pipeline_val if pipeline_val is not None else 0

    return llm_val

def adjudicate_v12(row):
    llm_val = safe_int(row.get("V12_Counterterrorism_Final"), default=None)
    pipeline_val = safe_int(row.get("Counterterrorism"), default=None)
    return llm_val if llm_val is not None else pipeline_val

def adjudicate_v13(row):
    return adjudicate_broader_frame(row, "V13_Sovereignty_Final", "Sovereignty")

def adjudicate_v14(row):
    llm_val = safe_int(row.get("V14_Human_Rights_Abuse_Final"), default=None)
    pipeline_val = safe_int(row.get("Human_Rights_Abuse"), default=None)
    return llm_val if llm_val is not None else pipeline_val

def adjudicate_v15(row):
    return adjudicate_broader_frame(row, "V15_Anti_or_Neocolonialism_Final", "Anti_or_Neocolonialism")

def adjudicate_v16(row):
    return adjudicate_broader_frame(row, "V16_Western_Failure_Final", "Western_Failure")

def adjudicate_v17(row):
    return adjudicate_broader_frame(row, "V17_Security_Effectiveness_Final", "Security_Effectiveness")

def adjudicate_v18(row):
    llm_val = safe_int(row.get("V18_Economic_Interests_Final"), default=None)
    pipeline_val = safe_int(row.get("Economic_Interests"), default=None)
    return llm_val if llm_val is not None else pipeline_val

def adjudicate_v19(row):
    return adjudicate_broader_frame(row, "V19_Geopolitical_Rivalry_Final", "Geopolitical_Rivalry")

def adjudicate_v20(row):
    llm_val = safe_int(row.get("V20_Main_Associated_Actor_Final"), default=None)
    pipeline_val = safe_int(row.get("Main_Associated_Actor"), default=None)

    if llm_val is None:
        return pipeline_val

    if is_short_context_case(row) == 1 and pipeline_val is not None and llm_val != pipeline_val:
        return pipeline_val

    return llm_val

def adjudicate_v21(row):
    llm_val = safe_int(row.get("V21_Dominant_Discourse_Final"), default=None)
    pipeline_support = safe_int(row.get("Dominant_Discourse_Support"), default=None)

    if llm_val is None:
        return pipeline_support

    if pipeline_support in [5, 6] and llm_val in [1, 2, 3, 4]:
        if is_bulletin_case(row) == 1 or is_short_context_case(row) == 1:
            return pipeline_support

    return llm_val

# =========================
# MANUAL CHECK TRIGGERS
# =========================
def build_manual_check_reasons(row):
    reasons = []

    pipeline_rel = safe_int(row.get("Relevance"), default=None)
    llm_rel = safe_int(row.get("V04_Relevance_Final"), default=None)

    if pipeline_rel == 2 and llm_rel == 3 and is_bulletin_case(row) == 1:
        reasons.append("relevance_upgrade_bulletin")

    pipeline_stance = safe_int(row.get("Stance_Support"), default=None)
    llm_stance = safe_int(row.get("V08_Stance_Final"), default=None)

    if pipeline_stance in [1, 2] and llm_stance == 3:
        reasons.append("positive_stance_shift")

    pipeline_leg = safe_int(row.get("Legitimation_Support"), default=None)
    llm_leg = safe_int(row.get("V11_Legitimation_Final"), default=None)

    if pipeline_leg in [1, 4] and llm_leg in [2, 3]:
        reasons.append("legitimation_shift")

    pipeline_succ = safe_int(row.get("Successor_Frame"), default=None)
    llm_succ = safe_int(row.get("V06_Successor_Frame_Final"), default=None)

    if pipeline_succ == 0 and llm_succ == 1:
        reasons.append("successor_upgrade")

    if is_short_context_case(row) == 1:
        active_broad = []
        for llm_col, _pipeline_col in BROADER_FRAME_MAP.items():
            llm_val = safe_int(row.get(llm_col), default=None)
            if llm_val == 1:
                active_broad.append(llm_col)

        if len(active_broad) >= 2:
            reasons.append("broad_frame_activation_short_context")

    pipeline_disc = safe_int(row.get("Dominant_Discourse_Support"), default=None)
    llm_disc = safe_int(row.get("V21_Dominant_Discourse_Final"), default=None)

    if pipeline_disc == 6 and llm_disc in [1, 2, 3, 4]:
        reasons.append("forced_discourse_resolution")

    pipeline_actor = safe_int(row.get("Main_Associated_Actor"), default=None)
    llm_actor = safe_int(row.get("V20_Main_Associated_Actor_Final"), default=None)
    review_count = safe_int(row.get("Review_Flag_Count"), default=0)

    if pipeline_actor is not None and llm_actor is not None and pipeline_actor != llm_actor and review_count >= 1:
        reasons.append("associated_actor_change")

    pro_review = safe_int(row.get("Pro_Review_Candidate"), default=None)

    if pro_review == 1:
        reasons.append("pro_review_candidate")

    llm_label = safe_int(row.get("V07_Dominant_Label_Final"), default=None)

    if llm_label == 6:
        reasons.append("multiple_labels")

    return reasons

def manual_check_required(row):
    reasons = build_manual_check_reasons(row)

    hard_reasons = {
        "relevance_upgrade_bulletin",
        "positive_stance_shift",
        "legitimation_shift",
        "successor_upgrade",
        "broad_frame_activation_short_context",
        "pro_review_candidate",
    }

    return 1 if any(r in hard_reasons for r in reasons) else 0

# =========================
# MAIN
# =========================
def main():
    ensure_output_dirs()

    if not LOCAL_INPUT_RAW_JSONL.exists():
        raise FileNotFoundError(f"Missing raw output file: {LOCAL_INPUT_RAW_JSONL}")

    if not LOCAL_INPUT_PAYLOADS_JSONL.exists():
        raise FileNotFoundError(f"Missing payload file: {LOCAL_INPUT_PAYLOADS_JSONL}")

    if not LOCAL_INPUT_CONSOLIDATED_CSV.exists():
        raise FileNotFoundError(
            f"Missing consolidated CSV: {LOCAL_INPUT_CONSOLIDATED_CSV}\n"
            f"Expected full CSV-only file with 2474 rows. "
            f"If missing, rerun LOCAL step12 or copy pilot/data/postConsolidated.csv into LOCAL/data/."
        )

    raw_records = load_jsonl(LOCAL_INPUT_RAW_JSONL)
    payload_records = load_jsonl(LOCAL_INPUT_PAYLOADS_JSONL)

    df_raw = pd.DataFrame(raw_records)
    df_payload = pd.DataFrame(payload_records)
    df_cons = pd.read_csv(LOCAL_INPUT_CONSOLIDATED_CSV, dtype={"Article_ID": str}, low_memory=False)

    # Defensive column guarantees
    if "article_id" not in df_raw.columns:
        df_raw["article_id"] = None

    if "article_id" not in df_payload.columns:
        df_payload["article_id"] = None

    if "Article_ID" not in df_cons.columns:
        raise ValueError("postConsolidated.csv is missing required column: Article_ID")

    df_raw["article_id"] = df_raw["article_id"].apply(normalize_article_id)
    df_payload["article_id"] = df_payload["article_id"].apply(normalize_article_id)
    df_cons["Article_ID"] = df_cons["Article_ID"].apply(normalize_article_id)

    merge_issues = []

    # Duplicate raw output issue note
    if not df_raw.empty and df_raw.duplicated(subset=["article_id"]).any():
        dup_ids = df_raw[df_raw.duplicated(subset=["article_id"], keep=False)]["article_id"].dropna().unique().tolist()

        for aid in dup_ids:
            payload_meta = {}
            payload_match = df_payload[df_payload["article_id"] == aid]

            if not payload_match.empty:
                payload_meta = payload_match.iloc[-1].to_dict()

            merge_issues.append(
                build_merge_issue(
                    article_id=aid,
                    issue_type="duplicate_raw_outputs",
                    issue_detail="multiple raw outputs found for article_id; last successful or last available record selected",
                    payload_meta=payload_meta
                )
            )

    # =========================
    # ROBUST RAW-BEST SELECTION
    # =========================
    best_rows = []

    if not df_raw.empty:
        grouped = df_raw.groupby("article_id", dropna=False)
        total_groups = len(grouped)

        print(f"\nLocal Step13: selecting best raw record from {total_groups} grouped article_ids")

        for i, (_, group) in enumerate(grouped, start=1):
            best_rows.append(choose_best_raw_record(group))
            print_progress(i, total_groups, every=50, prefix="Best-record selection progress")

    df_raw_best = pd.DataFrame(best_rows).reset_index(drop=True)

    if "article_id" not in df_raw_best.columns:
        df_raw_best["article_id"] = None

    # =========================
    # PARSE / VALIDATE
    # =========================
    parsed_rows = []
    parse_error_rows = []

    total = len(df_raw_best)
    print(f"\nLocal Step13: parsing {total} selected raw output records")

    for i, (_, raw) in enumerate(df_raw_best.iterrows(), start=1):
        article_id = raw.get("article_id")
        api_status = safe_str(raw.get("api_status"))
        http_status = raw.get("http_status")
        response_text = safe_str(raw.get("response_text"))
        system_prompt_version = safe_str(raw.get("system_prompt_version"))
        model_family = safe_str(raw.get("model_family"))
        prompt_layer = safe_str(raw.get("prompt_layer"))
        routing_reason = safe_str(raw.get("routing_reason"))
        prellm_pro_review_support = raw.get("prellm_pro_review_support", 0)
        error_message = safe_str(raw.get("error_message"))

        parse_status = "not_attempted"
        schema_valid = 0
        validation_valid = 0
        llm_output_valid = 0
        error_type = ""
        error_detail = ""

        if api_status != "success":
            error_type = "api_error"
            error_detail = error_message if error_message else "API request failed"

            parse_error_rows.append({
                "Article_ID": article_id,
                "System_Prompt_Version": system_prompt_version,
                "Model_Family": model_family,
                "Prompt_Layer": prompt_layer,
                "Routing_Reason": routing_reason,
                "PreLLM_Pro_Review_Support": prellm_pro_review_support,
                "API_Status": api_status,
                "HTTP_Status": http_status,
                "Parse_Status": parse_status,
                "Schema_Valid": schema_valid,
                "Validation_Valid": validation_valid,
                "LLM_Output_Valid": llm_output_valid,
                "Error_Type": error_type,
                "Error_Detail": error_detail,
                "Response_Text": response_text,
            })

            print_progress(i, total, every=50, prefix="Parsing progress")
            continue

        parsed_obj, parse_err = try_parse_response_json(response_text)

        if parsed_obj is None:
            parse_status = "parse_error"
            error_type = "parse_error"
            error_detail = parse_err

            parse_error_rows.append({
                "Article_ID": article_id,
                "System_Prompt_Version": system_prompt_version,
                "Model_Family": model_family,
                "Prompt_Layer": prompt_layer,
                "Routing_Reason": routing_reason,
                "PreLLM_Pro_Review_Support": prellm_pro_review_support,
                "API_Status": api_status,
                "HTTP_Status": http_status,
                "Parse_Status": parse_status,
                "Schema_Valid": schema_valid,
                "Validation_Valid": validation_valid,
                "LLM_Output_Valid": llm_output_valid,
                "Error_Type": error_type,
                "Error_Detail": error_detail,
                "Response_Text": response_text,
            })

            print_progress(i, total, every=50, prefix="Parsing progress")
            continue

        parse_status = "parsed"

        schema_valid, schema_err = validate_schema(parsed_obj)

        if schema_valid != 1:
            error_type = "schema_error"
            error_detail = schema_err

            parse_error_rows.append({
                "Article_ID": article_id,
                "System_Prompt_Version": system_prompt_version,
                "Model_Family": model_family,
                "Prompt_Layer": prompt_layer,
                "Routing_Reason": routing_reason,
                "PreLLM_Pro_Review_Support": prellm_pro_review_support,
                "API_Status": api_status,
                "HTTP_Status": http_status,
                "Parse_Status": parse_status,
                "Schema_Valid": schema_valid,
                "Validation_Valid": validation_valid,
                "LLM_Output_Valid": llm_output_valid,
                "Error_Type": error_type,
                "Error_Detail": error_detail,
                "Response_Text": response_text,
            })

            print_progress(i, total, every=50, prefix="Parsing progress")
            continue

        validation_valid, validation_err = validate_values(parsed_obj)

        if validation_valid != 1:
            error_type = "validation_error"
            error_detail = validation_err

            parse_error_rows.append({
                "Article_ID": article_id,
                "System_Prompt_Version": system_prompt_version,
                "Model_Family": model_family,
                "Prompt_Layer": prompt_layer,
                "Routing_Reason": routing_reason,
                "PreLLM_Pro_Review_Support": prellm_pro_review_support,
                "API_Status": api_status,
                "HTTP_Status": http_status,
                "Parse_Status": parse_status,
                "Schema_Valid": schema_valid,
                "Validation_Valid": validation_valid,
                "LLM_Output_Valid": llm_output_valid,
                "Error_Type": error_type,
                "Error_Detail": error_detail,
                "Response_Text": response_text,
            })

            print_progress(i, total, every=50, prefix="Parsing progress")
            continue

        llm_output_valid = 1
        parsed_row = parsed_obj_to_row(parsed_obj)

        parsed_row.update({
            "System_Prompt_Version": system_prompt_version,
            "Model_Family": model_family,
            "Prompt_Layer": prompt_layer,
            "Routing_Reason": routing_reason,
            "PreLLM_Pro_Review_Support": prellm_pro_review_support,
            "API_Status": api_status,
            "HTTP_Status": http_status,
            "Parse_Status": parse_status,
            "Schema_Valid": schema_valid,
            "Validation_Valid": validation_valid,
            "LLM_Output_Valid": llm_output_valid,
        })

        parsed_rows.append(parsed_row)

        print_progress(i, total, every=50, prefix="Parsing progress")

    df_parsed = pd.DataFrame(parsed_rows)
    df_parse_errors = pd.DataFrame(parse_error_rows)

    if df_parsed.empty:
        df_parsed = pd.DataFrame(columns=[
            "Article_ID",
            *FINAL_CODE_COLS,
            "LLM_Review_Note",
            "Pro_Review_Candidate",
            "Pro_Review_Reason",
            "System_Prompt_Version",
            "Model_Family",
            "Prompt_Layer",
            "Routing_Reason",
            "PreLLM_Pro_Review_Support",
            "API_Status",
            "HTTP_Status",
            "Parse_Status",
            "Schema_Valid",
            "Validation_Valid",
            "LLM_Output_Valid",
        ])

    if "Article_ID" not in df_parsed.columns:
        df_parsed["Article_ID"] = None

    if df_parse_errors.empty:
        df_parse_errors = pd.DataFrame(columns=[
            "Article_ID",
            "System_Prompt_Version",
            "Model_Family",
            "Prompt_Layer",
            "Routing_Reason",
            "PreLLM_Pro_Review_Support",
            "API_Status",
            "HTTP_Status",
            "Parse_Status",
            "Schema_Valid",
            "Validation_Valid",
            "LLM_Output_Valid",
            "Error_Type",
            "Error_Detail",
            "Response_Text",
        ])

    df_parsed.to_csv(LOCAL_OUTPUT_PARSED_CSV, index=False, encoding="utf-8-sig")
    df_parse_errors.to_csv(LOCAL_OUTPUT_PARSE_ERRORS_CSV, index=False, encoding="utf-8-sig")

    maybe_export_xlsx(df_parsed, LOCAL_OUTPUT_PARSED_XLSX)
    maybe_export_xlsx(df_parse_errors, LOCAL_OUTPUT_PARSE_ERRORS_XLSX)

    # =========================
    # BUILD WORKING TABLE
    # =========================
    payload_keep = [
        "article_id",
        "system_prompt_version",
        "model_family",
        "prompt_layer",
        "routing_reason",
        "prellm_pro_review_support",
    ]

    df_payload = ensure_columns(df_payload, payload_keep, fill_value=None)

    df_payload_sub = df_payload[payload_keep].copy()
    df_payload_sub = df_payload_sub.rename(columns={
        "article_id": "Article_ID",
        "system_prompt_version": "Payload_System_Prompt_Version",
        "model_family": "Payload_Model_Family",
        "prompt_layer": "Payload_Prompt_Layer",
        "routing_reason": "Payload_Routing_Reason",
        "prellm_pro_review_support": "Payload_PreLLM_Pro_Review_Support",
    })

    df_work = df_cons.merge(df_payload_sub, on="Article_ID", how="left")

    parsed_keep = [
        "Article_ID",
        *FINAL_CODE_COLS,
        "LLM_Review_Note",
        "Pro_Review_Candidate",
        "Pro_Review_Reason",
        "System_Prompt_Version",
        "Model_Family",
        "Prompt_Layer",
        "Routing_Reason",
        "PreLLM_Pro_Review_Support",
        "API_Status",
        "HTTP_Status",
        "Parse_Status",
        "Schema_Valid",
        "Validation_Valid",
        "LLM_Output_Valid",
    ]

    df_parsed = ensure_columns(df_parsed, parsed_keep, fill_value=None)

    df_work = df_work.merge(df_parsed[parsed_keep], on="Article_ID", how="left")

    df_work["Went_To_LLM"] = df_work["Article_ID"].isin(df_payload_sub["Article_ID"]).astype(int)

    def derive_merge_status(row):
        if safe_int(row.get("Went_To_LLM"), 0) == 0:
            return "excluded_relevance_1"

        if pd.notna(row.get("LLM_Output_Valid")) and safe_int(row.get("LLM_Output_Valid"), 0) == 1:
            return "merged"

        return "missing_llm_output"

    df_work["Merge_Status"] = df_work.apply(derive_merge_status, axis=1)

    # =========================
    # MERGE ISSUES
    # =========================
    if "Article_ID" not in df_payload_sub.columns:
        df_payload_sub["Article_ID"] = None
    if "Article_ID" not in df_parsed.columns:
        df_parsed["Article_ID"] = None
    if "article_id" not in df_raw_best.columns:
        df_raw_best["article_id"] = None
    if "Article_ID" not in df_cons.columns:
        df_cons["Article_ID"] = None

    payload_ids = set(df_payload_sub["Article_ID"].dropna().tolist())
    parsed_ids = set(df_parsed["Article_ID"].dropna().tolist())
    raw_ids = set(df_raw_best["article_id"].dropna().tolist())
    cons_ids = set(df_cons["Article_ID"].dropna().tolist())

    for aid in sorted(payload_ids - raw_ids):
        payload_match = df_payload[df_payload["article_id"] == aid]
        payload_meta = payload_match.iloc[-1].to_dict() if not payload_match.empty else {}

        merge_issues.append(
            build_merge_issue(
                article_id=aid,
                issue_type="missing_raw_output",
                issue_detail="article present in payloads but missing in raw outputs",
                payload_meta=payload_meta
            )
        )

    invalid_or_missing_ids = payload_ids - parsed_ids

    for aid in sorted(invalid_or_missing_ids):
        payload_match = df_payload[df_payload["article_id"] == aid]
        raw_match = df_raw_best[df_raw_best["article_id"] == aid]

        payload_meta = payload_match.iloc[-1].to_dict() if not payload_match.empty else {}
        raw_meta = raw_match.iloc[-1].to_dict() if not raw_match.empty else {}

        parse_match = df_parse_errors[df_parse_errors["Article_ID"] == aid]

        if not parse_match.empty:
            detail = safe_str(parse_match.iloc[-1]["Error_Detail"])
            issue_type = safe_str(parse_match.iloc[-1]["Error_Type"]) or "invalid_llm_output"
        else:
            detail = "no valid parsed output available"
            issue_type = "missing_llm_output"

        merge_issues.append(
            build_merge_issue(
                article_id=aid,
                issue_type=issue_type,
                issue_detail=detail,
                payload_meta=payload_meta,
                raw_meta=raw_meta
            )
        )

    for aid in sorted(parsed_ids - cons_ids):
        raw_match = df_raw_best[df_raw_best["article_id"] == aid]
        raw_meta = raw_match.iloc[-1].to_dict() if not raw_match.empty else {}

        merge_issues.append(
            build_merge_issue(
                article_id=aid,
                issue_type="id_mismatch",
                issue_detail="valid parsed output exists but Article_ID not found in postConsolidated.csv",
                raw_meta=raw_meta
            )
        )

    df_merge_issues = pd.DataFrame(merge_issues)

    if df_merge_issues.empty:
        df_merge_issues = pd.DataFrame(columns=[
            "Article_ID",
            "Issue_Type",
            "Issue_Detail",
            "Prompt_Layer",
            "Routing_Reason",
            "API_Status",
            "Parse_Status",
        ])

    df_merge_issues.to_csv(LOCAL_OUTPUT_MERGE_ISSUES_CSV, index=False, encoding="utf-8-sig")
    maybe_export_xlsx(df_merge_issues, LOCAL_OUTPUT_MERGE_ISSUES_XLSX)

    # =========================
    # ADJUDICATED FIELDS
    # =========================
    df_work["Adj_V04_Relevance"] = df_work.apply(adjudicate_v04, axis=1)
    df_work["Adj_V05_Actor_Mention"] = df_work.apply(adjudicate_v05, axis=1)
    df_work["Adj_V06_Successor_Frame"] = df_work.apply(adjudicate_v06, axis=1)
    df_work["Adj_V07_Dominant_Label"] = df_work.apply(adjudicate_v07, axis=1)
    df_work["Adj_V08_Stance"] = df_work.apply(adjudicate_v08, axis=1)
    df_work["Adj_V09_Dominant_Location"] = df_work.apply(adjudicate_v09, axis=1)
    df_work["Adj_V10_Ambivalence"] = df_work.apply(adjudicate_v10, axis=1)
    df_work["Adj_V11_Legitimation"] = df_work.apply(adjudicate_v11, axis=1)
    df_work["Adj_V12_Counterterrorism"] = df_work.apply(adjudicate_v12, axis=1)
    df_work["Adj_V13_Sovereignty"] = df_work.apply(adjudicate_v13, axis=1)
    df_work["Adj_V14_Human_Rights_Abuse"] = df_work.apply(adjudicate_v14, axis=1)
    df_work["Adj_V15_Anti_or_Neocolonialism"] = df_work.apply(adjudicate_v15, axis=1)
    df_work["Adj_V16_Western_Failure"] = df_work.apply(adjudicate_v16, axis=1)
    df_work["Adj_V17_Security_Effectiveness"] = df_work.apply(adjudicate_v17, axis=1)
    df_work["Adj_V18_Economic_Interests"] = df_work.apply(adjudicate_v18, axis=1)
    df_work["Adj_V19_Geopolitical_Rivalry"] = df_work.apply(adjudicate_v19, axis=1)
    df_work["Adj_V20_Main_Associated_Actor"] = df_work.apply(adjudicate_v20, axis=1)
    df_work["Adj_V21_Dominant_Discourse"] = df_work.apply(adjudicate_v21, axis=1)

    df_work["Manual_Check_Reasons"] = df_work.apply(
        lambda row: ";".join(build_manual_check_reasons(row)),
        axis=1
    )

    df_work["Manual_Check_Required"] = df_work.apply(manual_check_required, axis=1)

    for col in [
        "Human_Final_V04",
        "Human_Final_V08",
        "Human_Final_V11",
        "Human_Final_V21",
        "Human_Final_Notes"
    ]:
        if col not in df_work.columns:
            df_work[col] = None

    df_work.to_csv(LOCAL_OUTPUT_FINAL_WORKING_CSV, index=False, encoding="utf-8-sig")
    maybe_export_xlsx(df_work, LOCAL_OUTPUT_FINAL_WORKING_XLSX)

    # =========================
    # OUTPUT TABLES
    # =========================

    # 1. LLM-authoritative table = only valid parsed LLM outputs
    df_llm_auth = df_work[
        (df_work["Went_To_LLM"] == 1) &
        (df_work["LLM_Output_Valid"] == 1)
    ].copy()

    llm_auth_cols = [
        "Article_ID",
        "Outlet",
        "Date",
        *FINAL_CODE_COLS,
        "LLM_Review_Note",
        "Pro_Review_Candidate",
        "Pro_Review_Reason",
        "Prompt_Layer",
        "Routing_Reason",
        "System_Prompt_Version",
        "Model_Family",
    ]

    llm_auth_cols = [c for c in llm_auth_cols if c in df_llm_auth.columns]
    df_llm_auth_out = df_llm_auth[llm_auth_cols].copy()

    df_llm_auth_out.to_csv(LOCAL_OUTPUT_LLM_AUTH_CSV, index=False, encoding="utf-8-sig")
    maybe_export_xlsx(df_llm_auth_out, LOCAL_OUTPUT_LLM_AUTH_XLSX)

    # 2. Conservative adjudicated table = all LLM-sent rows, with fallback where invalid/missing
    df_adj = df_work[df_work["Went_To_LLM"] == 1].copy()

    adj_export_cols = [
        "Article_ID",
        "Outlet",
        "Date",
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

    adj_export_cols = [c for c in adj_export_cols if c in df_adj.columns]
    df_adj_out = df_adj[adj_export_cols].copy()

    df_adj_out.to_csv(LOCAL_OUTPUT_CONSERVATIVE_CSV, index=False, encoding="utf-8-sig")
    maybe_export_xlsx(df_adj_out, LOCAL_OUTPUT_CONSERVATIVE_XLSX)

    # 3. Manual verification table
    df_manual = df_work[df_work["Went_To_LLM"] == 1].copy()

    manual_cols = [
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
        "Counterterrorism",
        "Sovereignty",
        "Human_Rights_Abuse",
        "Anti_or_Neocolonialism",
        "Western_Failure",
        "Security_Effectiveness",
        "Economic_Interests",
        "Geopolitical_Rivalry",
        "Stance_Support",
        "Legitimation_Support",
        "Dominant_Discourse_Support",
        *FINAL_CODE_COLS,
        "LLM_Review_Note",
        "Pro_Review_Candidate",
        "Pro_Review_Reason",
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
        "Human_Final_V04",
        "Human_Final_V08",
        "Human_Final_V11",
        "Human_Final_V21",
        "Human_Final_Notes",
    ]

    manual_cols = [c for c in manual_cols if c in df_manual.columns]
    df_manual_out = df_manual[manual_cols].copy()

    df_manual_out.to_csv(LOCAL_OUTPUT_MANUAL_VERIFY_CSV, index=False, encoding="utf-8-sig")
    maybe_export_xlsx(df_manual_out, LOCAL_OUTPUT_MANUAL_VERIFY_XLSX)

    # 4. High-confidence subset
    df_high_conf = df_adj[
        (df_adj["Went_To_LLM"] == 1) &
        (df_adj["LLM_Output_Valid"] == 1) &
        (df_adj["Manual_Check_Required"] == 0) &
        (df_adj["Pro_Review_Candidate"].fillna(0).apply(lambda x: safe_int(x, 0)) == 0)
    ].copy()

    high_conf_cols = [
        "Article_ID",
        "Outlet",
        "Date",
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
    ]

    high_conf_cols = [c for c in high_conf_cols if c in df_high_conf.columns]
    df_high_conf_out = df_high_conf[high_conf_cols].copy()

    df_high_conf_out.to_csv(LOCAL_OUTPUT_HIGH_CONF_CSV, index=False, encoding="utf-8-sig")
    maybe_export_xlsx(df_high_conf_out, LOCAL_OUTPUT_HIGH_CONF_XLSX)

    # =========================
    # SUMMARY
    # =========================
    print(f"\nLoaded raw records: {len(df_raw)}")
    print(f"Loaded payload records: {len(df_payload)}")
    print(f"Loaded consolidated records: {len(df_cons)}")

    print(f"\nValid parsed outputs: {len(df_parsed)}")
    print(f"Parse/validation errors: {len(df_parse_errors)}")
    print(f"Merge issues: {len(df_merge_issues)}")

    print(f"\nLLM-authoritative rows: {len(df_llm_auth_out)}")
    print(f"Conservative adjudicated rows: {len(df_adj_out)}")
    print(f"Manual verification rows: {len(df_manual_out)}")
    print(f"High-confidence rows: {len(df_high_conf_out)}")

    print(f"\nSaved parsed outputs to {LOCAL_OUTPUT_PARSED_CSV}")
    print(f"Saved parse errors to {LOCAL_OUTPUT_PARSE_ERRORS_CSV}")
    print(f"Saved final working table to {LOCAL_OUTPUT_FINAL_WORKING_CSV}")
    print(f"Saved merge issues to {LOCAL_OUTPUT_MERGE_ISSUES_CSV}")
    print(f"Saved LLM-authoritative table to {LOCAL_OUTPUT_LLM_AUTH_CSV}")
    print(f"Saved conservative adjudicated table to {LOCAL_OUTPUT_CONSERVATIVE_CSV}")
    print(f"Saved manual verification table to {LOCAL_OUTPUT_MANUAL_VERIFY_CSV}")
    print(f"Saved high-confidence table to {LOCAL_OUTPUT_HIGH_CONF_CSV}")

    if EXPORT_XLSX:
        print("\nExcel exports also enabled.")
    else:
        print("\nExcel exports disabled. CSV-only mode active.")

if __name__ == "__main__":
    main()