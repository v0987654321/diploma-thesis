import os
import json
import time
import shutil
import requests
import pandas as pd
from pathlib import Path
import re

# =========================
# CONFIG
# =========================
SYSTEM_PROMPT_PATH = "prompts/gemini_system_instruction_v1.txt"
INPUT_JSONL = "data/gemini_batch_payloads.jsonl"

OUTPUT_RAW_JSONL = "data/local_raw_outputs.jsonl"
OUTPUT_LOG_CSV = "data/local_submission_log.csv"
OUTPUT_API_ERRORS_JSONL = "data/local_api_errors.jsonl"

OLLAMA_MODEL = "qwen2.5:14b-instruct"
OLLAMA_URL = "http://localhost:11434/api/chat"

REQUEST_TIMEOUT = 900
SLEEP_SECONDS = 1.0

# request-level retry
MAX_RETRIES = 3
INITIAL_RETRY_SLEEP = 5
BACKOFF_MULTIPLIER = 2
MAX_RETRY_SLEEP = 40

# batch-level rerun
MAX_BATCH_PASSES = 3
SLEEP_BETWEEN_PASSES = 20

COPY_PARENT_DATA_BEFORE_RUN = True

# =========================
# PATH LOGIC
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

LOCAL_DATA_DIR = SCRIPT_DIR / "data"
LOCAL_PROMPTS_DIR = SCRIPT_DIR / "prompts"
PARENT_DATA_DIR = PARENT_DIR / "data"

LOCAL_SYSTEM_PROMPT_PATH = SCRIPT_DIR / SYSTEM_PROMPT_PATH
LOCAL_INPUT_JSONL = SCRIPT_DIR / INPUT_JSONL

LOCAL_OUTPUT_RAW_JSONL = SCRIPT_DIR / OUTPUT_RAW_JSONL
LOCAL_OUTPUT_LOG_CSV = SCRIPT_DIR / OUTPUT_LOG_CSV
LOCAL_OUTPUT_API_ERRORS_JSONL = SCRIPT_DIR / OUTPUT_API_ERRORS_JSONL

# =========================
# LOCAL HARD OUTPUT ENFORCEMENT
# =========================
LOCAL_OUTPUT_SCHEMA = r'''
{
  "Article_ID": "",
  "V04_Relevance_Final": 0,
  "V05_Actor_Mention_Final": 0,
  "V06_Successor_Frame_Final": 0,
  "V07_Dominant_Label_Final": 0,
  "V08_Stance_Final": 0,
  "V09_Dominant_Location_Final": 0,
  "V10_Ambivalence_Final": 0,
  "V11_Legitimation_Final": 0,
  "V12_Counterterrorism_Final": 0,
  "V13_Sovereignty_Final": 0,
  "V14_Human_Rights_Abuse_Final": 0,
  "V15_Anti_or_Neocolonialism_Final": 0,
  "V16_Western_Failure_Final": 0,
  "V17_Security_Effectiveness_Final": 0,
  "V18_Economic_Interests_Final": 0,
  "V19_Geopolitical_Rivalry_Final": 0,
  "V20_Main_Associated_Actor_Final": 0,
  "V21_Dominant_Discourse_Final": 0,
  "LLM_Review_Note": "",
  "Pro_Review_Candidate": 0,
  "Pro_Review_Reason": ""
}
'''.strip()

LOCAL_ENFORCEMENT_SUFFIX = """
LOCAL OUTPUT RULES FOR THIS REQUEST:
- Return EXACTLY ONE JSON object.
- Return JSON ONLY.
- Do NOT write any explanation.
- Do NOT write any summary.
- Do NOT write any recommendations.
- Do NOT repeat pipeline notes outside the JSON.
- Do NOT use markdown code fences.
- Do NOT write any text before the JSON object.
- Do NOT write any text after the JSON object.
- Use EXACTLY the field names shown below.
- Do NOT rename any field.
- Every field below is REQUIRED.
- If uncertain, still fill every field using the allowed fallback code from the instructions.

CRITICAL ARTICLE_ID RULE:
- Article_ID MUST exactly equal the Article_ID shown in ARTICLE METADATA.
- Copy the Article_ID exactly as given.
- Do NOT write N/A.
- Do NOT leave Article_ID empty.
- Do NOT invent, shorten, transform, or replace Article_ID.
- The returned Article_ID is invalid unless it is an exact copy of the metadata value.

CRITICAL BINARY VARIABLE RULE:
- V12_Counterterrorism_Final must be only 0 or 1.
- V13_Sovereignty_Final must be only 0 or 1.
- V14_Human_Rights_Abuse_Final must be only 0 or 1.
- V15_Anti_or_Neocolonialism_Final must be only 0 or 1.
- V16_Western_Failure_Final must be only 0 or 1.
- V17_Security_Effectiveness_Final must be only 0 or 1.
- V18_Economic_Interests_Final must be only 0 or 1.
- V19_Geopolitical_Rivalry_Final must be only 0 or 1.
- Never use 2, 3, 4, 5, or any other number for V12 to V19.
- If uncertain for V12 to V19, output 0.

ADDITIONAL VALUE RULES:
- V08_Stance_Final must be only 1, 2, 3, 4, or 5.
- V20_Main_Associated_Actor_Final must be only 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10.
- If uncertain, still choose one allowed code. Never output an invalid code.

REQUIRED JSON SCHEMA:
""".strip()

# =========================
# HELPERS
# =========================
def safe_str(x):
    if x is None:
        return ""
    return str(x).strip()

def ensure_output_dirs():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

def copy_parent_data_to_local():
    if not PARENT_DATA_DIR.exists():
        raise FileNotFoundError(f"Parent data directory not found: {PARENT_DATA_DIR}")

    copied = 0
    skipped = 0

    for item in PARENT_DATA_DIR.iterdir():
        src = item
        dst = LOCAL_DATA_DIR / item.name

        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied += 1
        elif item.is_file():
            shutil.copy2(src, dst)
            copied += 1
        else:
            skipped += 1

    print(f"Copied {copied} items from parent data directory to local LOCAL/data")
    if skipped > 0:
        print(f"Skipped {skipped} non-standard items")

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Required text file is empty: {path}")
    return text

def load_jsonl(path):
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

def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def validate_payload_record(record):
    required_fields = [
        "article_id",
        "system_prompt_version",
        "model_family",
        "prompt_layer",
        "routing_reason",
        "prellm_pro_review_support",
        "user_payload",
    ]
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"Payload record missing required fields: {missing}")

def response_text_length(text):
    return len(text) if text else 0

def extract_ollama_response_text(response_json):
    try:
        msg = response_json.get("message", {})
        content = msg.get("content", "")
        return safe_str(content)
    except Exception:
        return ""

def extract_article_id_from_payload(user_payload):
    text = safe_str(user_payload)
    m = re.search(r"-\s*Article_ID:\s*([0-9A-Za-z_-]+)", text)
    if m:
        return m.group(1).strip()
    return ""

def build_local_user_payload(user_payload):
    exact_article_id = extract_article_id_from_payload(user_payload)

    suffix = f"""
{LOCAL_ENFORCEMENT_SUFFIX}

EXACT Article_ID TO COPY:
{exact_article_id}

REQUIRED JSON SCHEMA:
{LOCAL_OUTPUT_SCHEMA}
""".strip()

    return f"""
{user_payload}

{suffix}
""".strip()

def build_messages(system_instruction, user_payload):
    local_user_payload = build_local_user_payload(user_payload)

    return [
        {
            "role": "system",
            "content": system_instruction
        },
        {
            "role": "user",
            "content": local_user_payload
        }
    ]

def build_request_body(system_instruction, user_payload):
    return {
        "model": OLLAMA_MODEL,
        "messages": build_messages(system_instruction, user_payload),
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "repeat_penalty": 1.0,
        }
    }

def is_retryable_status(http_status):
    return http_status in [408, 429, 500, 502, 503, 504]

def print_progress(i, total, every=25, prefix="Progress"):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"{prefix}: {i}/{total} ({pct}%)")

def submit_request(system_instruction, user_payload):
    body = build_request_body(system_instruction, user_payload)

    response = requests.post(
        OLLAMA_URL,
        json=body,
        timeout=REQUEST_TIMEOUT
    )

    http_status = response.status_code

    try:
        response_json = response.json()
    except Exception:
        response_json = {
            "non_json_response_text": response.text
        }

    return http_status, response_json

def submit_with_retry(system_instruction, user_payload):
    last_http_status = None
    last_response_json = {}
    last_error_message = ""
    attempt_count = 0
    retried_flag = 0

    retry_sleep = INITIAL_RETRY_SLEEP

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_count = attempt
        try:
            http_status, response_json = submit_request(system_instruction, user_payload)
            last_http_status = http_status
            last_response_json = response_json

            if http_status == 200:
                return {
                    "api_status": "success",
                    "http_status": http_status,
                    "response_json": response_json,
                    "error_message": "",
                    "attempt_count": attempt_count,
                    "retried_flag": 1 if attempt_count > 1 else 0
                }

            last_error_message = safe_str(response_json)

            if is_retryable_status(http_status) and attempt < MAX_RETRIES:
                retried_flag = 1
                print(f"  Retryable HTTP {http_status}. Waiting {retry_sleep}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(retry_sleep)
                retry_sleep = min(retry_sleep * BACKOFF_MULTIPLIER, MAX_RETRY_SLEEP)
                continue

            return {
                "api_status": "error",
                "http_status": http_status,
                "response_json": response_json,
                "error_message": last_error_message,
                "attempt_count": attempt_count,
                "retried_flag": 1 if attempt_count > 1 else retried_flag
            }

        except Exception as e:
            last_http_status = None
            last_response_json = {}
            last_error_message = f"{type(e).__name__}: {e}"

            if attempt < MAX_RETRIES:
                retried_flag = 1
                print(f"  Exception on attempt {attempt}/{MAX_RETRIES}: {last_error_message}")
                print(f"  Waiting {retry_sleep}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(retry_sleep)
                retry_sleep = min(retry_sleep * BACKOFF_MULTIPLIER, MAX_RETRY_SLEEP)
                continue

            return {
                "api_status": "error",
                "http_status": last_http_status,
                "response_json": last_response_json,
                "error_message": last_error_message,
                "attempt_count": attempt_count,
                "retried_flag": 1 if attempt_count > 1 else retried_flag
            }

    return {
        "api_status": "error",
        "http_status": last_http_status,
        "response_json": last_response_json,
        "error_message": last_error_message,
        "attempt_count": attempt_count,
        "retried_flag": retried_flag
    }

def write_log_csv(log_rows):
    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(LOCAL_OUTPUT_LOG_CSV, index=False, encoding="utf-8-sig")

def load_existing_log_rows():
    if not LOCAL_OUTPUT_LOG_CSV.exists():
        return []
    try:
        df = pd.read_csv(LOCAL_OUTPUT_LOG_CSV, low_memory=False)
        return df.to_dict("records")
    except Exception:
        return []

def get_failed_article_ids_from_log(log_rows):
    if not log_rows:
        return []

    df_log = pd.DataFrame(log_rows)
    if df_log.empty or "Article_ID" not in df_log.columns:
        return []

    latest = (
        df_log.sort_values(["Article_ID", "Batch_Pass", "Request_Order_Within_Pass"])
        .groupby("Article_ID", as_index=False)
        .tail(1)
    )

    if "API_Status" not in latest.columns:
        return []

    failed_ids = latest[latest["API_Status"] != "success"]["Article_ID"].dropna().astype(str).tolist()
    return failed_ids

def get_success_article_ids_from_raw():
    if not LOCAL_OUTPUT_RAW_JSONL.exists():
        return set()

    success_ids = set()

    with open(LOCAL_OUTPUT_RAW_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            article_id = safe_str(obj.get("article_id"))
            api_status = safe_str(obj.get("api_status"))

            if article_id and api_status == "success":
                success_ids.add(article_id)

    return success_ids

def get_last_successful_article_id_from_raw():
    if not LOCAL_OUTPUT_RAW_JSONL.exists():
        return ""

    last_success = ""

    with open(LOCAL_OUTPUT_RAW_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            article_id = safe_str(obj.get("article_id"))
            api_status = safe_str(obj.get("api_status"))

            if article_id and api_status == "success":
                last_success = article_id

    return last_success

def output_files_exist():
    return (
        LOCAL_OUTPUT_RAW_JSONL.exists() or
        LOCAL_OUTPUT_LOG_CSV.exists() or
        LOCAL_OUTPUT_API_ERRORS_JSONL.exists()
    )

def ask_resume_mode():
    print("\nDetected existing LOCAL step12 output files.")
    print("Choose run mode:")
    print("  1 = continue previous run")
    print("  2 = start fresh (overwrite previous run files)")

    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice == "1":
            return "continue"
        if choice == "2":
            return "fresh"
        print("Invalid choice. Please enter 1 or 2.")

def init_output_files_fresh():
    for path in [
        LOCAL_OUTPUT_RAW_JSONL,
        LOCAL_OUTPUT_API_ERRORS_JSONL,
        LOCAL_OUTPUT_LOG_CSV
    ]:
        if path.exists():
            path.unlink()

def prepare_resume_payloads(all_payloads):
    success_ids = get_success_article_ids_from_raw()
    last_success_id = get_last_successful_article_id_from_raw()

    remaining = []
    for payload in all_payloads:
        aid = safe_str(payload.get("article_id"))
        if not aid:
            remaining.append(payload)
            continue
        if aid in success_ids:
            continue
        remaining.append(payload)

    return {
        "remaining_payloads": remaining,
        "success_ids": success_ids,
        "last_success_id": last_success_id
    }

# =========================
# PASS EXECUTION
# =========================
def run_batch_pass(pass_no, payloads, system_instruction, log_rows):
    error_rows_this_pass = []

    print(f"\n=== BATCH PASS {pass_no}/{MAX_BATCH_PASSES} ===")
    print(f"Payloads in this pass: {len(payloads)}")

    total = len(payloads)

    for idx, record in enumerate(payloads, start=1):
        validate_payload_record(record)

        article_id = safe_str(record.get("article_id"))
        system_prompt_version = safe_str(record.get("system_prompt_version"))
        prompt_layer = safe_str(record.get("prompt_layer"))
        routing_reason = safe_str(record.get("routing_reason"))
        prellm_pro_review_support = record.get("prellm_pro_review_support", 0)
        user_payload = safe_str(record.get("user_payload"))

        print(f"[pass {pass_no} | {idx}/{total}] Sending Article_ID={article_id} | layer={prompt_layer}")

        result = submit_with_retry(
            system_instruction=system_instruction,
            user_payload=user_payload
        )

        api_status = result["api_status"]
        http_status = result["http_status"]
        response_json = result["response_json"]
        error_message = result["error_message"]
        attempt_count = result["attempt_count"]
        retried_flag = result["retried_flag"]

        response_text = ""
        if api_status == "success":
            response_text = extract_ollama_response_text(response_json)

        raw_record = {
            "article_id": article_id,
            "system_prompt_version": system_prompt_version,
            "model_family": f"ollama::{OLLAMA_MODEL}",
            "prompt_layer": prompt_layer,
            "routing_reason": routing_reason,
            "prellm_pro_review_support": prellm_pro_review_support,
            "api_status": api_status,
            "http_status": http_status,
            "response_text": response_text,
            "raw_response_json": response_json,
            "error_message": error_message,
            "attempt_count": attempt_count,
            "retried_flag": retried_flag,
            "batch_pass": pass_no
        }

        append_jsonl(LOCAL_OUTPUT_RAW_JSONL, raw_record)

        if api_status != "success":
            append_jsonl(LOCAL_OUTPUT_API_ERRORS_JSONL, raw_record)
            error_rows_this_pass.append(raw_record)

        log_rows.append({
            "Batch_Pass": pass_no,
            "Request_Order_Within_Pass": idx,
            "Article_ID": article_id,
            "System_Prompt_Version": system_prompt_version,
            "Model_Family": f"ollama::{OLLAMA_MODEL}",
            "Prompt_Layer": prompt_layer,
            "Routing_Reason": routing_reason,
            "PreLLM_Pro_Review_Support": prellm_pro_review_support,
            "API_Status": api_status,
            "HTTP_Status": http_status,
            "Attempt_Count": attempt_count,
            "Retried_Flag": retried_flag,
            "Response_Text_Length": response_text_length(response_text),
            "Error_Message": error_message
        })

        write_log_csv(log_rows)

        print_progress(idx, total, every=25, prefix=f"Pass {pass_no} progress")
        time.sleep(SLEEP_SECONDS)

    success_count = sum(1 for r in log_rows if safe_str(r.get("Batch_Pass")) == str(pass_no) and safe_str(r.get("API_Status")) == "success")
    error_count = sum(1 for r in log_rows if safe_str(r.get("Batch_Pass")) == str(pass_no) and safe_str(r.get("API_Status")) != "success")

    print(f"\nPass {pass_no} summary:")
    print(f"Success: {success_count}")
    print(f"Error: {error_count}")

    return error_rows_this_pass

# =========================
# MAIN
# =========================
def main():
    ensure_output_dirs()

    if COPY_PARENT_DATA_BEFORE_RUN:
        print("Copying parent pipeline data into LOCAL/data ...")
        copy_parent_data_to_local()

    system_instruction = load_text_file(LOCAL_SYSTEM_PROMPT_PATH)
    all_payloads = load_jsonl(LOCAL_INPUT_JSONL)

    print(f"Loaded {len(all_payloads)} payloads from {LOCAL_INPUT_JSONL}")
    print(f"Submitting to local Ollama model: {OLLAMA_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_URL}")
    print(f"Request-level retry config: MAX_RETRIES={MAX_RETRIES}, INITIAL_RETRY_SLEEP={INITIAL_RETRY_SLEEP}, BACKOFF_MULTIPLIER={BACKOFF_MULTIPLIER}, MAX_RETRY_SLEEP={MAX_RETRY_SLEEP}")
    print(f"Batch-level rerun config: MAX_BATCH_PASSES={MAX_BATCH_PASSES}, SLEEP_BETWEEN_PASSES={SLEEP_BETWEEN_PASSES}")
    print(f"Base spacing between requests: {SLEEP_SECONDS}s")

    run_mode = "fresh"
    if output_files_exist():
        run_mode = ask_resume_mode()

    if run_mode == "fresh":
        print("\nStarting fresh run. Previous run files will be overwritten.")
        init_output_files_fresh()
        log_rows = []
        current_payloads = all_payloads
    else:
        print("\nContinuing previous run.")
        log_rows = load_existing_log_rows()
        resume_info = prepare_resume_payloads(all_payloads)
        current_payloads = resume_info["remaining_payloads"]

        print(f"Previously successful article_ids: {len(resume_info['success_ids'])}")
        if resume_info["last_success_id"]:
            print(f"Last successful article_id: {resume_info['last_success_id']}")
        print(f"Remaining payloads to process: {len(current_payloads)}")

        if len(current_payloads) == 0:
            print("No remaining payloads. Everything already appears to be successfully processed.")
            return

    payload_map = {safe_str(p.get("article_id")): p for p in all_payloads}
    previous_failed_count = None

    for pass_no in range(1, MAX_BATCH_PASSES + 1):
        error_rows_this_pass = run_batch_pass(
            pass_no=pass_no,
            payloads=current_payloads,
            system_instruction=system_instruction,
            log_rows=log_rows
        )

        failed_ids = get_failed_article_ids_from_log(log_rows)
        success_ids = get_success_article_ids_from_raw()
        failed_ids = [aid for aid in failed_ids if aid not in success_ids]
        failed_count = len(failed_ids)

        print(f"Remaining failed article_ids after pass {pass_no}: {failed_count}")

        if failed_count == 0:
            print("All requests succeeded. Stopping batch reruns.")
            break

        if previous_failed_count is not None and failed_count >= previous_failed_count:
            print("Failure count did not improve. Stopping batch reruns.")
            break

        if pass_no == MAX_BATCH_PASSES:
            print("Reached maximum batch passes.")
            break

        previous_failed_count = failed_count
        current_payloads = [payload_map[aid] for aid in failed_ids if aid in payload_map]

        print(f"Waiting {SLEEP_BETWEEN_PASSES}s before next failed-only pass...")
        time.sleep(SLEEP_BETWEEN_PASSES)

    final_failed_ids = get_failed_article_ids_from_log(log_rows)
    final_success_ids = get_success_article_ids_from_raw()
    final_failed_ids = [aid for aid in final_failed_ids if aid not in final_success_ids]

    print(f"\nSaved raw outputs to {LOCAL_OUTPUT_RAW_JSONL}")
    print(f"Saved submission log to {LOCAL_OUTPUT_LOG_CSV}")
    print(f"Saved API errors to {LOCAL_OUTPUT_API_ERRORS_JSONL}")

    print(f"\nFinal batch summary:")
    print(f"Total payloads: {len(all_payloads)}")
    print(f"Final successful article_ids: {len(final_success_ids)}")
    print(f"Final failed article_ids: {len(final_failed_ids)}")
    print(f"Total log rows written (all passes combined): {len(log_rows)}")

if __name__ == "__main__":
    main()