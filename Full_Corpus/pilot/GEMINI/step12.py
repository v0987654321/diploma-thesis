import os
import json
import time
import shutil
import requests
import pandas as pd
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
API_KEY_PATH = "APIkey.txt"
SYSTEM_PROMPT_PATH = "prompts/gemini_system_instruction_v1.txt"
INPUT_JSONL = "data/gemini_batch_payloads.jsonl"

OUTPUT_RAW_JSONL = "data/gemini_raw_outputs.jsonl"
OUTPUT_LOG_CSV = "data/gemini_submission_log.csv"
OUTPUT_API_ERRORS_JSONL = "data/gemini_api_errors.jsonl"

MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

REQUEST_TIMEOUT = 120
SLEEP_SECONDS = 3
1.0

# request-level retry
MAX_RETRIES = 40
INITIAL_RETRY_SLEEP = 5
BACKOFF_MULTIPLIER = 2
MAX_RETRY_SLEEP = 40

# batch-level rerun
MAX_BATCH_PASSES = 3
SLEEP_BETWEEN_PASSES = 20

COPY_PARENT_DATA_BEFORE_RUN = True
TEST_MODE_SAMPLE_SIZE = 4

# =========================
# PATH LOGIC
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

LOCAL_DATA_DIR = SCRIPT_DIR / "data"
LOCAL_PROMPTS_DIR = SCRIPT_DIR / "prompts"
PARENT_DATA_DIR = PARENT_DIR / "data"

LOCAL_API_KEY_PATH = SCRIPT_DIR / API_KEY_PATH
LOCAL_SYSTEM_PROMPT_PATH = SCRIPT_DIR / SYSTEM_PROMPT_PATH
LOCAL_INPUT_JSONL = SCRIPT_DIR / INPUT_JSONL

LOCAL_OUTPUT_RAW_JSONL = SCRIPT_DIR / OUTPUT_RAW_JSONL
LOCAL_OUTPUT_LOG_CSV = SCRIPT_DIR / OUTPUT_LOG_CSV
LOCAL_OUTPUT_API_ERRORS_JSONL = SCRIPT_DIR / OUTPUT_API_ERRORS_JSONL

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

    print(f"Copied {copied} items from parent data directory to local GEMINI/data")
    if skipped > 0:
        print(f"Skipped {skipped} non-standard items")

def load_api_key(path):
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError(f"API key file is empty: {path}")
    return key

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

def extract_response_text(response_json):
    try:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return ""

        text_chunks = []
        for part in parts:
            txt = part.get("text", "")
            if txt:
                text_chunks.append(txt)

        return "\n".join(text_chunks).strip()
    except Exception:
        return ""

def build_request_body(system_instruction, user_payload):
    return {
        "system_instruction": {
            "parts": [
                {
                    "text": system_instruction
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_payload
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0
        }
    }

def submit_request(api_key, system_instruction, user_payload):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    body = build_request_body(system_instruction, user_payload)

    response = requests.post(
        API_URL,
        headers=headers,
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

def is_retryable_status(http_status):
    return http_status in [408, 429, 500, 502, 503, 504]

def print_progress(i, total, every=25, prefix="Progress"):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"{prefix}: {i}/{total} ({pct}%)")

def submit_with_retry(api_key, system_instruction, user_payload):
    last_http_status = None
    last_response_json = {}
    last_error_message = ""
    attempt_count = 0
    retried_flag = 0

    retry_sleep = INITIAL_RETRY_SLEEP

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_count = attempt

        try:
            http_status, response_json = submit_request(
                api_key=api_key,
                system_instruction=system_instruction,
                user_payload=user_payload
            )

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

def ask_mode_with_existing_outputs():
    print("\nDetected existing GEMINI step12 output files.")
    print("Choose run mode:")
    print("  1 = continue previous run")
    print("  2 = start fresh (overwrite previous run files)")
    print("  3 = test mode (random 4 articles)")

    while True:
        choice = input("Enter choice [1/2/3]: ").strip()
        if choice == "1":
            return "continue"
        if choice == "2":
            return "fresh"
        if choice == "3":
            return "test"
        print("Invalid choice. Please enter 1, 2, or 3.")

def ask_mode_without_existing_outputs():
    print("\nChoose run mode:")
    print("  1 = full run")
    print("  2 = test mode (random 4 articles)")

    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice == "1":
            return "fresh"
        if choice == "2":
            return "test"
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

def build_test_payload_subset(all_payloads, sample_size=4):
    if len(all_payloads) <= sample_size:
        return all_payloads.copy()

    indices = sorted(random.sample(range(len(all_payloads)), sample_size))
    subset = [all_payloads[i] for i in indices]
    return subset

# =========================
# PASS EXECUTION
# =========================
def run_batch_pass(pass_no, payloads, api_key, system_instruction, log_rows):
    error_rows_this_pass = []

    print(f"\n=== BATCH PASS {pass_no}/{MAX_BATCH_PASSES} ===")
    print(f"Payloads in this pass: {len(payloads)}")

    total = len(payloads)

    for idx, record in enumerate(payloads, start=1):
        validate_payload_record(record)

        article_id = safe_str(record.get("article_id"))
        system_prompt_version = safe_str(record.get("system_prompt_version"))
        model_family = safe_str(record.get("model_family"))
        prompt_layer = safe_str(record.get("prompt_layer"))
        routing_reason = safe_str(record.get("routing_reason"))
        prellm_pro_review_support = record.get("prellm_pro_review_support", 0)
        user_payload = safe_str(record.get("user_payload"))

        print(f"[pass {pass_no} | {idx}/{total}] Sending Article_ID={article_id} | layer={prompt_layer}")

        result = submit_with_retry(
            api_key=api_key,
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
            response_text = extract_response_text(response_json)

        raw_record = {
            "article_id": article_id,
            "system_prompt_version": system_prompt_version,
            "model_family": model_family,
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
            "Model_Family": model_family,
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
        print("Copying parent pipeline data into GEMINI/data ...")
        copy_parent_data_to_local()

    api_key = load_api_key(LOCAL_API_KEY_PATH)
    system_instruction = load_text_file(LOCAL_SYSTEM_PROMPT_PATH)
    all_payloads = load_jsonl(LOCAL_INPUT_JSONL)

    print(f"Loaded {len(all_payloads)} payloads from {LOCAL_INPUT_JSONL}")
    print(f"Submitting to model: {MODEL_NAME}")
    print(f"Request-level retry config: MAX_RETRIES={MAX_RETRIES}, INITIAL_RETRY_SLEEP={INITIAL_RETRY_SLEEP}, BACKOFF_MULTIPLIER={BACKOFF_MULTIPLIER}, MAX_RETRY_SLEEP={MAX_RETRY_SLEEP}")
    print(f"Batch-level rerun config: MAX_BATCH_PASSES={MAX_BATCH_PASSES}, SLEEP_BETWEEN_PASSES={SLEEP_BETWEEN_PASSES}")
    print(f"Base spacing between requests: {SLEEP_SECONDS}s")

    selected_test_payloads = None

    if output_files_exist():
        run_mode = ask_mode_with_existing_outputs()
    else:
        run_mode = ask_mode_without_existing_outputs()

    if run_mode == "fresh":
        print("\nStarting fresh run. Previous run files will be overwritten.")
        init_output_files_fresh()
        log_rows = []
        current_payloads = all_payloads

    elif run_mode == "continue":
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

    elif run_mode == "test":
        print("\nStarting TEST MODE.")
        init_output_files_fresh()
        log_rows = []
        selected_test_payloads = build_test_payload_subset(all_payloads, sample_size=TEST_MODE_SAMPLE_SIZE)
        current_payloads = selected_test_payloads

        print(f"Random test sample size: {len(current_payloads)}")
        print("Selected article_ids:")
        for p in current_payloads:
            print(f"  - {safe_str(p.get('article_id'))}")

    else:
        raise ValueError(f"Unknown run mode: {run_mode}")

    payload_map = {safe_str(p.get("article_id")): p for p in all_payloads}
    previous_failed_count = None

    for pass_no in range(1, MAX_BATCH_PASSES + 1):
        error_rows_this_pass = run_batch_pass(
            pass_no=pass_no,
            payloads=current_payloads,
            api_key=api_key,
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
    print(f"Total payloads available in source batch: {len(all_payloads)}")
    print(f"Payloads actually attempted in this run mode: {len(log_rows)}")
    print(f"Final successful article_ids: {len(final_success_ids)}")
    print(f"Final failed article_ids: {len(final_failed_ids)}")
    print(f"Total log rows written (all passes combined): {len(log_rows)}")

    if run_mode == "test":
        selected_test_ids = [safe_str(p.get("article_id")) for p in selected_test_payloads] if selected_test_payloads else []
        selected_test_ids_set = set(selected_test_ids)
        successful_test_ids = sorted([aid for aid in final_success_ids if aid in selected_test_ids_set])
        failed_test_ids = sorted([aid for aid in final_failed_ids if aid in selected_test_ids_set])

        print("\nTEST MODE DIAGNOSTIC")
        print(f"Selected test payloads: {len(selected_test_ids)}")
        print(f"Successful test article_ids: {len(successful_test_ids)}")
        print(f"Failed test article_ids: {len(failed_test_ids)}")

        print("Selected test article_ids:")
        for aid in selected_test_ids:
            print(f"  - {aid}")

        if failed_test_ids:
            print("Failed test article_ids:")
            for aid in failed_test_ids:
                print(f"  - {aid}")

if __name__ == "__main__":
    main()