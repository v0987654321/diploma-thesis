import os
import json
import time
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
API_KEY_PATH = "APIkey.txt"
SYSTEM_PROMPT_PATH = "prompts/gemini_system_instruction_v1.txt"
INPUT_JSONL = "data/gemini_batch_payloads.jsonl"

OUTPUT_RAW_JSONL = "data/gemini_raw_outputs.jsonl"
OUTPUT_LOG_XLSX = "data/gemini_submission_log.xlsx"
OUTPUT_API_ERRORS_JSONL = "data/gemini_api_errors.jsonl"

MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

REQUEST_TIMEOUT = 40

# Base spacing between requests
SLEEP_SECONDS = 3.0

# Retry / backoff
MAX_RETRIES = 40
INITIAL_RETRY_SLEEP = 5
BACKOFF_MULTIPLIER = 2
MAX_RETRY_SLEEP = 40

# Safety / reproducibility
OVERWRITE_OUTPUTS = True


# =========================
# HELPERS
# =========================
def safe_str(x):
    if x is None:
        return ""
    return str(x).strip()

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

def ensure_output_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)

def init_output_files():
    if OVERWRITE_OUTPUTS:
        for path in [OUTPUT_RAW_JSONL, OUTPUT_API_ERRORS_JSONL, OUTPUT_LOG_XLSX]:
            if os.path.exists(path):
                os.remove(path)

def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_response_text(response_json):
    """
    Best-effort extraction of model text content from Gemini response.
    Final strict parsing happens later in step13.
    """
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


# =========================
# MAIN
# =========================
def main():
    ensure_output_dirs()
    init_output_files()

    api_key = load_api_key(API_KEY_PATH)
    system_instruction = load_text_file(SYSTEM_PROMPT_PATH)
    payloads = load_jsonl(INPUT_JSONL)

    log_rows = []
    error_rows = []

    print(f"Loaded {len(payloads)} payloads from {INPUT_JSONL}")
    print(f"Submitting to model: {MODEL_NAME}")
    print(f"Retry config: MAX_RETRIES={MAX_RETRIES}, INITIAL_RETRY_SLEEP={INITIAL_RETRY_SLEEP}, BACKOFF_MULTIPLIER={BACKOFF_MULTIPLIER}, MAX_RETRY_SLEEP={MAX_RETRY_SLEEP}")
    print(f"Base spacing between requests: {SLEEP_SECONDS}s")

    for idx, record in enumerate(payloads, start=1):
        validate_payload_record(record)

        article_id = safe_str(record.get("article_id"))
        system_prompt_version = safe_str(record.get("system_prompt_version"))
        model_family = safe_str(record.get("model_family"))
        prompt_layer = safe_str(record.get("prompt_layer"))
        routing_reason = safe_str(record.get("routing_reason"))
        prellm_pro_review_support = record.get("prellm_pro_review_support", 0)
        user_payload = safe_str(record.get("user_payload"))

        print(f"[{idx}/{len(payloads)}] Sending Article_ID={article_id} | layer={prompt_layer}")

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
            "retried_flag": retried_flag
        }

        append_jsonl(OUTPUT_RAW_JSONL, raw_record)

        if api_status != "success":
            append_jsonl(OUTPUT_API_ERRORS_JSONL, raw_record)
            error_rows.append(raw_record)

        log_rows.append({
            "Request_Order": idx,
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

        time.sleep(SLEEP_SECONDS)

    df_log = pd.DataFrame(log_rows)
    df_log.to_excel(OUTPUT_LOG_XLSX, index=False)

    print(f"\nSaved raw outputs to {OUTPUT_RAW_JSONL}")
    print(f"Saved submission log to {OUTPUT_LOG_XLSX}")

    if error_rows:
        print(f"Saved API errors to {OUTPUT_API_ERRORS_JSONL}")
        print(f"API errors count: {len(error_rows)}")
    else:
        print("No API errors recorded.")

    success_count = sum(1 for r in log_rows if r["API_Status"] == "success")
    error_count = len(log_rows) - success_count
    retried_count = sum(1 for r in log_rows if r["Retried_Flag"] == 1)

    print(f"Success: {success_count}")
    print(f"Error: {error_count}")
    print(f"Retried: {retried_count}")


if __name__ == "__main__":
    main()