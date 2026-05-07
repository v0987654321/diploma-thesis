import os
import json
import time
import requests
import pandas as pd

# =========================
# CONFIG
# =========================
SYSTEM_PROMPT_PATH = "prompts/gemini_system_instruction_v1.txt"
INPUT_JSONL = "data/gemini_batch_payloads.jsonl"

OUTPUT_RAW_JSONL = "data/local_raw_outputs.jsonl"
OUTPUT_LOG_XLSX = "data/local_submission_log_v3.xlsx"
OUTPUT_API_ERRORS_JSONL = "data/local_api_errors_v3.jsonl"

OLLAMA_MODEL = "qwen2.5:14b-instruct"
OLLAMA_URL = "http://localhost:11434/api/chat"

REQUEST_TIMEOUT = 900
SLEEP_SECONDS = 1.0

MAX_RETRIES = 3
INITIAL_RETRY_SLEEP = 5
BACKOFF_MULTIPLIER = 2

OVERWRITE_OUTPUTS = True

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
    os.makedirs("data", exist_ok=True)
    os.makedirs("prompts", exist_ok=True)

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

def init_output_files():
    if OVERWRITE_OUTPUTS:
        for path in [OUTPUT_RAW_JSONL, OUTPUT_API_ERRORS_JSONL, OUTPUT_LOG_XLSX]:
            if os.path.exists(path):
                os.remove(path)

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
    """
    Extract the exact Article_ID from the payload metadata block.
    """
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
                time.sleep(retry_sleep)
                retry_sleep *= BACKOFF_MULTIPLIER
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
                time.sleep(retry_sleep)
                retry_sleep *= BACKOFF_MULTIPLIER
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

    system_instruction = load_text_file(SYSTEM_PROMPT_PATH)
    payloads = load_jsonl(INPUT_JSONL)

    log_rows = []
    error_rows = []

    print(f"Loaded {len(payloads)} payloads from {INPUT_JSONL}")
    print(f"Submitting to local Ollama model: {OLLAMA_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_URL}")

    for idx, record in enumerate(payloads, start=1):
        validate_payload_record(record)

        article_id = safe_str(record.get("article_id"))
        system_prompt_version = safe_str(record.get("system_prompt_version"))
        prompt_layer = safe_str(record.get("prompt_layer"))
        routing_reason = safe_str(record.get("routing_reason"))
        prellm_pro_review_support = record.get("prellm_pro_review_support", 0)
        user_payload = safe_str(record.get("user_payload"))

        print(f"[{idx}/{len(payloads)}] Sending Article_ID={article_id} | layer={prompt_layer}")

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
    print(f"Success: {success_count}")
    print(f"Error: {error_count}")


if __name__ == "__main__":
    import re
    main()