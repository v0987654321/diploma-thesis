import os
import json
import re
import pandas as pd

INPUT_PATH = "data/postConsolidated.xlsx"
OUTPUT_JSONL = "data/gemini_batch_payloads.jsonl"
OUTPUT_XLSX = "data/gemini_batch_payloads.xlsx"

SYSTEM_PROMPT_VERSION = "gemini_system_instruction_v1"
MODEL_FAMILY = "gemini-2.5-flash"

# =========================
# 1. HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def safe_code_str(x):
    if pd.isna(x) or x is None:
        return ""
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
    except Exception:
        pass
    return str(x).strip()

def split_sentences(text):
    if not text:
        return []
    text = re.sub(r"\s+", " ", str(text)).strip()
    return [s.strip() for s in re.split(r'(?<=[\.\!\?])\s+', text) if s.strip()]

def has_any(text, patterns):
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False

def rough_token_estimate(text):
    if not text:
        return 0
    return max(1, int(len(text) / 4))

# =========================
# 2. TARGET PATTERNS
# =========================
WGAC_PATTERNS = [
    r"\bwagner\b",
    r"\bgroupe\s+wagner\b",
    r"\bwagner\s+group\b",
    r"\bafrica\s+corps\b",
    r"\bcorps\s+africain\b",
    r"\bcorps\s+africain\s+russe\b",
    r"\bmercenaires?\s+russes?\b",
    r"\bparamilitaires?\s+russes?\b",
    r"\binstructeurs?\s+russes?\b",
    r"\bformateurs?\s+russes?\b",
    r"\bcoop[ée]rants?\s+russes?\b",
]

FRAME_COLS = [
    "Counterterrorism",
    "Sovereignty",
    "Human_Rights_Abuse",
    "Anti_or_Neocolonialism",
    "Western_Failure",
    "Security_Effectiveness",
    "Economic_Interests",
    "Geopolitical_Rivalry",
]

# =========================
# 3. TARGET CONTEXT EXCERPT
# =========================
def build_target_context_excerpt(headline, lead, body, window=2, max_sentences=10):
    parts = [safe_str(headline), safe_str(lead), safe_str(body)]
    full_text = " ".join([p for p in parts if p]).strip()

    sentences = split_sentences(full_text)
    if not sentences:
        return ""

    selected = set()
    for i, sent in enumerate(sentences):
        if has_any(sent, WGAC_PATTERNS):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            for j in range(start, end):
                selected.add(j)

    if not selected:
        excerpt = sentences[:max_sentences]
    else:
        excerpt = [sentences[i] for i in sorted(selected)][:max_sentences]

    return " ".join(excerpt).strip()

# =========================
# 4. ROUTING
# =========================
def get_active_frames(row):
    active = []
    for col in FRAME_COLS:
        val = row.get(col, None)
        if pd.notna(val):
            try:
                if int(val) == 1:
                    active.append(col)
            except Exception:
                pass
    return active

def should_send_to_llm(row):
    if pd.isna(row.get("Relevance")):
        return 0, "missing_relevance"

    relevance = int(row["Relevance"])

    if relevance == 1:
        return 0, "skip_relevance_1"

    return 1, "send_relevance_2_3_4"

def choose_prompt_layer_and_reason(row):
    relevance = int(row["Relevance"]) if pd.notna(row.get("Relevance")) else None
    any_review = int(row["Any_Review_Flag"]) if pd.notna(row.get("Any_Review_Flag")) else 0
    review_count = int(row["Review_Flag_Count"]) if pd.notna(row.get("Review_Flag_Count")) else 0
    stance = int(row["Stance_Support"]) if pd.notna(row.get("Stance_Support")) else None
    legit = int(row["Legitimation_Support"]) if pd.notna(row.get("Legitimation_Support")) else None
    discourse = int(row["Dominant_Discourse_Support"]) if pd.notna(row.get("Dominant_Discourse_Support")) else None
    actor_mention = int(row["Actor_Mention"]) if pd.notna(row.get("Actor_Mention")) else None
    successor = int(row["Successor_Frame"]) if pd.notna(row.get("Successor_Frame")) else None
    dominant_label = int(row["Dominant_Label"]) if pd.notna(row.get("Dominant_Label")) else None
    assoc_actor = int(row["Main_Associated_Actor"]) if pd.notna(row.get("Main_Associated_Actor")) else None
    active_frames = get_active_frames(row)

    if any_review == 1:
        return "full", "any_review_flag"
    if review_count >= 2:
        return "full", "multiple_review_flags"
    if stance in [4, 5]:
        return "full", "unclear_or_mixed_stance"
    if legit == 4:
        return "full", "unclear_legitimation"
    if discourse == 6:
        return "full", "mixed_discourse"
    if relevance == 2 and int(row.get("Step2_Manual_Review", 0)) == 1:
        return "full", "borderline_relevance_2"
    if actor_mention == 2 and successor == 0:
        return "full", "africa_corps_without_successor_frame"
    if dominant_label == 6:
        return "full", "multiple_dominant_labels"
    if assoc_actor in [9, 10]:
        return "full", "unclear_associated_actor"
    if len(active_frames) >= 4:
        return "full", "many_active_frames"

    return "light", "clearer_case"

def build_pro_review_support_flag(row, layer, routing_reason):
    """
    This is NOT the model's final Pro review recommendation.
    It is only a pre-LLM heuristic support indicator for later auditing.
    """
    if layer == "full":
        return 1

    difficult_reasons = {
        "mixed_discourse",
        "unclear_or_mixed_stance",
        "unclear_legitimation",
        "multiple_dominant_labels",
        "unclear_associated_actor",
        "africa_corps_without_successor_frame",
        "borderline_relevance_2",
        "many_active_frames",
    }

    if routing_reason in difficult_reasons:
        return 1

    return 0

# =========================
# 5. PIPELINE SUMMARIES
# =========================
def build_pipeline_summary_light(row):
    active_frames = get_active_frames(row)

    lines = [
        f"Relevance draft: {safe_code_str(row.get('Relevance'))}",
        f"Actor_Mention draft: {safe_code_str(row.get('Actor_Mention'))}",
        f"Successor_Frame draft: {safe_code_str(row.get('Successor_Frame'))}",
        f"Dominant_Label draft: {safe_code_str(row.get('Dominant_Label'))}",
        f"Dominant_Location draft: {safe_code_str(row.get('Dominant_Location'))}",
        f"Main_Associated_Actor draft: {safe_code_str(row.get('Main_Associated_Actor'))}",
        f"Active frames: {', '.join(active_frames) if active_frames else 'none'}",
        f"Stance support: {safe_code_str(row.get('Stance_Support'))}",
        f"Ambivalence support: {safe_code_str(row.get('Ambivalence_Support'))}",
        f"Legitimation support: {safe_code_str(row.get('Legitimation_Support'))}",
        f"Dominant_Discourse support: {safe_code_str(row.get('Dominant_Discourse_Support'))}",
    ]

    review_sources = safe_str(row.get("Review_Sources"))
    if review_sources:
        lines.append(f"Review sources: {review_sources}")

    return "\n".join(lines)

def build_pipeline_summary_full(row):
    active_frames = get_active_frames(row)

    lines = [
        f"In scope period: {safe_code_str(row.get('In_Scope_Period'))}",
        f"Target hits total: {safe_code_str(row.get('target_hits_total'))}",
        f"Target types found: {safe_str(row.get('target_types_found'))}",
        f"Mali hits total: {safe_code_str(row.get('mali_hits_total'))}",
        f"Non-Mali hits total: {safe_code_str(row.get('non_mali_hits_total'))}",
        f"Strong Mali-focus hits: {safe_code_str(row.get('strong_mali_focus_hits'))}",
        f"Mali-specific linkage hits: {safe_code_str(row.get('mali_specific_linkage_hits'))}",
        f"Generic linkage hits: {safe_code_str(row.get('generic_linkage_hits'))}",
        f"Relevance draft: {safe_code_str(row.get('Relevance'))}",
        f"Actor_Mention draft: {safe_code_str(row.get('Actor_Mention'))}",
        f"Successor_Frame draft: {safe_code_str(row.get('Successor_Frame'))}",
        f"Dominant_Label draft: {safe_code_str(row.get('Dominant_Label'))}",
        f"Dominant_Location draft: {safe_code_str(row.get('Dominant_Location'))}",
        f"Main_Associated_Actor draft: {safe_code_str(row.get('Main_Associated_Actor'))}",
        f"Active frames: {', '.join(active_frames) if active_frames else 'none'}",
        f"Stance support: {safe_code_str(row.get('Stance_Support'))}",
        f"Ambivalence support: {safe_code_str(row.get('Ambivalence_Support'))}",
        f"Legitimation support: {safe_code_str(row.get('Legitimation_Support'))}",
        f"Dominant_Discourse support: {safe_code_str(row.get('Dominant_Discourse_Support'))}",
    ]

    key_notes = [
        ("Relevance note", row.get("Relevance_Note")),
        ("Dominant_Label note", row.get("Dominant_Label_Note")),
        ("Main_Associated_Actor note", row.get("Main_Associated_Actor_Note")),
        ("Stance support note", row.get("Stance_Support_Note")),
        ("Legitimation support note", row.get("Legitimation_Support_Note")),
        ("Dominant_Discourse support note", row.get("Dominant_Discourse_Support_Note")),
    ]

    for label, value in key_notes:
        v = safe_str(value)
        if v:
            lines.append(f"{label}: {v}")

    review_sources = safe_str(row.get("Review_Sources"))
    if review_sources:
        lines.append(f"Review sources: {review_sources}")

    return "\n".join(lines)

# =========================
# 6. USER PAYLOAD
# =========================
def build_user_payload(row, layer, routing_reason, target_excerpt, pipeline_summary):
    article_id = safe_str(row.get("Article_ID"))
    outlet = safe_str(row.get("Outlet"))
    date = safe_str(row.get("Date"))
    headline = safe_str(row.get("Headline"))
    lead = safe_str(row.get("Lead"))
    body = safe_str(row.get("Body_Postclean"))

    payload = f"""
ARTICLE METADATA
- Article_ID: {article_id}
- Outlet: {outlet}
- Date: {date}
- Model_Family: {MODEL_FAMILY}
- Prompt_Layer: {layer}
- Routing_Reason: {routing_reason}

TARGET CONTEXT EXCERPT
{target_excerpt}

FULL ARTICLE
Headline:
{headline}

Lead:
{lead}

Body:
{body}

PIPELINE SUMMARY
{pipeline_summary}
""".strip()

    return payload

# =========================
# 7. MAIN
# =========================
def main():
    os.makedirs("data", exist_ok=True)

    df = pd.read_excel(INPUT_PATH, dtype={"Article_ID": str})

    records = []
    preview_rows = []

    for _, row in df.iterrows():
        article_id = safe_str(row.get("Article_ID"))

        send_to_llm, send_reason = should_send_to_llm(row)

        target_excerpt = build_target_context_excerpt(
            row.get("Headline"),
            row.get("Lead"),
            row.get("Body_Postclean")
        )

        if send_to_llm == 0:
            # Keep in preview sheet for audit, but do not include in JSONL batch file.
            preview_rows.append({
                "Article_ID": article_id,
                "Outlet": safe_str(row.get("Outlet")),
                "Date": safe_str(row.get("Date")),
                "Headline": safe_str(row.get("Headline")),
                "Relevance": safe_code_str(row.get("Relevance")),
                "Send_To_LLM": 0,
                "Send_Reason": send_reason,
                "Prompt_Layer": "skip",
                "Routing_Reason": "",
                "System_Prompt_Version": SYSTEM_PROMPT_VERSION,
                "Model_Family": MODEL_FAMILY,
                "PreLLM_Pro_Review_Support": 0,
                "Estimated_Chars": 0,
                "Estimated_Tokens_Rough": 0,
                "Target_Context_Excerpt": target_excerpt,
                "User_Payload": ""
            })
            continue

        layer, routing_reason = choose_prompt_layer_and_reason(row)

        if layer == "light":
            pipeline_summary = build_pipeline_summary_light(row)
        else:
            pipeline_summary = build_pipeline_summary_full(row)

        user_payload = build_user_payload(
            row=row,
            layer=layer,
            routing_reason=routing_reason,
            target_excerpt=target_excerpt,
            pipeline_summary=pipeline_summary
        )

        prellm_pro_support = build_pro_review_support_flag(row, layer, routing_reason)

        record = {
            "article_id": article_id,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
            "model_family": MODEL_FAMILY,
            "prompt_layer": layer,
            "routing_reason": routing_reason,
            "prellm_pro_review_support": prellm_pro_support,
            "user_payload": user_payload
        }
        records.append(record)

        preview_rows.append({
            "Article_ID": article_id,
            "Outlet": safe_str(row.get("Outlet")),
            "Date": safe_str(row.get("Date")),
            "Headline": safe_str(row.get("Headline")),
            "Relevance": safe_code_str(row.get("Relevance")),
            "Send_To_LLM": 1,
            "Send_Reason": send_reason,
            "Prompt_Layer": layer,
            "Routing_Reason": routing_reason,
            "System_Prompt_Version": SYSTEM_PROMPT_VERSION,
            "Model_Family": MODEL_FAMILY,
            "PreLLM_Pro_Review_Support": prellm_pro_support,
            "Estimated_Chars": len(user_payload),
            "Estimated_Tokens_Rough": rough_token_estimate(user_payload),
            "Target_Context_Excerpt": target_excerpt,
            "User_Payload": user_payload
        })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    df_preview = pd.DataFrame(preview_rows)
    df_preview.to_excel(OUTPUT_XLSX, index=False)

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Send_To_LLM",
        "Prompt_Layer",
        "Routing_Reason",
        "PreLLM_Pro_Review_Support",
        "Estimated_Tokens_Rough"
    ]
    preview_cols = [c for c in preview_cols if c in df_preview.columns]

    print(df_preview[preview_cols].head(30))
    print(f"\nSaved to {OUTPUT_JSONL}")
    print(f"Saved to {OUTPUT_XLSX}")

if __name__ == "__main__":
    main()