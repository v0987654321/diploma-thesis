import os
import re
import json
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PILOT_DIR = os.path.join(BASE_DIR, "pilot")
CODERS_DIR = os.path.join(BASE_DIR, "coders")
REL_DIR = os.path.join(BASE_DIR, "REL")
REL_DATA_DIR = os.path.join(REL_DIR, "data")

N_A_CODE = 99

POST_CONSOLIDATED = os.path.join(PILOT_DIR, "data", "postConsolidated.xlsx")
PIPELINE_DRAFT = os.path.join(PILOT_DIR, "data", "draft_coding_table_working.xlsx")

GEMINI_AUTH = os.path.join(PILOT_DIR, "GEMINI", "data", "final_llm_authoritative_table.xlsx")
GEMINI_CONS = os.path.join(PILOT_DIR, "GEMINI", "data", "final_conservative_adjudicated_table.xlsx")
GEMINI_HIGH = os.path.join(PILOT_DIR, "GEMINI", "data", "final_high_confidence_coding_table.xlsx")

LOCAL_AUTH = os.path.join(PILOT_DIR, "LOCAL", "data", "final_local_llm_authoritative_table.xlsx")
LOCAL_CONS = os.path.join(PILOT_DIR, "LOCAL", "data", "final_local_conservative_adjudicated_table.xlsx")
LOCAL_HIGH = os.path.join(PILOT_DIR, "LOCAL", "data", "final_local_high_confidence_coding_table.xlsx")

OUT_CODERS_LONG_XLSX = os.path.join(REL_DATA_DIR, "pilot_coders_long.xlsx")
OUT_CODERS_LONG_CSV = os.path.join(REL_DATA_DIR, "pilot_coders_long.csv")
OUT_CODERS_WIDE_XLSX = os.path.join(REL_DATA_DIR, "pilot_coders_wide.xlsx")

OUT_PIPELINE_LONG_XLSX = os.path.join(REL_DATA_DIR, "pilot_pipeline_long.xlsx")
OUT_PIPELINE_LONG_CSV = os.path.join(REL_DATA_DIR, "pilot_pipeline_long.csv")

OUT_BRANCHES_LONG_XLSX = os.path.join(REL_DATA_DIR, "pilot_llm_branches_long.xlsx")
OUT_BRANCHES_LONG_CSV = os.path.join(REL_DATA_DIR, "pilot_llm_branches_long.csv")

OUT_MASTER_LONG_XLSX = os.path.join(REL_DATA_DIR, "pilot_comparison_master_long.xlsx")
OUT_MASTER_LONG_CSV = os.path.join(REL_DATA_DIR, "pilot_comparison_master_long.csv")
OUT_MASTER_WIDE_XLSX = os.path.join(REL_DATA_DIR, "pilot_comparison_master_wide.xlsx")

OUT_SOURCE_INVENTORY_XLSX = os.path.join(REL_DATA_DIR, "pilot_source_inventory.xlsx")
OUT_SOURCE_INVENTORY_CSV = os.path.join(REL_DATA_DIR, "pilot_source_inventory.csv")

FINAL_VARS = [
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

PIPELINE_MAP = {
    "relevance": "V04_Relevance_Final",
    "actor_mention": "V05_Actor_Mention_Final",
    "successor_frame": "V06_Successor_Frame_Final",
    "dominant_label": "V07_Dominant_Label_Final",
    "stance": "V08_Stance_Final",
    "dominant_location": "V09_Dominant_Location_Final",
    "ambivalence": "V10_Ambivalence_Final",
    "legitimation": "V11_Legitimation_Final",
    "counterterrorism": "V12_Counterterrorism_Final",
    "sovereignty": "V13_Sovereignty_Final",
    "human_rights_abuse": "V14_Human_Rights_Abuse_Final",
    "anti_or_neocolonialism": "V15_Anti_or_Neocolonialism_Final",
    "western_failure": "V16_Western_Failure_Final",
    "security_effectiveness": "V17_Security_Effectiveness_Final",
    "economic_interests": "V18_Economic_Interests_Final",
    "geopolitical_rivalry": "V19_Geopolitical_Rivalry_Final",
    "main_associated_actor": "V20_Main_Associated_Actor_Final",
    "dominant_discourse": "V21_Dominant_Discourse_Final",
}

FINAL_CANONICAL_MAP = {v: v for v in FINAL_VARS}

CONSERVATIVE_MAP = {
    "Adj_V04_Relevance": "V04_Relevance_Final",
    "Adj_V05_Actor_Mention": "V05_Actor_Mention_Final",
    "Adj_V06_Successor_Frame": "V06_Successor_Frame_Final",
    "Adj_V07_Dominant_Label": "V07_Dominant_Label_Final",
    "Adj_V08_Stance": "V08_Stance_Final",
    "Adj_V09_Dominant_Location": "V09_Dominant_Location_Final",
    "Adj_V10_Ambivalence": "V10_Ambivalence_Final",
    "Adj_V11_Legitimation": "V11_Legitimation_Final",
    "Adj_V12_Counterterrorism": "V12_Counterterrorism_Final",
    "Adj_V13_Sovereignty": "V13_Sovereignty_Final",
    "Adj_V14_Human_Rights_Abuse": "V14_Human_Rights_Abuse_Final",
    "Adj_V15_Anti_or_Neocolonialism": "V15_Anti_or_Neocolonialism_Final",
    "Adj_V16_Western_Failure": "V16_Western_Failure_Final",
    "Adj_V17_Security_Effectiveness": "V17_Security_Effectiveness_Final",
    "Adj_V18_Economic_Interests": "V18_Economic_Interests_Final",
    "Adj_V19_Geopolitical_Rivalry": "V19_Geopolitical_Rivalry_Final",
    "Adj_V20_Main_Associated_Actor": "V20_Main_Associated_Actor_Final",
    "Adj_V21_Dominant_Discourse": "V21_Dominant_Discourse_Final",
}

def ensure_dirs():
    os.makedirs(REL_DIR, exist_ok=True)
    os.makedirs(REL_DATA_DIR, exist_ok=True)

def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def normalize_article_id(x):
    if pd.isna(x) or x is None:
        return None
    digits = re.sub(r"\D", "", str(x))
    if not digits:
        return None
    return digits.zfill(6)

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def try_parse_json_text(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("Could not parse JSON from text file.")

def force_list(obj):
    return obj if isinstance(obj, list) else [obj]

def coerce_int_or_none(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if pd.isna(x):
            return None
        if float(x).is_integer():
            return int(x)
        return None
    s = safe_str(x)
    if s == "" or s.lower() == "null":
        return None
    try:
        f = float(s)
        if f.is_integer():
            return int(f)
    except Exception:
        return None
    return None

def apply_explicit_na_coding(record):
    rel = coerce_int_or_none(record.get("V04_Relevance_Final"))
    if rel == 1:
        for var in FINAL_VARS:
            if var == "V04_Relevance_Final":
                continue
            val = record.get(var, None)
            if val is None or (isinstance(val, float) and pd.isna(val)) or safe_str(val) == "":
                record[var] = N_A_CODE
    return record

def canonical_empty_record():
    rec = {var: None for var in FINAL_VARS}
    rec["LLM_Review_Note"] = ""
    rec["Pro_Review_Candidate"] = None
    rec["Pro_Review_Reason"] = ""
    return rec

def list_txt_files(folder):
    if not os.path.exists(folder):
        return []
    return sorted([os.path.join(folder, fn) for fn in os.listdir(folder) if fn.lower().endswith(".txt")])

def flatten_coder_record(source_file, coder_type, coder_id, raw_record):
    article_id = normalize_article_id(raw_record.get("article_id"))
    parsed = raw_record.get("parsed_output", {}) or {}

    out = canonical_empty_record()
    out["Article_ID"] = normalize_article_id(parsed.get("Article_ID") or article_id)

    for var in FINAL_VARS:
        out[var] = coerce_int_or_none(parsed.get(var))

    out["LLM_Review_Note"] = safe_str(parsed.get("LLM_Review_Note"))
    out["Pro_Review_Candidate"] = coerce_int_or_none(parsed.get("Pro_Review_Candidate"))
    out["Pro_Review_Reason"] = safe_str(parsed.get("Pro_Review_Reason"))
    out = apply_explicit_na_coding(out)

    row = {
        "source_file": source_file,
        "source_group": "coders",
        "coder_type": coder_type,
        "coder_id": coder_id,
        "article_id": out["Article_ID"],
    }
    for k, v in out.items():
        if k != "Article_ID":
            row[k] = v
    return row

def read_coder_files():
    rows = []
    inventory = []

    for path in list_txt_files(CODERS_DIR):
        fn = os.path.basename(path)
        text = load_text(path)

        try:
            obj = try_parse_json_text(text)
            records = force_list(obj)
        except Exception as e:
            inventory.append({
                "source_file": fn,
                "source_group": "coders",
                "parse_status": "error",
                "record_count": 0,
                "error_detail": str(e),
            })
            continue

        for rec in records:
            human_coder = rec.get("human_coder")
            model_name = rec.get("model_name")

            if human_coder:
                coder_type = "human"
                coder_id = safe_str(human_coder)
            elif model_name:
                coder_type = "benchmark_model"
                coder_id = safe_str(model_name)
            else:
                lower_fn = fn.lower()
                coder_type = "human" if lower_fn.startswith("human") else "benchmark_model"
                coder_id = os.path.splitext(fn)[0]

            rows.append(flatten_coder_record(fn, coder_type, coder_id, rec))

        inventory.append({
            "source_file": fn,
            "source_group": "coders",
            "parse_status": "ok",
            "record_count": len(records),
            "error_detail": ""
        })

    return pd.DataFrame(rows), pd.DataFrame(inventory)

def add_article_metadata(df_long, df_cons):
    if df_long.empty or df_cons.empty:
        return df_long

    keep_cols = ["Article_ID", "Outlet", "Date", "Headline", "Relevance", "Relevance_Label", "Any_Review_Flag", "Review_Flag_Count", "Review_Sources"]
    keep_cols = [c for c in keep_cols if c in df_cons.columns]

    df_meta = df_cons[keep_cols].copy()
    df_meta["Article_ID"] = df_meta["Article_ID"].apply(normalize_article_id)

    return df_long.merge(df_meta, left_on="article_id", right_on="Article_ID", how="left").drop(columns=["Article_ID"], errors="ignore")

def extract_rows_from_mapped_table(path, source_group, coder_type, coder_id, column_map, extra_cols=None):
    inventory_rows = []

    if not os.path.exists(path):
        inventory_rows.append({
            "source_file": os.path.basename(path),
            "source_group": source_group,
            "parse_status": "missing",
            "record_count": 0,
            "error_detail": "file not found",
        })
        return pd.DataFrame(), pd.DataFrame(inventory_rows)

    df = pd.read_excel(path)

    id_col = "Article_ID" if "Article_ID" in df.columns else "article_id" if "article_id" in df.columns else None
    if id_col is None:
        inventory_rows.append({
            "source_file": os.path.basename(path),
            "source_group": source_group,
            "parse_status": "error",
            "record_count": 0,
            "error_detail": "missing Article_ID/article_id column",
        })
        return pd.DataFrame(), pd.DataFrame(inventory_rows)

    df[id_col] = df[id_col].apply(normalize_article_id)

    rows = []
    for _, r in df.iterrows():
        rec = canonical_empty_record()
        rec["Article_ID"] = r[id_col]

        for src_col, dst_col in column_map.items():
            if src_col in df.columns:
                rec[dst_col] = coerce_int_or_none(r.get(src_col))

        if "LLM_Review_Note" in df.columns:
            rec["LLM_Review_Note"] = safe_str(r.get("LLM_Review_Note"))
        if "Pro_Review_Candidate" in df.columns:
            rec["Pro_Review_Candidate"] = coerce_int_or_none(r.get("Pro_Review_Candidate"))
        if "Pro_Review_Reason" in df.columns:
            rec["Pro_Review_Reason"] = safe_str(r.get("Pro_Review_Reason"))

        rec = apply_explicit_na_coding(rec)

        row = {
            "source_file": os.path.basename(path),
            "source_group": source_group,
            "coder_type": coder_type,
            "coder_id": coder_id,
            "article_id": rec["Article_ID"],
        }

        for k, v in rec.items():
            if k != "Article_ID":
                row[k] = v

        if extra_cols:
            for extra_col in extra_cols:
                if extra_col in df.columns:
                    row[extra_col] = r.get(extra_col)

        rows.append(row)

    inventory_rows.append({
        "source_file": os.path.basename(path),
        "source_group": source_group,
        "parse_status": "ok",
        "record_count": len(rows),
        "error_detail": "",
    })

    return pd.DataFrame(rows), pd.DataFrame(inventory_rows)

def read_pipeline_draft():
    return extract_rows_from_mapped_table(
        PIPELINE_DRAFT, "pipeline", "pipeline", "local_pipeline_draft", PIPELINE_MAP
    )

def read_gemini_authoritative():
    return extract_rows_from_mapped_table(
        GEMINI_AUTH, "gemini_branch", "llm_operational", "gemini_authoritative",
        FINAL_CANONICAL_MAP,
        extra_cols=["Prompt_Layer", "Routing_Reason", "System_Prompt_Version", "Model_Family"]
    )

def read_gemini_conservative():
    return extract_rows_from_mapped_table(
        GEMINI_CONS, "gemini_branch", "llm_operational", "gemini_conservative",
        CONSERVATIVE_MAP,
        extra_cols=["Manual_Check_Required", "Manual_Check_Reasons"]
    )

def read_gemini_high():
    return extract_rows_from_mapped_table(
        GEMINI_HIGH, "gemini_branch", "high_confidence_branch", "gemini_high_confidence",
        CONSERVATIVE_MAP
    )

def read_local_authoritative():
    return extract_rows_from_mapped_table(
        LOCAL_AUTH, "local_branch", "llm_operational", "local_authoritative",
        FINAL_CANONICAL_MAP,
        extra_cols=["Prompt_Layer", "Routing_Reason", "System_Prompt_Version", "Model_Family"]
    )

def read_local_conservative():
    return extract_rows_from_mapped_table(
        LOCAL_CONS, "local_branch", "llm_operational", "local_conservative",
        CONSERVATIVE_MAP,
        extra_cols=["Manual_Check_Required", "Manual_Check_Reasons"]
    )

def read_local_high():
    return extract_rows_from_mapped_table(
        LOCAL_HIGH, "local_branch", "high_confidence_branch", "local_high_confidence",
        CONSERVATIVE_MAP
    )

def long_to_wide(df_long, prefix_from="coder_id"):
    if df_long.empty:
        return pd.DataFrame()

    index_cols = ["article_id"]
    meta_cols = [c for c in ["Outlet", "Date", "Headline"] if c in df_long.columns]
    base = df_long[index_cols + meta_cols].drop_duplicates()

    parts = [base]

    for var in FINAL_VARS:
        tmp = df_long[["article_id", prefix_from, var]].copy()
        tmp[prefix_from] = tmp[prefix_from].astype(str)
        wide = tmp.pivot_table(index="article_id", columns=prefix_from, values=var, aggfunc="first")
        wide.columns = [f"{var}__{c}" for c in wide.columns]
        wide = wide.reset_index()
        parts.append(wide)

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="article_id", how="left")
    return out

def main():
    ensure_dirs()

    if os.path.exists(POST_CONSOLIDATED):
        df_cons = pd.read_excel(POST_CONSOLIDATED, dtype={"Article_ID": str})
        df_cons["Article_ID"] = df_cons["Article_ID"].apply(normalize_article_id)
    else:
        df_cons = pd.DataFrame()

    df_coders, inv_coders = read_coder_files()
    df_coders = add_article_metadata(df_coders, df_cons)

    df_pipeline, inv_pipeline = read_pipeline_draft()
    df_pipeline = add_article_metadata(df_pipeline, df_cons)

    gem_auth, inv_gem_auth = read_gemini_authoritative()
    gem_cons, inv_gem_cons = read_gemini_conservative()
    gem_high, inv_gem_high = read_gemini_high()

    loc_auth, inv_loc_auth = read_local_authoritative()
    loc_cons, inv_loc_cons = read_local_conservative()
    loc_high, inv_loc_high = read_local_high()

    branch_frames = [df_ for df_ in [gem_auth, gem_cons, gem_high, loc_auth, loc_cons, loc_high] if not df_.empty]
    df_branches = pd.concat(branch_frames, ignore_index=True) if branch_frames else pd.DataFrame()
    df_branches = add_article_metadata(df_branches, df_cons)

    master_frames = [df_ for df_ in [df_coders, df_pipeline, df_branches] if not df_.empty]
    df_master_long = pd.concat(master_frames, ignore_index=True) if master_frames else pd.DataFrame()

    preferred_cols = [
        "source_group", "source_file", "coder_type", "coder_id", "article_id",
        "Outlet", "Date", "Headline", "Relevance", "Relevance_Label",
        *FINAL_VARS,
        "LLM_Review_Note", "Pro_Review_Candidate", "Pro_Review_Reason",
        "Any_Review_Flag", "Review_Flag_Count", "Review_Sources",
        "Manual_Check_Required", "Manual_Check_Reasons",
        "Prompt_Layer", "Routing_Reason", "System_Prompt_Version", "Model_Family"
    ]
    preferred_cols = [c for c in preferred_cols if c in df_master_long.columns]
    remaining_cols = [c for c in df_master_long.columns if c not in preferred_cols]
    df_master_long = df_master_long[preferred_cols + remaining_cols]

    df_coders_wide = long_to_wide(df_coders, prefix_from="coder_id")
    df_master_wide = long_to_wide(df_master_long, prefix_from="coder_id")

    df_inventory = pd.concat([
        inv_coders,
        inv_pipeline,
        inv_gem_auth,
        inv_gem_cons,
        inv_gem_high,
        inv_loc_auth,
        inv_loc_cons,
        inv_loc_high,
    ], ignore_index=True)

    df_coders.to_excel(OUT_CODERS_LONG_XLSX, index=False)
    df_coders.to_csv(OUT_CODERS_LONG_CSV, index=False, encoding="utf-8-sig")
    df_coders_wide.to_excel(OUT_CODERS_WIDE_XLSX, index=False)

    df_pipeline.to_excel(OUT_PIPELINE_LONG_XLSX, index=False)
    df_pipeline.to_csv(OUT_PIPELINE_LONG_CSV, index=False, encoding="utf-8-sig")

    df_branches.to_excel(OUT_BRANCHES_LONG_XLSX, index=False)
    df_branches.to_csv(OUT_BRANCHES_LONG_CSV, index=False, encoding="utf-8-sig")

    df_master_long.to_excel(OUT_MASTER_LONG_XLSX, index=False)
    df_master_long.to_csv(OUT_MASTER_LONG_CSV, index=False, encoding="utf-8-sig")
    df_master_wide.to_excel(OUT_MASTER_WIDE_XLSX, index=False)

    df_inventory.to_excel(OUT_SOURCE_INVENTORY_XLSX, index=False)
    df_inventory.to_csv(OUT_SOURCE_INVENTORY_CSV, index=False, encoding="utf-8-sig")

    print("\n=== REL PREPARATION SUMMARY ===")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"PILOT_DIR: {PILOT_DIR}")
    print(f"CODERS_DIR: {CODERS_DIR}")
    print(f"REL_DATA_DIR: {REL_DATA_DIR}")

    print(f"\nCoder rows: {len(df_coders)}")
    if not df_coders.empty:
        print(df_coders['coder_type'].value_counts(dropna=False))

    print(f"\nPipeline rows: {len(df_pipeline)}")
    print(f"Branch rows: {len(df_branches)}")
    print(f"Master long rows: {len(df_master_long)}")
    print(f"Inventory rows: {len(df_inventory)}")

    print(f"\nSaved:")
    print(f"- {OUT_CODERS_LONG_XLSX}")
    print(f"- {OUT_CODERS_WIDE_XLSX}")
    print(f"- {OUT_PIPELINE_LONG_XLSX}")
    print(f"- {OUT_BRANCHES_LONG_XLSX}")
    print(f"- {OUT_MASTER_LONG_XLSX}")
    print(f"- {OUT_MASTER_WIDE_XLSX}")
    print(f"- {OUT_SOURCE_INVENTORY_XLSX}")

if __name__ == "__main__":
    main()