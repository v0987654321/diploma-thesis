import pandas as pd
import re

INPUT_STEP1 = "data/postStep1.csv"
INPUT_STEP2 = "data/postStep2.csv"
INPUT_STEP3 = "data/postStep3.csv"
INPUT_STEP4 = "data/postStep4.csv"
INPUT_STEP5 = "data/postStep5.csv"
INPUT_STEP6 = "data/postStep6.csv"
INPUT_STEP7 = "data/postStep7.csv"
INPUT_STEP8 = "data/postStep8.csv"

OUTPUT_CSV = "data/postConsolidated.csv"

# =========================
# 1. HELPERS
# =========================
def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

def safe_int(x, default=0):
    if pd.isna(x) or x is None:
        return default
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default

def normalize_id(series):
    return series.apply(normalize_article_id)

def normalize_article_id(x):
    raw = safe_str(x).strip()
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return digits.zfill(6)

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def print_progress(i, total, label="Progress", every=1):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"{label}: {i}/{total} ({pct}%)")

def collect_review_sources(row):
    review_cols = [
        "Step2_Manual_Review",
        "Step3_Manual_Review",
        "Step4_Manual_Review",
        "Step5_Manual_Review",
        "Step6_Manual_Review",
        "Step7_Manual_Review",
        "Step8_Manual_Review",
    ]
    active = [col for col in review_cols if safe_int(row.get(col)) == 1]
    return ", ".join(active) if active else ""

def build_full_text_for_llm(row):
    parts = []
    for col in ["Headline", "Lead", "Body_Postclean"]:
        val = row.get(col, None)
        if pd.notna(val) and safe_str(val).strip():
            parts.append(safe_str(val).strip())
    return "\n\n".join(parts)

def safe_merge(left_df, right_df, on_col, keep_cols, step_label):
    keep_cols = [c for c in keep_cols if c in right_df.columns]
    sub = right_df[keep_cols].copy()
    merged = left_df.merge(sub, on=on_col, how="left")
    print(f"{step_label}: merged {len(sub)} rows")
    return merged

# =========================
# 2. LOAD DATA
# =========================
def main():
    print("Loading step files...")

    df1 = pd.read_csv(INPUT_STEP1, dtype={"article_id": str}, low_memory=False)
    df2 = pd.read_csv(INPUT_STEP2, dtype={"article_id": str}, low_memory=False)
    df3 = pd.read_csv(INPUT_STEP3, dtype={"Article_ID": str}, low_memory=False)
    df4 = pd.read_csv(INPUT_STEP4, dtype={"Article_ID": str}, low_memory=False)
    df5 = pd.read_csv(INPUT_STEP5, dtype={"Article_ID": str}, low_memory=False)
    df6 = pd.read_csv(INPUT_STEP6, dtype={"Article_ID": str}, low_memory=False)
    df7 = pd.read_csv(INPUT_STEP7, dtype={"Article_ID": str}, low_memory=False)
    df8 = pd.read_csv(INPUT_STEP8, dtype={"Article_ID": str}, low_memory=False)

    # normalize ids
    print("Normalizing IDs...")
    df1["article_id"] = normalize_id(df1["article_id"])
    df2["article_id"] = normalize_id(df2["article_id"])

    for df in [df3, df4, df5, df6, df7, df8]:
        df["Article_ID"] = normalize_id(df["Article_ID"])

    # =========================
    # 3. STEP 1 BASE
    # =========================
    print("Preparing Step1 base...")

    step1_keep = [
        "article_id",
        "url",
        "outlet",
        "date_iso_full",
        "date_year",
        "date_month",
        "date_day",
        "date_precision",
        "headline_clean",
        "lead_clean",
        "body_postclean",
    ]
    df1 = ensure_columns(df1, step1_keep)

    df_base = df1[step1_keep].copy()

    df_base = df_base.rename(columns={
        "article_id": "Article_ID",
        "url": "URL",
        "outlet": "Outlet",
        "date_iso_full": "Date",
        "date_year": "Date_Year",
        "date_month": "Date_Month",
        "date_day": "Date_Day",
        "date_precision": "Date_Precision",
        "headline_clean": "Headline",
        "lead_clean": "Lead",
        "body_postclean": "Body_Postclean",
    })

    df = df_base.copy()
    print(f"Base rows: {len(df)}")

    # =========================
    # 4. STEP 2 MERGE
    # =========================
    print_progress(1, 7, label="Merge progress", every=1)

    step2_keep = [
        "article_id",
        "in_scope_period",
        "relevance_code",
        "relevance_label",
        "needs_manual_review",
        "relevance_notes",
        "target_hits_total",
        "target_types_found",
        "mali_hits_total",
        "non_mali_hits_total",
        "strong_mali_focus_hits",
        "mali_specific_linkage_hits",
        "generic_linkage_hits",
    ]
    df2 = ensure_columns(df2, step2_keep)
    df2_sub = df2[step2_keep].copy()

    df2_sub = df2_sub.rename(columns={
        "article_id": "Article_ID",
        "in_scope_period": "In_Scope_Period",
        "relevance_code": "Relevance",
        "relevance_label": "Relevance_Label",
        "needs_manual_review": "Step2_Manual_Review",
        "relevance_notes": "Relevance_Note",
    })

    df = df.merge(df2_sub, on="Article_ID", how="left")

    # =========================
    # 5. STEP 3 MERGE
    # =========================
    print_progress(2, 7, label="Merge progress", every=1)

    step3_keep = [
        "Article_ID",
        "Actor_Mention",
        "Actor_Mention_Note",
        "Successor_Frame",
        "Successor_Frame_Note",
        "Dominant_Label",
        "Dominant_Label_Note",
        "Step3_Manual_Review",
    ]
    df3 = ensure_columns(df3, step3_keep)
    df = df.merge(df3[step3_keep], on="Article_ID", how="left")

    # =========================
    # 6. STEP 4 MERGE
    # =========================
    print_progress(3, 7, label="Merge progress", every=1)

    step4_keep = [
        "Article_ID",
        "Dominant_Location",
        "Dominant_Location_Note",
        "Step4_Manual_Review",
    ]
    df4 = ensure_columns(df4, step4_keep)
    df = df.merge(df4[step4_keep], on="Article_ID", how="left")

    # =========================
    # 7. STEP 5 MERGE
    # =========================
    print_progress(4, 7, label="Merge progress", every=1)

    step5_keep = [
        "Article_ID",
        "Main_Associated_Actor",
        "Main_Associated_Actor_Note",
        "Step5_Manual_Review",
    ]
    df5 = ensure_columns(df5, step5_keep)
    df = df.merge(df5[step5_keep], on="Article_ID", how="left")

    # =========================
    # 8. STEP 6 MERGE
    # =========================
    print_progress(5, 7, label="Merge progress", every=1)

    step6_keep = [
        "Article_ID",
        "Counterterrorism",
        "Sovereignty",
        "Human_Rights_Abuse",
        "Anti_or_Neocolonialism",
        "Western_Failure",
        "Security_Effectiveness",
        "Economic_Interests",
        "Geopolitical_Rivalry",
        "Frame_Note",
        "Step6_Manual_Review",
    ]
    df6 = ensure_columns(df6, step6_keep)
    df = df.merge(df6[step6_keep], on="Article_ID", how="left")

    # =========================
    # 9. STEP 7 MERGE
    # =========================
    print_progress(6, 7, label="Merge progress", every=1)

    step7_keep = [
        "Article_ID",
        "Positive_Signal_Count",
        "Negative_Signal_Count",
        "Stance_Support",
        "Stance_Support_Note",
        "Ambivalence_Support",
        "Ambivalence_Support_Note",
        "Legitimation_Positive_Count",
        "Legitimation_Negative_Count",
        "Legitimation_Support",
        "Legitimation_Support_Note",
        "Step7_Manual_Review",
    ]
    df7 = ensure_columns(df7, step7_keep)
    df = df.merge(df7[step7_keep], on="Article_ID", how="left")

    # =========================
    # 10. STEP 8 MERGE
    # =========================
    print_progress(7, 7, label="Merge progress", every=1)

    step8_keep = [
        "Article_ID",
        "Dominant_Discourse_Support",
        "Dominant_Discourse_Support_Note",
        "Step8_Manual_Review",
    ]
    df8 = ensure_columns(df8, step8_keep)
    df = df.merge(df8[step8_keep], on="Article_ID", how="left")

    # =========================
    # 11. REVIEW META
    # =========================
    print("\nBuilding review metadata...")

    review_cols = [
        "Step2_Manual_Review",
        "Step3_Manual_Review",
        "Step4_Manual_Review",
        "Step5_Manual_Review",
        "Step6_Manual_Review",
        "Step7_Manual_Review",
        "Step8_Manual_Review",
    ]

    for col in review_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).apply(lambda x: safe_int(x))

    df["Review_Flag_Count"] = df[review_cols].sum(axis=1)
    df["Any_Review_Flag"] = (df["Review_Flag_Count"] > 0).astype(int)

    total = len(df)
    review_sources = []
    full_texts = []

    print("\nBuilding derived fields...")
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        review_sources.append(collect_review_sources(row))
        full_texts.append(build_full_text_for_llm(row))
        print_progress(i, total, label="Derived fields progress", every=200)

    df["Review_Sources"] = review_sources
    df["Full_Text_For_LLM"] = full_texts

    # =========================
    # 12. EXPORT ORDER
    # =========================
    export_cols = [
        "Article_ID",
        "Outlet",
        "Date",
        "Date_Year",
        "Date_Month",
        "Date_Day",
        "Date_Precision",
        "URL",

        "Headline",
        "Lead",
        "Body_Postclean",
        "Full_Text_For_LLM",

        "In_Scope_Period",
        "Relevance",
        "Relevance_Label",
        "Relevance_Note",

        "Actor_Mention",
        "Actor_Mention_Note",
        "Successor_Frame",
        "Successor_Frame_Note",
        "Dominant_Label",
        "Dominant_Label_Note",
        "Dominant_Location",
        "Dominant_Location_Note",
        "Main_Associated_Actor",
        "Main_Associated_Actor_Note",

        "Counterterrorism",
        "Sovereignty",
        "Human_Rights_Abuse",
        "Anti_or_Neocolonialism",
        "Western_Failure",
        "Security_Effectiveness",
        "Economic_Interests",
        "Geopolitical_Rivalry",
        "Frame_Note",

        "Positive_Signal_Count",
        "Negative_Signal_Count",
        "Stance_Support",
        "Stance_Support_Note",
        "Ambivalence_Support",
        "Ambivalence_Support_Note",
        "Legitimation_Positive_Count",
        "Legitimation_Negative_Count",
        "Legitimation_Support",
        "Legitimation_Support_Note",

        "Dominant_Discourse_Support",
        "Dominant_Discourse_Support_Note",

        "target_hits_total",
        "target_types_found",
        "mali_hits_total",
        "non_mali_hits_total",
        "strong_mali_focus_hits",
        "mali_specific_linkage_hits",
        "generic_linkage_hits",

        "Step2_Manual_Review",
        "Step3_Manual_Review",
        "Step4_Manual_Review",
        "Step5_Manual_Review",
        "Step6_Manual_Review",
        "Step7_Manual_Review",
        "Step8_Manual_Review",
        "Review_Flag_Count",
        "Any_Review_Flag",
        "Review_Sources",
    ]

    export_cols = [col for col in export_cols if col in df.columns]
    df_out = df[export_cols].copy()

    # =========================
    # 13. DIAGNOSTICS
    # =========================
    print("\nStep9 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    print(f"Rows with Relevance 1: {(df_out['Relevance'] == 1).sum() if 'Relevance' in df_out.columns else 'NA'}")
    print(f"Rows with Relevance 2: {(df_out['Relevance'] == 2).sum() if 'Relevance' in df_out.columns else 'NA'}")
    print(f"Rows with Relevance 3: {(df_out['Relevance'] == 3).sum() if 'Relevance' in df_out.columns else 'NA'}")
    print(f"Rows with Relevance 4: {(df_out['Relevance'] == 4).sum() if 'Relevance' in df_out.columns else 'NA'}")
    print(f"Any_Review_Flag=1: {(df_out['Any_Review_Flag'] == 1).sum() if 'Any_Review_Flag' in df_out.columns else 'NA'}")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Actor_Mention",
        "Dominant_Label",
        "Dominant_Location",
        "Main_Associated_Actor",
        "Stance_Support",
        "Legitimation_Support",
        "Dominant_Discourse_Support",
        "Any_Review_Flag",
        "Review_Flag_Count",
    ]
    preview_cols = [col for col in preview_cols if col in df_out.columns]
    print(df_out[preview_cols].head(30))

    # =========================
    # 14. EXPORT
    # =========================
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()