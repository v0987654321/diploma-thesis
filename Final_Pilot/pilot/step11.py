# ====================
# SOUBOR: step11.py
# ====================

import pandas as pd

INPUT_STEP1 = "data/postStep1.xlsx"
INPUT_STEP2 = "data/postStep2.xlsx"
INPUT_STEP3 = "data/postStep3.xlsx"
INPUT_STEP4 = "data/postStep4.xlsx"
INPUT_STEP5 = "data/postStep5.xlsx"
INPUT_STEP6 = "data/postStep6.xlsx"
INPUT_STEP7 = "data/postStep7.xlsx"
INPUT_STEP8 = "data/postStep8.xlsx"

OUTPUT_XLSX = "data/postDiagnostic.xlsx"
OUTPUT_CSV = "data/postDiagnostic.csv"


# =========================
# 1. HELPERS
# =========================
def normalize_id(series):
    return series.astype(str).str.extract(r"(\d+)")[0].str.zfill(6)


def ensure_review_col(df, col):
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(int)
    return df


def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df


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
    active = [col for col in review_cols if row.get(col, 0) == 1]
    return ", ".join(active) if active else ""


# =========================
# 2. LOAD DATA
# =========================
df1 = pd.read_excel(INPUT_STEP1, dtype={"article_id": str})
df2 = pd.read_excel(INPUT_STEP2, dtype={"article_id": str})
df3 = pd.read_excel(INPUT_STEP3, dtype={"Article_ID": str})
df4 = pd.read_excel(INPUT_STEP4, dtype={"Article_ID": str})
df5 = pd.read_excel(INPUT_STEP5, dtype={"Article_ID": str})
df6 = pd.read_excel(INPUT_STEP6, dtype={"Article_ID": str})
df7 = pd.read_excel(INPUT_STEP7, dtype={"Article_ID": str})
df8 = pd.read_excel(INPUT_STEP8, dtype={"Article_ID": str})

# normalize ids
df1["article_id"] = normalize_id(df1["article_id"])
df2["article_id"] = normalize_id(df2["article_id"])
for df in [df3, df4, df5, df6, df7, df8]:
    df["Article_ID"] = normalize_id(df["Article_ID"])


# =========================
# 3. BASE FROM STEP 1
# =========================
step1_keep = [
    "article_id",
    "outlet",
    "url",
    "date_iso_full",
    "date_year",
    "date_month",
    "date_day",
    "date_precision",
    "headline_clean",
]

df1 = ensure_columns(df1, step1_keep)

df_base = df1[step1_keep].copy()

df_base = df_base.rename(columns={
    "article_id": "Article_ID",
    "outlet": "Outlet",
    "url": "URL",
    "date_iso_full": "Date",
    "date_year": "Date_Year",
    "date_month": "Date_Month",
    "date_day": "Date_Day",
    "date_precision": "Date_Precision",
    "headline_clean": "Headline",
})


# =========================
# 4. STEP 2 FULL DIAGNOSTIC MERGE
# =========================
step2_keep = [
    "article_id",
    "target_hits_headline",
    "target_hits_lead",
    "target_hits_body",
    "target_hits_total",
    "target_types_found",
    "mali_hits_headline",
    "mali_hits_lead",
    "mali_hits_body",
    "mali_hits_total",
    "mali_terms_found",
    "strong_mali_focus_hits",
    "non_mali_hits_total",
    "relevance_score",
    "relevance_code",
    "relevance_label",
    "needs_manual_review",
    "relevance_notes",
]

df2 = ensure_columns(df2, step2_keep)

df2_sub = df2[step2_keep].copy()
df2_sub = df2_sub.rename(columns={
    "article_id": "Article_ID",
    "relevance_code": "Relevance",
    "relevance_label": "Relevance_Label",
    "needs_manual_review": "Step2_Manual_Review",
    "relevance_notes": "Relevance_Note",
})

df = df_base.merge(df2_sub, on="Article_ID", how="left")


# =========================
# 5. STEP 3 MERGE
# =========================
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
# current real export version: no per-frame *_Note columns
# =========================
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
    df = ensure_review_col(df, col)

df["Review_Flag_Count"] = df[review_cols].sum(axis=1)
df["Any_Review_Flag"] = (df["Review_Flag_Count"] > 0).astype(int)
df["Review_Sources"] = df.apply(collect_review_sources, axis=1)


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

    # Step 2
    "target_hits_headline",
    "target_hits_lead",
    "target_hits_body",
    "target_hits_total",
    "target_types_found",
    "mali_hits_headline",
    "mali_hits_lead",
    "mali_hits_body",
    "mali_hits_total",
    "mali_terms_found",
    "strong_mali_focus_hits",
    "non_mali_hits_total",
    "relevance_score",
    "Relevance",
    "Relevance_Label",
    "Relevance_Note",
    "Step2_Manual_Review",

    # Step 3
    "Actor_Mention",
    "Actor_Mention_Note",
    "Successor_Frame",
    "Successor_Frame_Note",
    "Dominant_Label",
    "Dominant_Label_Note",
    "Step3_Manual_Review",

    # Step 4
    "Dominant_Location",
    "Dominant_Location_Note",
    "Step4_Manual_Review",

    # Step 5
    "Main_Associated_Actor",
    "Main_Associated_Actor_Note",
    "Step5_Manual_Review",

    # Step 6
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

    # Step 7
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

    # Step 8
    "Dominant_Discourse_Support",
    "Dominant_Discourse_Support_Note",
    "Step8_Manual_Review",

    # Review meta
    "Review_Flag_Count",
    "Any_Review_Flag",
    "Review_Sources",
]

export_cols = [col for col in export_cols if col in df.columns]
df_out = df[export_cols].copy()


# =========================
# 13. EXPORT
# =========================
df_out.to_excel(OUTPUT_XLSX, index=False)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

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

print(f"\nSaved to {OUTPUT_XLSX}")
print(f"Saved to {OUTPUT_CSV}")