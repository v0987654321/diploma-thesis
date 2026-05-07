import os
import json
import pandas as pd

INPUT_XLSX = "data/postConsolidated.xlsx"

OUTPUT_CSV = "data/draft_coding_table.csv"
OUTPUT_XLSX = "data/draft_coding_table.xlsx"
OUTPUT_JSONL = "data/draft_coding_table.jsonl"

OUTPUT_WORK_CSV = "data/draft_coding_table_working.csv"
OUTPUT_WORK_XLSX = "data/draft_coding_table_working.xlsx"


# =========================
# HELPERS
# =========================
def safe_value(x):
    if pd.isna(x):
        return None
    return x


def normalize_article_id(x):
    if pd.isna(x):
        return None
    return str(x).strip().zfill(6)


# =========================
# MAIN
# =========================
def main():
    os.makedirs("data", exist_ok=True)

    df = pd.read_excel(INPUT_XLSX, dtype={"Article_ID": str})
    df["Article_ID"] = df["Article_ID"].apply(normalize_article_id)

    # =========================
    # 1. BUILD CLEAN DRAFT CODING TABLE
    # =========================
    df_out = pd.DataFrame({
        "article_id": df["Article_ID"].apply(safe_value),
        "outlet": df["Outlet"].apply(safe_value) if "Outlet" in df.columns else None,
        "date": df["Date"].apply(safe_value) if "Date" in df.columns else None,

        "relevance": df["Relevance"].apply(safe_value) if "Relevance" in df.columns else None,
        "actor_mention": df["Actor_Mention"].apply(safe_value) if "Actor_Mention" in df.columns else None,
        "successor_frame": df["Successor_Frame"].apply(safe_value) if "Successor_Frame" in df.columns else None,
        "dominant_label": df["Dominant_Label"].apply(safe_value) if "Dominant_Label" in df.columns else None,

        "stance": df["Stance_Support"].apply(safe_value) if "Stance_Support" in df.columns else None,
        "dominant_location": df["Dominant_Location"].apply(safe_value) if "Dominant_Location" in df.columns else None,
        "ambivalence": df["Ambivalence_Support"].apply(safe_value) if "Ambivalence_Support" in df.columns else None,
        "legitimation": df["Legitimation_Support"].apply(safe_value) if "Legitimation_Support" in df.columns else None,

        "counterterrorism": df["Counterterrorism"].apply(safe_value) if "Counterterrorism" in df.columns else None,
        "sovereignty": df["Sovereignty"].apply(safe_value) if "Sovereignty" in df.columns else None,
        "human_rights_abuse": df["Human_Rights_Abuse"].apply(safe_value) if "Human_Rights_Abuse" in df.columns else None,
        "anti_or_neocolonialism": df["Anti_or_Neocolonialism"].apply(safe_value) if "Anti_or_Neocolonialism" in df.columns else None,
        "western_failure": df["Western_Failure"].apply(safe_value) if "Western_Failure" in df.columns else None,
        "security_effectiveness": df["Security_Effectiveness"].apply(safe_value) if "Security_Effectiveness" in df.columns else None,
        "economic_interests": df["Economic_Interests"].apply(safe_value) if "Economic_Interests" in df.columns else None,
        "geopolitical_rivalry": df["Geopolitical_Rivalry"].apply(safe_value) if "Geopolitical_Rivalry" in df.columns else None,

        "main_associated_actor": df["Main_Associated_Actor"].apply(safe_value) if "Main_Associated_Actor" in df.columns else None,
        "dominant_discourse": df["Dominant_Discourse_Support"].apply(safe_value) if "Dominant_Discourse_Support" in df.columns else None,
    })

    # Optional: convert float-like codes (e.g. 2.0) to integers where possible
    code_cols = [
        "relevance",
        "actor_mention",
        "successor_frame",
        "dominant_label",
        "stance",
        "dominant_location",
        "ambivalence",
        "legitimation",
        "counterterrorism",
        "sovereignty",
        "human_rights_abuse",
        "anti_or_neocolonialism",
        "western_failure",
        "security_effectiveness",
        "economic_interests",
        "geopolitical_rivalry",
        "main_associated_actor",
        "dominant_discourse",
    ]

    for col in code_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda x: int(x) if pd.notna(x) and str(x).replace(".0", "").isdigit() else x
            )

    # =========================
    # 2. BUILD WORKING VERSION WITH REVIEW META
    # =========================
    df_work = df_out.copy()

    if "Any_Review_Flag" in df.columns:
        df_work["any_review_flag"] = df["Any_Review_Flag"].apply(safe_value)
    if "Review_Flag_Count" in df.columns:
        df_work["review_flag_count"] = df["Review_Flag_Count"].apply(safe_value)
    if "Review_Sources" in df.columns:
        df_work["review_sources"] = df["Review_Sources"].apply(safe_value)

    # =========================
    # 3. EXPORT CLEAN TABLE
    # =========================
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    df_out.to_excel(OUTPUT_XLSX, index=False)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df_out.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    # =========================
    # 4. EXPORT WORKING TABLE
    # =========================
    df_work.to_csv(OUTPUT_WORK_CSV, index=False, encoding="utf-8-sig")
    df_work.to_excel(OUTPUT_WORK_XLSX, index=False)

    # =========================
    # 5. PREVIEW
    # =========================
    print(df_out.head(20))
    print(f"\nSaved clean draft table to: {OUTPUT_CSV}")
    print(f"Saved clean draft table to: {OUTPUT_XLSX}")
    print(f"Saved clean draft table to: {OUTPUT_JSONL}")
    print(f"Saved working draft table to: {OUTPUT_WORK_CSV}")
    print(f"Saved working draft table to: {OUTPUT_WORK_XLSX}")


if __name__ == "__main__":
    main()