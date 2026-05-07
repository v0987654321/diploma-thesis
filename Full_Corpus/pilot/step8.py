import pandas as pd

INPUT_STEP7 = "data/postStep7.csv"
OUTPUT_CSV = "data/postStep8.csv"

# =========================
# 1. HELPERS
# =========================
def safe_val(x, default=0):
    if pd.isna(x):
        return default
    return x

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

def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x)

def ensure_columns(df, columns, fill_value=None):
    for col in columns:
        if col not in df.columns:
            df[col] = fill_value
    return df

def print_progress(i, total, every=100):
    if total <= 0:
        return
    if i == 1 or i == total or i % every == 0:
        pct = round((i / total) * 100, 1)
        print(f"Progress: {i}/{total} ({pct}%)")

# =========================
# 2. DISCOURSE CODING
# =========================
def code_dominant_discourse(row):
    counterterrorism = safe_int(row.get("Counterterrorism"))
    sovereignty = safe_int(row.get("Sovereignty"))
    hra = safe_int(row.get("Human_Rights_Abuse"))
    anti_neocol = safe_int(row.get("Anti_or_Neocolonialism"))
    western_failure = safe_int(row.get("Western_Failure"))
    sec_effect = safe_int(row.get("Security_Effectiveness"))
    econ = safe_int(row.get("Economic_Interests"))
    geo = safe_int(row.get("Geopolitical_Rivalry"))

    stance = safe_int(row.get("Stance_Support"))
    legit = safe_int(row.get("Legitimation_Support"))
    relevance = safe_int(row.get("Relevance"))

    scores = {
        1: (sovereignty * 2) + (anti_neocol * 2) + (western_failure * 1),    # sovereignty and emancipation
        2: (counterterrorism * 2) + (sec_effect * 2),                         # security and stabilization
        3: (hra * 3) + (1 if stance == 1 else 0) + (1 if legit == 1 else 0), # violence and abuse
        4: (geo * 2) + (western_failure * 1) + (econ * 1),                   # geopolitical competition
    }

    max_score = max(scores.values())

    if max_score == 0:
        return 5, f"no strong discourse score detected; coded as technocratic / factual reporting with scores={scores}"

    top_codes = [k for k, v in scores.items() if v == max_score]

    if len(top_codes) == 1:
        code = top_codes[0]
        note_map = {
            1: f"sovereignty and emancipation discourse dominates with scores={scores}",
            2: f"security and stabilization discourse dominates with scores={scores}",
            3: f"violence and abuse discourse dominates with scores={scores}",
            4: f"geopolitical competition discourse dominates with scores={scores}",
        }
        return code, note_map[code]

    # =========================
    # 3. TIE-BREAKING LOGIC
    # =========================

    # A. HRA + negative/delegitimizing support
    if 3 in top_codes and hra == 1 and (stance == 1 or legit == 1):
        return 3, f"tie resolved in favor of violence and abuse due to HRA + negative/delegitimizing support; scores={scores}"

    # B. Counterterrorism without geopolitics
    if 2 in top_codes and counterterrorism == 1 and geo == 0:
        return 2, f"tie resolved in favor of security and stabilization due to counterterrorism without geopolitical rivalry; scores={scores}"

    # C. Geopolitics reinforced
    if 4 in top_codes and geo == 1 and (western_failure == 1 or econ == 1):
        return 4, f"tie resolved in favor of geopolitical competition due to geopolitical frame reinforcement; scores={scores}"

    # D. Sovereignty + anti/neocolonialism
    if 1 in top_codes and sovereignty == 1 and anti_neocol == 1:
        return 1, f"tie resolved in favor of sovereignty and emancipation due to sovereignty + anti/neocolonial combination; scores={scores}"

    # E. Marginal relevance: avoid over-interpretation
    if relevance == 2:
        if hra == 1 and 3 in top_codes:
            return 3, f"marginal article tie resolved in favor of violence and abuse due to explicit HRA; scores={scores}"
        if counterterrorism == 1 and 2 in top_codes:
            return 2, f"marginal article tie resolved in favor of security and stabilization due to explicit counterterrorism; scores={scores}"
        if geo == 1 and 4 in top_codes:
            return 4, f"marginal article tie resolved in favor of geopolitical competition due to explicit geopolitical rivalry; scores={scores}"
        return 5, f"marginal article with no clear discourse dominance; coded as technocratic / factual reporting with scores={scores}"

    # F. Relevance 3/4: prefer explicit frame if present
    if relevance >= 3:
        if hra == 1 and 3 in top_codes:
            return 3, f"tie resolved in favor of violence and abuse in substantively relevant article; scores={scores}"
        if geo == 1 and 4 in top_codes and counterterrorism == 0 and sovereignty == 0:
            return 4, f"tie resolved in favor of geopolitical competition in substantively relevant article; scores={scores}"
        if counterterrorism == 1 and 2 in top_codes and geo == 0:
            return 2, f"tie resolved in favor of security and stabilization in substantively relevant article; scores={scores}"
        if sovereignty == 1 and 1 in top_codes and geo == 0:
            return 1, f"tie resolved in favor of sovereignty and emancipation in substantively relevant article; scores={scores}"

    return 6, f"multiple discourse scores tied at top with no clear tie-break resolution: top_codes={top_codes}, scores={scores}"

# =========================
# 4. REVIEW FLAG
# =========================
def step8_manual_review(row):
    discourse = safe_int(row.get("Dominant_Discourse_Support"))
    relevance = safe_int(row.get("Relevance"))

    if discourse == 6:
        return 1

    if relevance >= 3 and discourse == 5:
        return 1

    return 0

# =========================
# 5. ROW PROCESSOR
# =========================
def process_row(row):
    code, note = code_dominant_discourse(row)

    out = row.to_dict()
    out["Dominant_Discourse_Support"] = code
    out["Dominant_Discourse_Support_Note"] = note
    return out

# =========================
# 6. MAIN
# =========================
def main():
    df7 = pd.read_csv(
        INPUT_STEP7,
        dtype={"Article_ID": str},
        low_memory=False
    )

    df7 = ensure_columns(df7, [
        "Article_ID", "Outlet", "Date", "Headline",
        "Relevance", "Relevance_Label",
        "Actor_Mention", "Successor_Frame",
        "Dominant_Label", "Dominant_Location",
        "Main_Associated_Actor",
        "Counterterrorism", "Sovereignty", "Human_Rights_Abuse",
        "Anti_or_Neocolonialism", "Western_Failure",
        "Security_Effectiveness", "Economic_Interests",
        "Geopolitical_Rivalry",
        "Stance_Support", "Ambivalence_Support", "Legitimation_Support",
        "URL"
    ])

    total = len(df7)
    processed_rows = []

    print(f"\nStep8: processing {total} articles")

    for i, (_, row) in enumerate(df7.iterrows(), start=1):
        processed_rows.append(process_row(row))
        print_progress(i, total, every=100)

    df_processed = pd.DataFrame(processed_rows)

    df_processed["Step8_Manual_Review"] = df_processed.apply(step8_manual_review, axis=1)

    df_out = pd.DataFrame({
        "Article_ID": df_processed["Article_ID"],
        "Outlet": df_processed["Outlet"],
        "Date": df_processed["Date"],
        "Headline": df_processed["Headline"],
        "Relevance": df_processed["Relevance"],
        "Relevance_Label": df_processed["Relevance_Label"],
        "Actor_Mention": df_processed["Actor_Mention"],
        "Successor_Frame": df_processed["Successor_Frame"],
        "Dominant_Label": df_processed["Dominant_Label"],
        "Dominant_Location": df_processed["Dominant_Location"],
        "Main_Associated_Actor": df_processed["Main_Associated_Actor"],

        "Counterterrorism": df_processed["Counterterrorism"],
        "Sovereignty": df_processed["Sovereignty"],
        "Human_Rights_Abuse": df_processed["Human_Rights_Abuse"],
        "Anti_or_Neocolonialism": df_processed["Anti_or_Neocolonialism"],
        "Western_Failure": df_processed["Western_Failure"],
        "Security_Effectiveness": df_processed["Security_Effectiveness"],
        "Economic_Interests": df_processed["Economic_Interests"],
        "Geopolitical_Rivalry": df_processed["Geopolitical_Rivalry"],

        "Stance_Support": df_processed["Stance_Support"],
        "Ambivalence_Support": df_processed["Ambivalence_Support"],
        "Legitimation_Support": df_processed["Legitimation_Support"],

        "Dominant_Discourse_Support": df_processed["Dominant_Discourse_Support"],
        "Dominant_Discourse_Support_Note": df_processed["Dominant_Discourse_Support_Note"],
        "Step8_Manual_Review": df_processed["Step8_Manual_Review"],

        "URL": df_processed["URL"]
    })

    print("\nStep8 diagnostics:")
    print(f"Rows total: {len(df_out)}")
    print(f"Dominant_Discourse_Support=1: {(df_out['Dominant_Discourse_Support'] == 1).sum()}")
    print(f"Dominant_Discourse_Support=2: {(df_out['Dominant_Discourse_Support'] == 2).sum()}")
    print(f"Dominant_Discourse_Support=3: {(df_out['Dominant_Discourse_Support'] == 3).sum()}")
    print(f"Dominant_Discourse_Support=4: {(df_out['Dominant_Discourse_Support'] == 4).sum()}")
    print(f"Dominant_Discourse_Support=5: {(df_out['Dominant_Discourse_Support'] == 5).sum()}")
    print(f"Dominant_Discourse_Support=6: {(df_out['Dominant_Discourse_Support'] == 6).sum()}")
    print(f"Manual review: {df_out['Step8_Manual_Review'].sum()}")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Dominant_Discourse_Support",
        "Step8_Manual_Review"
    ]
    print(df_out[preview_cols].head(30))

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()