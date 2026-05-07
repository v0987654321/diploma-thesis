import pandas as pd

INPUT_STEP8 = "data/postStep7.xlsx"
OUTPUT_XLSX = "data/postStep8.xlsx"
OUTPUT_CSV = "data/postStep8.csv"

# =========================
# 1. HELPERS
# =========================
def safe_val(x, default=0):
    if pd.isna(x):
        return default
    return x

# =========================
# 2. DISCOURSE CODING
# =========================
def code_dominant_discourse(row):
    # load frame values
    counterterrorism = safe_val(row["Counterterrorism"])
    sovereignty = safe_val(row["Sovereignty"])
    hra = safe_val(row["Human_Rights_Abuse"])
    anti_neocol = safe_val(row["Anti_or_Neocolonialism"])
    western_failure = safe_val(row["Western_Failure"])
    sec_effect = safe_val(row["Security_Effectiveness"])
    econ = safe_val(row["Economic_Interests"])
    geo = safe_val(row["Geopolitical_Rivalry"])

    # stance/legitimation support
    stance = safe_val(row["Stance_Support"])
    legit = safe_val(row["Legitimation_Support"])
    relevance = safe_val(row["Relevance"])

    # discourse scores
    scores = {
        1: (sovereignty * 2) + (anti_neocol * 2) + (western_failure * 1),   # sovereignty and emancipation
        2: (counterterrorism * 2) + (sec_effect * 2),                        # security and stabilization
        3: (hra * 3) + (1 if stance == 1 else 0) + (1 if legit == 1 else 0),# violence and abuse
        4: (geo * 2) + (western_failure * 1) + (econ * 1),                  # geopolitical competition
    }

    max_score = max(scores.values())

    # no strong discourse signals
    if max_score == 0:
        return 5, f"no strong discourse score detected; coded as technocratic / factual reporting with scores={scores}"

    top_codes = [k for k, v in scores.items() if v == max_score]

    # clear winner
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

    # A. If human-rights abuse is explicitly present and evaluative support is negative/delegitimizing,
    # prefer violence and abuse
    if 3 in top_codes and hra == 1 and (stance == 1 or legit == 1):
        return 3, f"tie resolved in favor of violence and abuse due to HRA + negative/delegitimizing support; scores={scores}"

    # B. If counterterrorism is present and geopolitical rivalry is absent, prefer security/stabilization
    if 2 in top_codes and counterterrorism == 1 and geo == 0:
        return 2, f"tie resolved in favor of security and stabilization due to counterterrorism without geopolitical rivalry; scores={scores}"

    # C. If geopolitical rivalry is present and western failure/economic interests reinforce it, prefer geopolitics
    if 4 in top_codes and geo == 1 and (western_failure == 1 or econ == 1):
        return 4, f"tie resolved in favor of geopolitical competition due to geopolitical frame reinforcement; scores={scores}"

    # D. If sovereignty and anti/neocolonialism co-occur, prefer sovereignty/emancipation
    if 1 in top_codes and sovereignty == 1 and anti_neocol == 1:
        return 1, f"tie resolved in favor of sovereignty and emancipation due to sovereignty + anti/neocolonial combination; scores={scores}"

    # E. If relevance is only marginal, avoid over-interpreting mixed signals
    if relevance == 2:
        # prefer most concrete frame if available
        if hra == 1 and 3 in top_codes:
            return 3, f"marginal article tie resolved in favor of violence and abuse due to explicit HRA; scores={scores}"
        if counterterrorism == 1 and 2 in top_codes:
            return 2, f"marginal article tie resolved in favor of security and stabilization due to explicit counterterrorism; scores={scores}"
        if geo == 1 and 4 in top_codes:
            return 4, f"marginal article tie resolved in favor of geopolitical competition due to explicit geopolitical rivalry; scores={scores}"
        return 5, f"marginal article with no clear discourse dominance; coded as technocratic / factual reporting with scores={scores}"

    # F. If relevance is 3 or 4 and one frame is explicit while others are only support-driven, prefer the explicit frame
    if relevance >= 3:
        if hra == 1 and 3 in top_codes:
            return 3, f"tie resolved in favor of violence and abuse in substantively relevant article; scores={scores}"
        if geo == 1 and 4 in top_codes and counterterrorism == 0 and sovereignty == 0:
            return 4, f"tie resolved in favor of geopolitical competition in substantively relevant article; scores={scores}"
        if counterterrorism == 1 and 2 in top_codes and geo == 0:
            return 2, f"tie resolved in favor of security and stabilization in substantively relevant article; scores={scores}"
        if sovereignty == 1 and 1 in top_codes and geo == 0:
            return 1, f"tie resolved in favor of sovereignty and emancipation in substantively relevant article; scores={scores}"

    # G. fallback to mixed only if no tie-break rule applies
    return 6, f"multiple discourse scores tied at top with no clear tie-break resolution: top_codes={top_codes}, scores={scores}"

# =========================
# 4. REVIEW FLAG
# =========================
def step8_manual_review(row):
    discourse = row["Dominant_Discourse_Support"]

    if discourse == 6:
        return 1

    # relevance >= 3 but still coded as technocratic factual → worth checking
    if row["Relevance"] >= 3 and discourse == 5:
        return 1

    return 0

# =========================
# 5. MAIN
# =========================
def main():
    df7 = pd.read_excel(
        INPUT_STEP8,
        dtype={"Article_ID": str}
    )

    # Dominant discourse support
    discourse_results = df7.apply(code_dominant_discourse, axis=1)
    df7["Dominant_Discourse_Support"] = discourse_results.apply(lambda x: x[0])
    df7["Dominant_Discourse_Support_Note"] = discourse_results.apply(lambda x: x[1])

    # review
    df7["Step8_Manual_Review"] = df7.apply(step8_manual_review, axis=1)

    # clean export
    df_out = pd.DataFrame({
        "Article_ID": df7["Article_ID"],
        "Outlet": df7["Outlet"],
        "Date": df7["Date"],
        "Headline": df7["Headline"],
        "Relevance": df7["Relevance"],
        "Relevance_Label": df7["Relevance_Label"],
        "Actor_Mention": df7["Actor_Mention"],
        "Successor_Frame": df7["Successor_Frame"],
        "Dominant_Label": df7["Dominant_Label"],
        "Dominant_Location": df7["Dominant_Location"],
        "Main_Associated_Actor": df7["Main_Associated_Actor"],

        "Counterterrorism": df7["Counterterrorism"],
        "Sovereignty": df7["Sovereignty"],
        "Human_Rights_Abuse": df7["Human_Rights_Abuse"],
        "Anti_or_Neocolonialism": df7["Anti_or_Neocolonialism"],
        "Western_Failure": df7["Western_Failure"],
        "Security_Effectiveness": df7["Security_Effectiveness"],
        "Economic_Interests": df7["Economic_Interests"],
        "Geopolitical_Rivalry": df7["Geopolitical_Rivalry"],

        "Stance_Support": df7["Stance_Support"],
        "Ambivalence_Support": df7["Ambivalence_Support"],
        "Legitimation_Support": df7["Legitimation_Support"],

        "Dominant_Discourse_Support": df7["Dominant_Discourse_Support"],
        "Dominant_Discourse_Support_Note": df7["Dominant_Discourse_Support_Note"],
        "Step8_Manual_Review": df7["Step8_Manual_Review"],

        "URL": df7["URL"]
    })

    df_out.to_excel(OUTPUT_XLSX, index=False)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    preview_cols = [
        "Article_ID",
        "Headline",
        "Relevance",
        "Dominant_Discourse_Support",
        "Step8_Manual_Review"
    ]

    print(df_out[preview_cols].head(30))
    print(f"\nSaved to {OUTPUT_XLSX}")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()