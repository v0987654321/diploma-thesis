README.txt
==========

Project:
Media Discourses of the Wagner Group in Malian Online Press
Diploma Thesis Replication Package

Author:
Bc. Petr Vrána

Supervisor:
Mgr. Martin Schmiedl, Ph.D.

Repository:
https://github.com/v0987654321/diploma-thesis

Methodological status:
This repository contains the computational workflow, scripts, prompts, intermediate outputs,
diagnostic files, and supplementary exploratory analyses used for the diploma thesis
"Media Discourses of the Wagner Group in Malian Online Press".

The repository is intended as a transparency and replication package. It documents how the
article corpus was cleaned, coded, validated, adjudicated, analysed, and prepared for both
quantitative content analysis and qualitative critical discourse analysis.


============================================================
1. THESIS AND WORKFLOW OVERVIEW
============================================================

This project studies how Wagner Group and Africa Corps were represented in selected Malian
online news outlets between July 2021 and December 2025.

The empirical workflow follows an explanatory sequential mixed-methods design:

1. Corpus construction and preprocessing
   - Raw article data are cleaned, normalized, deduplicated, and transformed into structured
     article-level records.
   - Article metadata, outlet identifiers, dates, headlines, leads, and body texts are
     standardized.

2. Rule-based and semi-automated quantitative coding
   - Python scripts identify target mentions, Mali-context signals, actor labels, locations,
     associated actors, thematic frames, and evaluative support indicators.
   - These steps create a transparent local pipeline draft.

3. Pilot validation and reliability assessment
   - A pilot set of articles is compared across human coders, benchmark LLM coders,
     the local rule-based pipeline, and operational adjudication branches.
   - Krippendorff's alpha, percent agreement, Cohen's kappa, and distance matrices are used
     to assess variable stability and output-layer suitability.

4. AI-assisted adjudication
   - Relevance-filtered articles are submitted to standardized LLM prompts.
   - Gemini and local Ollama branches are run separately.
   - Outputs are strictly parsed, schema-validated, value-validated, and merged back with
     the pipeline outputs.

5. Conservative adjudication
   - Direct LLM outputs are not treated as automatically authoritative.
   - Conservative fallback rules preserve pipeline values in selected risk cases and flag
     cases requiring manual verification.
   - The Gemini conservative adjudicated table is selected as the principal full-corpus
     analytical output because pilot evaluation showed the best balance between human-majority
     alignment and pipeline consistency.

6. Source and republication enrichment
   - Rule-based source analysis identifies explicit external sources, Malian media references,
     republication phrases, attribution phrases, and near-duplicate article clusters.
   - This layer is used diagnostically to interpret circulation and outlet-level differences.

7. Corpus-level quantitative analysis
   - Corpus overview, frame prevalence, actor representation, source/republication structure,
     outlet differences, and temporal patterns are exported as tables and figures.

8. Qualitative sample selection
   - A purposive candidate pool for critical discourse analysis is generated using coding,
     salience, transition, anomaly, outlet contrast, and source/republication criteria.

9. Supplementary exploratory analysis
   - KWIC, lexical normalization, word clouds, NER, BERTopic topic modelling, and synthesis
     tables are produced as supporting layers.
   - These outputs are exploratory and should not be treated as primary evidence without
     close interpretation.

10. Final chapter-oriented analysis
   - Final tables, figures, diagnostics, and CDA support files are produced for thesis
     Chapters 5 and 6.


============================================================
2. MAIN METHODOLOGICAL PRINCIPLES
============================================================

The workflow is grounded in quantitative content analysis and computer-aided text analysis.
It follows several principles:

- explicit variable definitions,
- predefined codebook categories,
- article-level unit of analysis,
- transparent preprocessing,
- rule-based baseline coding,
- pilot validation,
- variable-level reliability assessment,
- strict parsing and validation of model outputs,
- conservative adjudication rather than direct acceptance of raw LLM output,
- separation between primary coding outputs and supplementary exploratory outputs.

The workflow does not treat automated coding or LLM output as an autonomous source of truth.
The scripts operationalize a researcher-defined codebook and generate structured support for
analysis. Final interpretation remains the responsibility of the researcher.


============================================================
3. PYTHON ENVIRONMENT AND DEPENDENCIES
============================================================

Recommended Python version:
    Python 3.10+

Core external libraries:
    pandas
    numpy
    requests
    openpyxl
    dateparser
    rapidfuzz
    matplotlib
    seaborn
    scipy
    statsmodels
    scikit-learn
    krippendorff
    tqdm
    wordcloud

Optional libraries:
    spacy
    fr_core_news_sm or fr_core_news_md
    bertopic
    sentence-transformers
    umap-learn
    hdbscan

Standard-library modules used throughout:
    os
    re
    json
    time
    shutil
    random
    pathlib
    datetime
    math
    warnings
    itertools
    unicodedata
    collections
    argparse
    csv
    zipfile

Suggested base installation:

    pip install pandas numpy requests openpyxl dateparser rapidfuzz matplotlib seaborn scipy statsmodels scikit-learn krippendorff tqdm wordcloud

Optional installation for NER and topic modelling:

    pip install spacy bertopic sentence-transformers umap-learn hdbscan
    python -m spacy download fr_core_news_sm

For local LLM inference:
    Ollama must be installed and running.
    The local scripts expect:
        qwen2.5:14b-instruct

For Gemini API inference:
    A Google Gemini API key is required in:
        APIkey.txt


============================================================
4. FOLDER STRUCTURE
============================================================

The repository contains two main workspaces:

1. Final_Pilot/
   Pilot workflow and reliability validation.

2. Full_Corpus/
   Full corpus workflow and final thesis analysis.

Main folders:

Final_Pilot/
    pilot/
        step1.py ... step11.py
        stepA.py
        stepB.py
        GEMINI/
            step12.py
            step13.py
        local/
            step12.py
            step13.py
        data/
        prompts/

    RELIABILITY/
        pilot1.py
        pilot2.py
        pilot3.py
        heatmaps.py
        DBplot.py

Full_Corpus/
    pilot/
        step1.py ... step11.py
        step10.py
        stepB.py
        stepBA.py
        stepC.py
        stepD.py
        GEMINI/
            step12.py
            step13.py
        local/
            step12.py
            step13.py
        data/
        prompts/

    CORPUS/
        corpus1.py ... corpus8.py
        synthesis.py
        data/
        tables/
        figures/
        summary/
        subsets/
        evidence/
        concordance/
        lexical/
        lexical_norm/
        topics/
        ner/
        synthesis/

    ANA/
        ANALYSIS.py


============================================================
5. IMPORTANT PATH NOTE
============================================================

Most scripts use relative paths. Scripts should normally be run from their own workspace.

Examples:

Full corpus pipeline:
    cd Full_Corpus/pilot
    python step1.py

Gemini branch:
    cd Full_Corpus/pilot/GEMINI
    python step12.py
    python step13.py

Corpus outputs:
    cd Full_Corpus/CORPUS
    python corpus1.py

Final chapter analysis:
    cd Full_Corpus/ANA
    python ANALYSIS.py --all

If a file is missing, first check:
    1. whether the previous step has been run,
    2. whether you are in the correct directory,
    3. whether the expected input file exists in the local data/ folder.


============================================================
6. FULL CORPUS PIPELINE SCRIPTS
============================================================

Location:
    Full_Corpus/pilot/


------------------------------------------------------------
step1.py
------------------------------------------------------------

Purpose:
    Cleans raw article data and prepares the basic article-level text table.

Input:
    data/pilot.xlsx

Output:
    data/postStep1.csv

Methodology:
    - Normalizes article IDs to six-digit strings.
    - Derives outlet codes and outlet names.
    - Parses dates and records date precision.
    - Cleans HTML residues and generic boilerplate.
    - Deduplicates lead text from body text where the body repeats the lead.
    - Applies outlet-specific body cleaning rules.
    - Builds full cleaned article text.

Main variables produced:
    article_id
    outlet_code
    article_seq
    outlet
    date_iso_full
    date_year
    date_month
    date_day
    date_precision
    headline_clean
    lead_clean
    body_postclean
    full_text_postclean

Libraries:
    pandas:
        reading Excel, dataframe operations, export to CSV.
    re:
        regular expressions for cleaning, IDs, sentence splitting, and boilerplate removal.
    unicodedata:
        Unicode normalization.
    dateparser:
        parsing French and English date strings.


------------------------------------------------------------
step2.py
------------------------------------------------------------

Purpose:
    Performs relevance screening and target/Mali-context scoring.

Input:
    data/postStep1.csv

Output:
    data/postStep2.csv

Methodology:
    - Detects target actors and labels:
        Wagner,
        Groupe Wagner,
        Wagner Group,
        Africa Corps,
        Corps africain,
        Russian mercenaries,
        Russian instructors.
    - Detects Mali-context markers:
        Mali,
        Bamako,
        FAMa,
        Malian army,
        Assimi Goita,
        Kidal,
        Gao,
        Mopti,
        MINUSMA, etc.
    - Scores target centrality by segment:
        headline,
        lead,
        body.
    - Computes sentence-level target metrics:
        target sentence count,
        target sentence share,
        target cluster run,
        whether target appears in first third.
    - Detects bulletin-style articles.
    - Assigns relevance code:
        1 = not relevant,
        2 = marginal mention only,
        3 = substantively relevant,
        4 = main topic.
    - Flags possible manual review cases.

Libraries:
    pandas:
        dataframe operations and CSV export.
    re:
        regex-based target and context detection.


------------------------------------------------------------
step3.py
------------------------------------------------------------

Purpose:
    Codes actor mention, successor framing, and dominant label.

Inputs:
    data/postStep2.csv
    data/postStep1.csv

Output:
    data/postStep3.csv

Methodology:
    - Keeps only relevance 2, 3, and 4 articles.
    - Merges cleaned lead and body text from step1.
    - Codes Actor_Mention:
        1 = Wagner explicitly mentioned,
        2 = Africa Corps explicitly mentioned,
        3 = both explicitly mentioned,
        4 = indirect Russian military-contractor terminology,
        5 = cannot determine.
    - Codes Successor_Frame:
        whether Africa Corps is explicitly linked to Wagner as successor, replacement,
        continuation, or post-Wagner structure.
    - Codes Dominant_Label:
        mercenaries,
        instructors/advisers,
        allies/partners,
        foreign/occupying forces,
        neutral designation,
        multiple/no clear dominance.
    - Uses weighted local target-context scoring.

Libraries:
    pandas:
        input/output and merging.
    re:
        pattern matching, sentence splitting, local-context extraction.


------------------------------------------------------------
step4.py
------------------------------------------------------------

Purpose:
    Codes dominant location of referred Wagner/Africa Corps activity.

Inputs:
    data/postStep3.csv
    data/postStep1.csv

Output:
    data/postStep4.csv

Methodology:
    - Merges article text from step1.
    - Counts location markers for:
        Mali,
        other African countries,
        Ukraine,
        other locations.
    - Applies tie-breaking logic.
    - Codes mixed Mali + other location only when both are strongly present.
    - Flags fallback or tie cases for review.

Libraries:
    pandas:
        reading, merging, exporting.
    re:
        regex detection of location markers.


------------------------------------------------------------
step5.py
------------------------------------------------------------

Purpose:
    Codes the main associated actor.

Inputs:
    data/postStep4.csv
    data/postStep1.csv

Output:
    data/postStep5.csv

Methodology:
    - Identifies the main actor most closely associated with Wagner/Africa Corps.
    - Actor categories:
        1 = Malian army / junta,
        2 = Russia / Russian state,
        3 = France,
        4 = UN / MINUSMA,
        5 = ECOWAS / regional actors,
        6 = local civilians,
        7 = jihadist / terrorist groups,
        8 = Western states broadly,
        9 = no clear dominant actor,
        10 = other.
    - Uses segment weighting:
        headline > lead > body.
    - Adds proximity bonuses when associated actor appears in the same or nearby sentence as WG/AC terms.
    - Flags unclear, tied, or broad-Western associated-actor cases.

Libraries:
    pandas:
        dataframe handling.
    re:
        actor-pattern detection and sentence-level proximity logic.


------------------------------------------------------------
step6.py
------------------------------------------------------------

Purpose:
    Codes thematic frame variables.

Inputs:
    data/postStep5.csv
    data/postStep1.csv

Output:
    data/postStep6.csv

Methodology:
    Codes eight binary frame variables:
        Counterterrorism
        Sovereignty
        Human_Rights_Abuse
        Anti_or_Neocolonialism
        Western_Failure
        Security_Effectiveness
        Economic_Interests
        Geopolitical_Rivalry

    - Uses headline, lead, and local WG/AC context.
    - Applies frame-specific thresholds.
    - Requires core terms for some frames:
        anti/neocolonialism,
        geopolitical rivalry,
        economic interests.
    - Suppresses over-interpretation in marginal and bulletin-style articles.
    - Flags cases with no frames in relevant articles or too many active frames.

Libraries:
    pandas:
        dataframe operations.
    re:
        frame lexicon matching and sentence extraction.


------------------------------------------------------------
step7.py
------------------------------------------------------------

Purpose:
    Generates evaluative support indicators for stance, ambivalence, and legitimation.

Inputs:
    data/postStep6.csv
    data/postStep1.csv

Output:
    data/postStep7.csv

Methodology:
    - Extracts local WG/AC context.
    - Counts positive and negative stance signals.
    - Uses hard and soft negative lexicons.
    - Downweights reported or quoted claims.
    - Codes Stance_Support:
        1 = negative,
        2 = neutral,
        3 = positive,
        4 = mixed/ambivalent,
        5 = cannot determine.
    - Codes Ambivalence_Support:
        0 = no strong ambivalence,
        1 = positive and negative signals both present.
    - Codes Legitimation_Support:
        1 = delegitimized,
        2 = normalized / implicitly legitimized,
        3 = explicitly legitimized,
        4 = cannot determine.
    - These variables are treated cautiously because pilot reliability was lower.

Libraries:
    pandas:
        dataframe handling.
    re:
        signal detection and sentence splitting.


------------------------------------------------------------
step8.py
------------------------------------------------------------

Purpose:
    Generates dominant discourse support.

Input:
    data/postStep7.csv

Output:
    data/postStep8.csv

Methodology:
    - Computes discourse scores from coded frames and evaluative support variables.
    - Discourse categories:
        1 = sovereignty and emancipation,
        2 = security and stabilization,
        3 = violence and abuse,
        4 = geopolitical competition,
        5 = technocratic / factual reporting,
        6 = mixed / no clear dominance.
    - Uses tie-breaking logic based on explicit frame combinations.
    - Flags mixed or suspicious factual-reporting cases for review.
    - This variable is treated cautiously because pilot reliability was lower.

Libraries:
    pandas:
        dataframe operations.


------------------------------------------------------------
step9.py
------------------------------------------------------------

Purpose:
    Consolidates all pipeline coding outputs.

Inputs:
    data/postStep1.csv
    data/postStep2.csv
    data/postStep3.csv
    data/postStep4.csv
    data/postStep5.csv
    data/postStep6.csv
    data/postStep7.csv
    data/postStep8.csv

Output:
    data/postConsolidated.csv

Methodology:
    - Merges all step outputs by Article_ID.
    - Preserves article text, metadata, code values, notes, and review flags.
    - Builds Review_Flag_Count, Any_Review_Flag, and Review_Sources.
    - Builds Full_Text_For_LLM for later adjudication.

Libraries:
    pandas:
        CSV loading, merging, exporting.
    re:
        article ID normalization.


------------------------------------------------------------
step10.py
------------------------------------------------------------

Purpose:
    Builds JSONL payloads for LLM adjudication.

Input:
    data/postConsolidated.csv

Outputs:
    data/gemini_batch_payloads.jsonl
    data/gemini_batch_payloads.csv

Methodology:
    - Skips articles with Relevance = 1.
    - Sends relevance 2, 3, and 4 articles to LLM review.
    - Chooses prompt layer:
        light = clearer cases,
        full = review-flagged or ambiguous cases.
    - Builds target-context excerpt around WG/AC sentences.
    - Adds pipeline summary as structured support.
    - Estimates token length roughly by character count.
    - Adds pre-LLM pro-review support flag for auditing.

Libraries:
    pandas:
        input/output and dataframe operations.
    json:
        writing JSONL payloads.
    os:
        directory creation.
    re:
        sentence splitting and target-context extraction.


------------------------------------------------------------
GEMINI/step12.py
------------------------------------------------------------

Purpose:
    Submits LLM payloads to the Gemini API.

Inputs:
    data/gemini_batch_payloads.jsonl
    prompts/gemini_system_instruction_v1.txt
    APIkey.txt

Outputs:
    GEMINI/data/gemini_raw_outputs.jsonl
    GEMINI/data/gemini_submission_log.csv
    GEMINI/data/gemini_api_errors.jsonl

Methodology:
    - Sends each user payload to Gemini 2.5 Flash.
    - Uses a system instruction prompt.
    - Sets temperature to 0.
    - Implements request-level retry logic.
    - Supports continuation of previous runs.
    - Supports test mode with a random small sample.
    - Copies parent pipeline data into local GEMINI/data for self-contained execution.
    - Saves full raw API JSON for auditability.

Libraries:
    os, pathlib:
        path handling and directory management.
    json:
        JSONL reading and writing.
    time:
        sleep and backoff.
    shutil:
        copying data into the GEMINI workspace.
    requests:
        HTTP API calls to Gemini.
    pandas:
        writing submission logs.
    random:
        test-mode sampling.


------------------------------------------------------------
GEMINI/step13.py
------------------------------------------------------------

Purpose:
    Parses, validates, and adjudicates Gemini outputs.

Inputs:
    GEMINI/data/gemini_raw_outputs.jsonl
    GEMINI/data/gemini_batch_payloads.jsonl
    GEMINI/data/postConsolidated.csv

Outputs:
    GEMINI/data/gemini_parsed_outputs.csv
    GEMINI/data/gemini_parse_errors.csv
    GEMINI/data/final_llm_coding_working.csv
    GEMINI/data/final_llm_merge_issues.csv
    GEMINI/data/final_conservative_adjudicated_table.csv
    GEMINI/data/final_llm_authoritative_table.csv
    GEMINI/data/final_manual_verification_table.csv
    GEMINI/data/final_high_confidence_coding_table.csv

Methodology:
    - Selects the best raw record per Article_ID.
    - Parses JSON from model responses.
    - Removes code fences if needed.
    - Validates required fields.
    - Validates allowed code values.
    - Merges valid LLM outputs with the consolidated pipeline table.
    - Produces:
        authoritative LLM table,
        conservative adjudicated table,
        manual verification table,
        high-confidence subset.
    - Applies conservative adjudication rules:
        selected LLM upgrades are suppressed in short or bulletin-like cases,
        pipeline values are preserved in certain high-risk disagreements,
        broader-frame activations in weak context are controlled.
    - Builds manual-check reasons.

Libraries:
    os, pathlib:
        output directories and file paths.
    json:
        JSONL loading and JSON parsing.
    re:
        article ID normalization and JSON-block extraction.
    pandas:
        dataframe merging, validation tables, CSV output.


------------------------------------------------------------
local/step12.py
------------------------------------------------------------

Purpose:
    Submits LLM payloads to a local Ollama model.

Inputs:
    data/gemini_batch_payloads.jsonl
    prompts/gemini_system_instruction_v1.txt

Outputs:
    local/data/local_raw_outputs.jsonl
    local/data/local_submission_log.csv
    local/data/local_api_errors.jsonl

Methodology:
    - Sends prompts to Ollama chat API.
    - Uses qwen2.5:14b-instruct.
    - Adds strict JSON-only output enforcement.
    - Explicitly instructs the model to copy Article_ID.
    - Enforces binary variable constraints in the prompt.
    - Supports continuation and batch reruns.
    - Copies parent data into LOCAL/data.

Libraries:
    os, pathlib:
        path handling.
    json:
        JSONL input/output.
    time:
        request spacing and retry backoff.
    shutil:
        data copying.
    requests:
        local Ollama HTTP requests.
    pandas:
        submission logs.
    re:
        extracting Article_ID from payload.


------------------------------------------------------------
local/step13.py
------------------------------------------------------------

Purpose:
    Parses and adjudicates local Ollama outputs.

Inputs:
    local/data/local_raw_outputs.jsonl
    local/data/gemini_batch_payloads.jsonl
    local/data/postConsolidated.csv

Outputs:
    local/data/local_parsed_outputs.csv
    local/data/local_parse_errors.csv
    local/data/final_local_coding_working.csv
    local/data/final_local_merge_issues.csv
    local/data/final_local_conservative_adjudicated_table.csv
    local/data/final_local_llm_authoritative_table.csv
    local/data/final_local_manual_verification_table.csv
    local/data/final_local_high_confidence_coding_table.csv

Methodology:
    Same general parsing, validation, merging, and conservative adjudication logic as
    GEMINI/step13.py, but applied to local Ollama model outputs.

Libraries:
    os, pathlib:
        paths and optional Excel export.
    json:
        JSONL and JSON parsing.
    re:
        JSON block extraction and ID normalization.
    pandas:
        dataframe processing and CSV/XLSX output.


------------------------------------------------------------
step11.py
------------------------------------------------------------

Purpose:
    Builds an expanded diagnostic table from the rule-based pipeline outputs.

Inputs:
    data/postStep1.csv ... data/postStep8.csv

Output:
    data/postDiagnostic.csv

Methodology:
    - Merges all intermediate step outputs.
    - Preserves detailed notes, support counts, and review flags.
    - Builds review metadata.
    - Used mainly for debugging, audit, and inspection.

Libraries:
    pandas:
        CSV loading, merging, exporting.
    re:
        ID normalization.


------------------------------------------------------------
stepB.py
------------------------------------------------------------

Purpose:
    Detects source attribution, republication patterns, and near-duplicate articles.

Inputs:
    data/postStep1.csv
    data/postStep2.csv

Outputs:
    data/postStepB.csv
    data/postStepB_duplicate_pairs.csv
    data/postStepB_duplicate_clusters.csv
    data/postStepB_cluster_members.csv

Methodology:
    - User selects relevance filter:
        all,
        2+,
        3+,
        4.
    - Detects explicit external sources:
        AFP,
        Reuters,
        AP,
        Anadolu,
        RFI,
        France24,
        BBC,
        Le Monde,
        Sputnik,
        TASS, etc.
    - Detects external source names in author field.
    - Detects references to other Malian media.
    - Detects republication and attribution phrases.
    - Computes near-duplicate similarity using RapidFuzz.
    - Builds duplicate pairs, duplicate clusters, and article-level duplicate indicators.
    - Produces a republication index and likely_republished_flag.

Libraries:
    os:
        file existence checks.
    re:
        source, attribution, and republication pattern detection.
    itertools:
        pairwise article comparisons.
    pandas:
        dataframe processing and CSV output.
    rapidfuzz.fuzz:
        fuzzy string similarity for near-duplicate detection.


------------------------------------------------------------
stepBA.py
------------------------------------------------------------

Purpose:
    Supplementary Russian / Russia-attributed source environment analysis.

Inputs:
    data/postStepB.csv
    data/postConsolidated.csv
    GEMINI/data/final_conservative_adjudicated_table.csv

Outputs:
    data/stepBA/postStepBA_russian_sources.csv
    data/stepBA/postStepBA_russian_sources_review.csv
    data/stepBA/discussion/
    data/stepBA/figures/

Methodology:
    - Detects Russian state media, Russian news agencies, Russian official sources,
      Russian embassy sources, Russian defence/MFA sources, and Russia-attributed claims.
    - Separates Russian-source presence from Russian-source dominance.
    - Computes Russian source score, attribution score, and non-Russian source score.
    - Merges final coding layer.
    - Compares frame prevalence between Russian-source and non-Russian-source subsets.
    - Produces heatmaps and discussion notes.
    - This is a supplementary source-environment layer, not a measure of editorial endorsement.

Libraries:
    os, pathlib:
        output paths and directory creation.
    re:
        source-pattern detection.
    datetime:
        timestamped summaries.
    pandas:
        data processing and merging.
    matplotlib:
        heatmap export.
    seaborn:
        optional heatmap rendering.


------------------------------------------------------------
stepC.py
------------------------------------------------------------

Purpose:
    Selects candidate articles for qualitative critical discourse analysis.

Inputs:
    data/postConsolidated.csv
    data/postStepB.csv
    GEMINI/data/final_conservative_adjudicated_table.csv
    or local equivalent.

Outputs:
    backend data folder:
        postStepC_candidates.csv
        postStepC_review.csv
        postStepC_summary.txt

Methodology:
    - User selects backend:
        GEMINI,
        LOCAL.
    - Merges consolidated pipeline data, adjudicated final coding, and StepB enrichment.
    - Filters qualitatively eligible articles.
    - Excludes or penalizes:
        bulletin-style articles,
        very short texts,
        thin target references,
        near-duplicates,
        likely republished texts.
    - Scores candidate articles by:
        dominant-case value,
        strong evaluative/framing value,
        anomaly value,
        transitional value,
        outlet-contrastive value,
        discursive density.
    - Balances selected candidate pool across categories and outlets.
    - Produces a 24-article candidate pool for researcher review.

Libraries:
    os:
        paths and output directories.
    re:
        ID normalization and sentence splitting.
    pandas:
        merging, scoring, selection, export.
    numpy:
        mode and missing-value support.


------------------------------------------------------------
stepD.py
------------------------------------------------------------

Purpose:
    Supplementary diagnostic for Prigozhin/Wagner mutiny references.

Inputs:
    data/postConsolidated.csv
    GEMINI/data/final_conservative_adjudicated_table.csv

Outputs:
    data/postStepD_prigozhin_mutiny_article_flags.csv
    data/postStepD_prigozhin_mutiny_monthly_long.csv
    data/postStepD_prigozhin_mutiny_monthly_wide.csv
    data/postStepD_prigozhin_mutiny_summary.txt
    figures/stepD_prigozhin_mutiny_timeline_*.png

Methodology:
    - Merges consolidated corpus with conservative adjudicated coding.
    - Classifies actor subsets:
        Wagner_only,
        AC_only,
        Both_mentioned.
    - Detects Prigozhin-any and strict mutiny-event references.
    - Strict flag requires Prigozhin/Wagner/Russia context plus mutiny, rebellion,
      Moscow, Rostov, or June 2023 event language.
    - Aggregates monthly counts and percentages by actor subset and relevance filter.
    - Produces timeline figures with denominator panels.

Libraries:
    os, pathlib:
        paths and output directories.
    re:
        lexical event detection and sentence splitting.
    unicodedata:
        accent-insensitive text normalization.
    datetime:
        summary timestamp.
    pandas:
        merging, grouping, monthly tables.
    matplotlib:
        timeline figures.


============================================================
7. CORPUS FOLDER SCRIPTS
============================================================

Location:
    Full_Corpus/CORPUS/


------------------------------------------------------------
corpus1.py
------------------------------------------------------------

Purpose:
    Builds the main CORPUS workspace from pilot outputs.

Inputs:
    Full_Corpus/pilot/data/
    Full_Corpus/pilot/GEMINI/data/
    Full_Corpus/pilot/local/data/ optional

Outputs:
    CORPUS/data/review_master.csv
    CORPUS/data/corpus_overview.csv
    CORPUS/data/relevance_profile.csv
    CORPUS/data/outlet_profile.csv
    CORPUS/data/representation_profile.csv
    CORPUS/data/frame_profile.csv
    CORPUS/data/review_profile.csv
    CORPUS/data/source_republication_profile.csv
    CORPUS/data/adjudication_profile.csv
    CORPUS/tables/
    CORPUS/figures/
    CORPUS/summary/
    CORPUS/schemata/

Methodology:
    - Copies important pipeline, Gemini, and local outputs into CORPUS/data.
    - Builds review_master.csv as the central analysis table.
    - Adds presence flags for Gemini/local outputs.
    - Produces descriptive corpus overview tables.
    - Produces basic bar charts.
    - Creates Mermaid workflow diagrams.

Libraries:
    os:
        directory operations.
    shutil:
        copying outputs.
    pathlib:
        robust path handling.
    datetime:
        summary timestamps.
    pandas:
        CSV loading, merging, profile tables.
    matplotlib:
        bar charts.


------------------------------------------------------------
corpus2.py
------------------------------------------------------------

Purpose:
    Creates analytical subsets and evidence packs.

Input:
    CORPUS/data/review_master.csv

Outputs:
    CORPUS/subsets/
    CORPUS/evidence/
    CORPUS/summary/corpus2_subset_summary.csv

Methodology:
    - Builds subsets by relevance, actor, label, location, associated actor, frame,
      discourse, review status, source/republication, and special analytical criteria.
    - Builds target-context excerpts around WG/AC mentions.
    - Produces evidence packs for later qualitative checking.

Libraries:
    re:
        target-context extraction and sentence splitting.
    pathlib:
        output paths.
    datetime:
        summaries.
    pandas:
        subset filtering and CSV export.


------------------------------------------------------------
corpus3.py
------------------------------------------------------------

Purpose:
    Generates KWIC concordance outputs.

Input:
    CORPUS/data/review_master.csv

Outputs:
    CORPUS/kwic/
    CORPUS/concordance/
    CORPUS/summary/corpus3_summary.txt

Methodology:
    - Defines keyword groups:
        core actor terms,
        labels,
        security terms,
        sovereignty terms,
        abuse terms,
        anti/neocolonial terms,
        Western failure terms,
        security effectiveness terms,
        economic terms,
        Mali state terms,
        external actor terms,
        geopolitical terms,
        source attribution terms.
    - Extracts keyword-in-context rows.
    - Exports per-keyword and per-group concordance files.
    - Builds keyword summary, group summary, outlet profile, and subset profile.

Libraries:
    re:
        keyword matching and sentence splitting.
    pathlib:
        paths.
    datetime:
        summary timestamp.
    pandas:
        KWIC table construction and export.


------------------------------------------------------------
corpus4.py
------------------------------------------------------------

Purpose:
    Summarizes lexical and KWIC outputs.

Inputs:
    review_master.csv
    kwic_all_keywords.csv
    kwic_keyword_summary.csv
    kwic_group_summary.csv
    keyword_outlet_profile.csv
    keyword_subset_profile.csv

Outputs:
    CORPUS/lexical/
    CORPUS/figures/
    CORPUS/summary/corpus4_summary.txt
    CORPUS/summary/corpus4_lexical_findings.txt

Methodology:
    - Identifies top keywords overall.
    - Identifies top keyword groups.
    - Builds top keywords by subset.
    - Builds outlet keyword profiles.
    - Identifies repeated sentence patterns.
    - Exports lexical coverage summaries and figures.

Libraries:
    re:
        repeated sentence normalization.
    pathlib:
        paths.
    datetime:
        summaries.
    pandas:
        grouping and aggregation.
    matplotlib:
        bar charts.


------------------------------------------------------------
corpus5.py
------------------------------------------------------------

Purpose:
    Performs lexical normalization and word-cloud generation.

Input:
    CORPUS/data/review_master.csv

Outputs:
    CORPUS/lexical_norm/
    CORPUS/figures/wordclouds/
    CORPUS/summary/corpus5_summary.txt

Methodology:
    - Builds full article text.
    - Tokenizes text.
    - Applies stopword removal.
    - Supports four token modes:
        surface,
        lemma,
        custom normalization,
        analytical grouping.
    - Creates top-token tables for selected subsets.
    - Creates word clouds.
    - Optional spaCy lemmatization is used when a French model is available.

Libraries:
    re:
        tokenization.
    pathlib:
        output paths.
    datetime:
        summaries.
    collections.Counter:
        token counting.
    pandas:
        subset tables and token profiles.
    matplotlib:
        word-cloud figure rendering.
    wordcloud:
        word cloud generation.
    spacy:
        optional French lemmatization.


------------------------------------------------------------
corpus6.py
------------------------------------------------------------

Purpose:
    Exploratory BERTopic topic modelling.

Input:
    CORPUS/data/review_master.csv

Outputs:
    CORPUS/topics/relevance4/
    CORPUS/topics/africa_corps_or_both/
    CORPUS/topics/corpus6_summary.csv
    CORPUS/figures/
    CORPUS/summary/corpus6_summary.txt

Methodology:
    - Runs BERTopic on:
        relevance 4 articles,
        Africa Corps or both-mentioned articles.
    - Cleans topic text with French/custom stopwords.
    - Uses sentence-transformer embeddings.
    - Uses UMAP for dimensionality reduction.
    - Uses HDBSCAN for clustering.
    - Exports topic info, topic terms, representative documents, outlet/year distributions.
    - Builds broader topic-family groupings from micro-topics.
    - This script is exploratory and supplementary, not a primary coding layer.

Libraries:
    re:
        topic-text cleaning.
    pathlib:
        output paths.
    datetime:
        summaries.
    collections.Counter:
        topic family labels.
    pandas:
        topic tables.
    matplotlib:
        bar charts.
    bertopic:
        topic modelling.
    sentence_transformers:
        embeddings.
    umap:
        dimensionality reduction.
    hdbscan:
        clustering.
    sklearn.feature_extraction.text:
        CountVectorizer and TfidfVectorizer.
    sklearn.cluster:
        AgglomerativeClustering.
    numpy:
        clustering labels and numerical operations.


------------------------------------------------------------
corpus7.py
------------------------------------------------------------

Purpose:
    Exploratory named-entity recognition.

Input:
    CORPUS/data/review_master.csv

Outputs:
    CORPUS/ner/
    CORPUS/summary/corpus7_summary.txt
    CORPUS/summary/corpus7_ner_notes.txt

Methodology:
    - Uses French spaCy NER.
    - Extracts article-level and sentence-level entities.
    - Keeps entity types:
        PER / PERSON,
        ORG,
        GPE,
        LOC,
        NORP.
    - Applies custom normalization for key entities:
        Russia,
        France,
        FAMa,
        Assimi Goita,
        MINUSMA,
        ECOWAS,
        jihadist groups, etc.
    - Builds entity profiles overall, by subset, by outlet, and by label.
    - Builds sentence-level co-occurrence with Wagner/Africa Corps.

Libraries:
    re:
        sentence splitting and target detection.
    pathlib:
        output paths.
    datetime:
        summaries.
    collections.Counter:
        entity counting.
    pandas:
        entity tables.
    spacy:
        French named-entity recognition.


------------------------------------------------------------
corpus8.py
------------------------------------------------------------

Purpose:
    Integrated synthesis layer for writing support.

Inputs:
    review_master.csv
    corpus overview/profile files
    evidence packs
    KWIC outputs
    lexical outputs
    NER outputs
    BERTopic outputs

Outputs:
    CORPUS/synthesis/
    CORPUS/summary/chapter52_working_notes.txt
    CORPUS/summary/discussion_working_notes.txt
    CORPUS/summary/corpus8_summary.txt

Methodology:
    - Integrates structured coding, lexical, NER, source/republication, and topic outputs.
    - Produces writing-oriented synthesis tables:
        corpus_wide_synthesis.csv,
        actor_representation_synthesis.csv,
        frame_discourse_synthesis.csv,
        outlet_difference_synthesis.csv,
        source_republication_synthesis.csv,
        wagner_vs_ac_integrated_comparison.csv,
        source_attributed_vs_nonattributed_integrated_comparison.csv,
        review_vs_highconfidence_integrated_comparison.csv,
        key_findings_ch52.csv,
        key_findings_discussion.csv.
    - Does not create new coding.
    - Reorganizes existing outputs into interpretable thesis-support tables.

Libraries:
    os:
        file checks.
    pathlib:
        paths.
    datetime:
        summaries.
    pandas:
        loading and synthesis tables.


------------------------------------------------------------
synthesis.py
------------------------------------------------------------

Purpose:
    Exports the most useful discussion-support files into one folder and ZIP archive.

Outputs:
    CORPUS/discussion_export/
    CORPUS/discussion_export_bundle.zip

Methodology:
    - Copies selected synthesis, NER, lexical, topic, source/republication, and profile files.
    - Writes a manifest.
    - Writes a README for the export bundle.
    - Creates a ZIP archive.

Libraries:
    pathlib:
        paths.
    shutil:
        file copying.
    csv:
        manifest writing.
    zipfile:
        ZIP archive creation.
    datetime:
        timestamp.


============================================================
8. FINAL ANALYSIS SCRIPT
============================================================

Location:
    Full_Corpus/ANA/ANALYSIS.py

Purpose:
    Produces final chapter-oriented analysis outputs.

Inputs:
    CORPUS/data/review_master.csv
    pilot/data/postConsolidated.csv
    pilot/data/postStepB.csv
    adjudicated conservative table if available

Outputs:
    ANA/output/
        timeline/
        chapter_5_2/
        chapter_5_3/
        diagnostics/
        cda_sample/

Methodology:
    - Harmonizes final variables from adjudicated or fallback columns.
    - Derives period groups:
        P1_Early_Wagner,
        P2_Mutiny_Transition,
        P3_AfricaCorps_Phase.
    - Builds actor subsets:
        Wagner_only,
        AC_or_Both,
        Other.
    - Exports:
        timeline figures,
        corpus overview tables,
        quantitative content-analysis tables,
        frame and discourse profiles,
        outlet profiles,
        source/republication profiles,
        chi-square diagnostics,
        logistic regression summaries where possible,
        CDA sample text bundles,
        diagnostic tables for missing/invalid values.
    - This is the main chapter-output script.

Libraries:
    pathlib:
        path handling.
    datetime:
        timestamps.
    argparse:
        command-line flags.
    math:
        Cramér's V calculation.
    re:
        ID and filename normalization.
    warnings:
        suppressing non-critical warnings.
    numpy:
        numerical operations.
    pandas:
        data analysis and export.
    matplotlib:
        plots.
    seaborn:
        optional heatmaps.
    scipy.stats:
        chi-square tests.
    statsmodels.formula.api:
        logistic regression models.


============================================================
9. FINAL_PILOT RELIABILITY SCRIPTS
============================================================

Location:
    Final_Pilot/RELIABILITY/


------------------------------------------------------------
pilot1.py
------------------------------------------------------------

Purpose:
    Builds master comparison tables for pilot reliability analysis.

Methodology:
    - Reads human coder and benchmark model outputs.
    - Reads pipeline, Gemini, and local branch outputs.
    - Normalizes all variables into the same final code structure.
    - Applies explicit N/A coding for irrelevant articles.
    - Produces long and wide comparison tables.

Libraries:
    os, re, json:
        file handling and parsing coder TXT/JSON outputs.
    pandas:
        dataframe construction and export.


------------------------------------------------------------
pilot2.py
------------------------------------------------------------

Purpose:
    Computes pilot reliability and agreement statistics.

Methodology:
    - Computes Krippendorff's alpha for human coders.
    - Builds human-majority reference.
    - Computes pairwise agreement and Cohen's kappa:
        benchmark models vs pipeline,
        benchmark models vs human majority,
        operational branches vs pipeline,
        operational branches vs human majority,
        high-confidence outputs vs references.
    - Exports variable-level reliability summaries, N/A diagnostics, and coverage diagnostics.

Libraries:
    os:
        paths.
    numpy:
        numerical arrays and NaN handling.
    pandas:
        tables.
    collections.Counter:
        majority vote with tie handling.
    sklearn.metrics.cohen_kappa_score:
        Cohen's kappa.
    krippendorff:
        Krippendorff's alpha.


------------------------------------------------------------
pilot3.py
------------------------------------------------------------

Purpose:
    Selects the most suitable output layer from pilot comparison.

Methodology:
    - Builds human-majority reference.
    - Compares output layers against:
        human majority,
        benchmark LLM coders,
        pre-LLM pipeline.
    - Computes agreement, distance, critical-variable agreement, weighted score, and balance score.
    - Produces dumbbell plots, heatmaps, and full distance matrix.
    - Supports selection of Gemini conservative adjudicated output.

Libraries:
    os:
        paths.
    numpy:
        numerical operations.
    pandas:
        data tables.
    matplotlib:
        dumbbell plots.
    seaborn:
        heatmaps.
    tqdm:
        progress bars.


------------------------------------------------------------
heatmaps.py
------------------------------------------------------------

Purpose:
    Produces additional heatmaps and output-layer comparison summaries.

Methodology:
    - Builds human-majority reference.
    - Computes agreement, distance, and kappa between output layers and benchmark references.
    - Produces:
        output layer vs benchmark summary,
        dumbbell plot,
        benchmark heatmap,
        full distance matrix.

Libraries:
    os:
        paths.
    pandas:
        tables.
    numpy:
        numerical calculations.
    matplotlib:
        plots.
    seaborn:
        heatmaps.
    sklearn.metrics.cohen_kappa_score:
        kappa scores.


------------------------------------------------------------
DBplot.py
------------------------------------------------------------

Purpose:
    Produces a simplified dumbbell plot of output-layer alignment.

Methodology:
    - Compares each target output layer against:
        human majority,
        local pipeline draft.
    - Computes mean agreement and distance.
    - Exports summary table and plot.

Libraries:
    os:
        paths.
    pandas:
        data tables.
    numpy:
        numerical calculations.
    matplotlib:
        dumbbell plot.


============================================================
10. FINAL_PILOT PILOT SCRIPTS
============================================================

The scripts in Final_Pilot/pilot/ mirror the full-corpus workflow but use the smaller pilot
dataset and often export XLSX as well as CSV.

Key differences:
    - The pilot uses data/pilot.xlsx as initial input.
    - Several outputs are XLSX-based for easier manual inspection.
    - The pilot includes both Gemini and local branch outputs.
    - Reliability scripts then compare human coders, benchmark models, pipeline, and adjudication layers.

Methodologically:
    - Final_Pilot/pilot/step1.py to step9.py correspond to Full_Corpus/pilot/step1.py to step9.py.
    - Final_Pilot/pilot/step10.py builds pilot LLM payloads.
    - Final_Pilot/pilot/GEMINI/step12.py and step13.py run and parse Gemini pilot outputs.
    - Final_Pilot/pilot/local/step12.py and step13.py run and parse local Ollama pilot outputs.
    - Final_Pilot/pilot/stepA.py exports a clean draft coding table.
    - Final_Pilot/pilot/stepB.py performs source/republication and duplicate diagnostics.

Main libraries:
    pandas
    re
    json
    os
    time
    requests
    rapidfuzz
    openpyxl


============================================================
11. LLM PROMPTING AND VALIDATION
============================================================

LLM adjudication is constrained by:

1. fixed system prompt,
2. structured user payload,
3. predefined codebook variables,
4. JSON-only response requirement,
5. strict required-field validation,
6. strict allowed-value validation,
7. conservative post-processing and fallback rules.

Required LLM output fields include:

    Article_ID
    V04_Relevance_Final
    V05_Actor_Mention_Final
    V06_Successor_Frame_Final
    V07_Dominant_Label_Final
    V08_Stance_Final
    V09_Dominant_Location_Final
    V10_Ambivalence_Final
    V11_Legitimation_Final
    V12_Counterterrorism_Final
    V13_Sovereignty_Final
    V14_Human_Rights_Abuse_Final
    V15_Anti_or_Neocolonialism_Final
    V16_Western_Failure_Final
    V17_Security_Effectiveness_Final
    V18_Economic_Interests_Final
    V19_Geopolitical_Rivalry_Final
    V20_Main_Associated_Actor_Final
    V21_Dominant_Discourse_Final
    LLM_Review_Note
    Pro_Review_Candidate
    Pro_Review_Reason

Invalid model outputs are not silently accepted. They are written to parse or validation error files.


============================================================
12. MAIN OUTPUTS USED IN THE THESIS
============================================================

Core full-corpus pipeline:
    Full_Corpus/pilot/data/postConsolidated.csv

Main adjudicated coding table:
    Full_Corpus/pilot/GEMINI/data/final_conservative_adjudicated_table.csv

Source and republication layer:
    Full_Corpus/pilot/data/postStepB.csv

Russian-source supplementary layer:
    Full_Corpus/pilot/data/stepBA/

Qualitative candidate selection:
    Full_Corpus/pilot/GEMINI/data/postStepC_candidates.csv
    Full_Corpus/pilot/GEMINI/data/postStepC_review.csv

Prigozhin/Wagner mutiny diagnostic:
    Full_Corpus/pilot/data/postStepD_prigozhin_mutiny_*.csv

Central corpus master:
    Full_Corpus/CORPUS/data/review_master.csv

Chapter-oriented analysis:
    Full_Corpus/ANA/output/

Pilot reliability outputs:
    Final_Pilot/REL/data/


============================================================
13. REPRODUCIBILITY CAUTIONS
============================================================

1. The corpus is availability-based.
   It should not be interpreted as a statistically representative sample of all Malian media.

2. The retrieval strategy is explicit-term based.
   Articles that refer indirectly to Russian instructors or contractors without the target terms
   may be underrepresented.

3. Online news is only one part of Mali's information environment.
   Radio, television, social media, WhatsApp, TikTok, oral communication, and local-language
   media are not directly analysed.

4. Some variables are more reliable than others.
   Pilot reliability was stronger for manifest variables such as relevance, actor mention,
   successor framing, location, and several frames. Stance, legitimation, geopolitical rivalry,
   and dominant discourse were less stable and are interpreted cautiously.

5. LLM outputs are not treated as final truth.
   They are constrained, parsed, validated, and conservatively adjudicated.

6. Source/republication detection is diagnostic.
   It does not prove editorial endorsement or political alignment.

7. KWIC, lexical, NER, BERTopic, and word-cloud outputs are exploratory.
   They support interpretation but do not replace close reading or codebook-based analysis.

8. API-based outputs may change if model versions change.
   Gemini API behaviour may not be perfectly reproducible over time.

9. Local Ollama outputs depend on local hardware and model version.


============================================================
14. MINIMAL REPRODUCTION SEQUENCE
============================================================

To reproduce the main full-corpus pipeline:

    cd Full_Corpus/pilot

    python step1.py
    python step2.py
    python step3.py
    python step4.py
    python step5.py
    python step6.py
    python step7.py
    python step8.py
    python step9.py
    python step10.py

    cd GEMINI
    python step12.py
    python step13.py

    cd ..
    python stepB.py
    python stepBA.py
    python stepC.py
    python stepD.py

    cd ../CORPUS
    python corpus1.py
    python corpus2.py
    python corpus3.py
    python corpus4.py
    python corpus5.py
    python corpus7.py
    python corpus8.py

    cd ../ANA
    python ANALYSIS.py --all

Optional exploratory topic modelling:

    cd Full_Corpus/CORPUS
    python corpus6.py


============================================================
15. HOW TO INTERPRET THE OUTPUTS
============================================================

Primary outputs:
    postConsolidated.csv
    final_conservative_adjudicated_table.csv
    review_master.csv
    ANA/output/

Diagnostic outputs:
    parse errors,
    merge issues,
    manual verification tables,
    review flags,
    duplicate clusters,
    source/republication profiles.

Supplementary outputs:
    KWIC,
    lexical normalization,
    word clouds,
    NER,
    BERTopic,
    synthesis tables.

Only the primary and selected diagnostic outputs are used as direct evidence in the thesis.
Supplementary outputs are included for transparency, triangulation, and future research.


============================================================
16. CITATION
============================================================

If using or referring to this code package, cite:

    Vrána, P. (2026). diploma-thesis [Source code].
    GitHub. https://github.com/v0987654321/diploma-thesis

Associated thesis:

    Vrána, P. Media Discourses of the Wagner Group in Malian Online Press.


============================================================
17. FINAL NOTE
============================================================

This repository is best understood as an open empirical and methodological workspace rather
than only as a collection of final results. It contains the scripts and outputs necessary to
inspect how the corpus was processed, how the variables were coded, how the pilot validation
was performed, how the LLM adjudication layer was constrained and checked, and how the final
analytical tables and figures were produced.

The central methodological decision of the workflow is the use of the Gemini conservative
adjudicated layer as the principal analytical output for the full corpus, supported by rule-based
pipeline diagnostics, source/republication enrichment, and qualitative close reading.

This file was created in cooperation with Chat GPT 5.5. 