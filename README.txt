README.txt
==========

Project:
Media Discourses of the Wagner Group in Malian Online Press

Author:
Bc. Petr Vrána

Supervisor:
Mgr. Martin Schmiedl, Ph.D.

Institution:
Mendel University in Brno
Faculty of Regional Development and International Studies

Repository:
https://github.com/v0987654321/diploma-thesis

Associated thesis:
Vrána, P. (2026). Media Discourses of the Wagner Group in Malian Online Press [Master’s thesis, Mendel University in Brno]. https://theses.cz/id/ieu87x/

Repository status:
This repository contains the computational workflow, codebook-related prompts, intermediate outputs, diagnostic files, and supplementary analyses prepared for the diploma thesis Media Discourses of the Wagner Group in Malian Online Press. It also contains an article-specific corrected workflow created after a downstream provenance audit.

The repository is intended as a transparency, replication, and reuse package. It documents how the article corpus was cleaned, coded, validated, enriched with source/republication information, analysed, and prepared for quantitative content analysis and qualitative discourse interpretation.

The repository includes both:
1. the original thesis workflow and its frozen output layer; and
2. a corrected article workflow using the Gemini conservative adjudicated layer as its primary coding layer.

The original thesis folders are retained unchanged for historical transparency and reproducibility of the submitted thesis. The article-specific folders are separate and do not overwrite original outputs.

======================================================================
1. RESEARCH OVERVIEW
======================================================================

The project examines how Wagner Group and Africa Corps were represented in selected Malian online news outlets between July 2021 and December 2025.

The broader thesis used an explanatory sequential mixed-methods design:
1. corpus construction and preprocessing;
2. rule-based and semi-automated quantitative coding;
3. pilot validation and reliability assessment;
4. model-assisted coding and adjudication;
5. source/republication enrichment;
6. corpus-level quantitative analysis;
7. purposive qualitative sample selection;
8. qualitative discourse interpretation;
9. supplementary exploratory lexical, NER, and topic-modelling analysis.

The article derived from the thesis focuses more narrowly on whether selected Malian online news can be characterized as straightforwardly or uniformly pro-Russian. Its central argument is that Russian-linked military actors may be normalized indirectly through security framing, sovereignty claims, anti-French positioning, source hierarchy, and institutional reclassification, while remaining associated with mercenarism, civilian harm, human-rights allegations, and international criticism.

======================================================================
2. MAIN METHODOLOGICAL PRINCIPLES
======================================================================

The workflow is grounded in quantitative content analysis and computer-assisted text analysis. It follows the following principles:

- explicit variable definitions;
- predefined codebook categories;
- article-level unit of analysis;
- transparent preprocessing;
- rule-based baseline coding;
- pilot validation;
- variable-level reliability assessment;
- strict parsing and validation of model outputs;
- conservative adjudication rather than direct acceptance of raw LLM output;
- separation between primary coding outputs, diagnostics, and exploratory outputs;
- researcher-led interpretation of quantitative and qualitative findings.

The workflow does not treat automated coding or LLM output as an autonomous source of truth. Scripts operationalize a researcher-defined codebook and generate structured support for analysis. Final interpretation remains the responsibility of the researcher.

======================================================================
3. IMPORTANT DATA-LAYER NOTE
======================================================================

The repository contains two analytically distinct downstream workflows.

A. ORIGINAL THESIS WORKFLOW

The original thesis workflow is preserved in:

Full_Corpus/CORPUS/
Full_Corpus/ANA/

These folders reproduce the data layer used in the submitted diploma thesis. A later provenance audit found that the original downstream workflow used pipeline coding values from review_master.csv because the conservative Gemini adjudicated variables were copied into the CORPUS workspace but were not merged into review_master.csv.

As a result, the original ANA output layer uses the original pipeline-coded variables for its Final_* outputs through fallback logic. The original thesis results, tables, and figures remain internally consistent with this frozen pipeline-based analytical layer.

B. CORRECTED ARTICLE WORKFLOW

The corrected article workflow is located in:

Full_Corpus/CORPUS_article/
Full_Corpus/ANA_article/

This workflow was created to correct the downstream provenance issue. In this version:

- the Gemini conservative adjudicated table is merged directly into review_master.csv;
- original pipeline values are preserved as Pipeline_* variables;
- article-level Final_* variables are created from Gemini conservative adjudication with pipeline fallback where adjudicated values are unavailable;
- copied corpus and ANA scripts use the corrected final coding layer;
- original thesis outputs remain untouched.

The article-specific workflow should be used for future journal-article analysis based on the conservative Gemini adjudicated coding layer.

======================================================================
4. PYTHON ENVIRONMENT AND DEPENDENCIES
======================================================================

Recommended Python version:
    Python 3.10 or newer

Core libraries:
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

Optional or supplementary libraries:
    spacy
    fr_core_news_sm or fr_core_news_md
    bertopic
    sentence-transformers
    umap-learn
    hdbscan

Standard-library modules used:
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
    hashlib
    subprocess
    sys

Suggested base installation:
    pip install pandas numpy requests openpyxl dateparser rapidfuzz matplotlib seaborn scipy statsmodels scikit-learn krippendorff tqdm wordcloud

Optional installation for NER and topic modelling:
    pip install spacy bertopic sentence-transformers umap-learn hdbscan
    python -m spacy download fr_core_news_sm

For local LLM inference:
    Ollama must be installed and running.
    The local workflow expects:
        qwen2.5:14b-instruct

For Gemini API inference:
    A Google Gemini API key is required in:
        APIkey.txt

======================================================================
5. FOLDER STRUCTURE
======================================================================

The repository contains two major workspaces:

1. Final_Pilot/
   Pilot workflow and reliability validation.

2. Full_Corpus/
   Full-corpus processing, source/republication enrichment, corpus workspaces,
   chapter-oriented analysis, and corrected article workflow.

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
        stepE.py, if present
        GEMINI/
            step12.py
            step13.py
        local/
            step12.py
            step13.py
        data/
        prompts/

    CORPUS/
        Original thesis corpus workspace.
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
        Original thesis chapter-oriented analysis workspace.
        ANALYSIS.py
        output/

    CORPUS_article/
        Corrected article-specific corpus workspace.
        Created as a copy of CORPUS/ and then patched.
        Uses Gemini conservative adjudication with pipeline fallback.
        Contains its own:
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
            ARTICLE_WORKFLOW_README.txt
            ARTICLE_LAYER_PROVENANCE.txt

    ANA_article/
        Corrected article-specific chapter-oriented analysis workspace.
        Created as a copy of ANA/ and patched to read:
            CORPUS_article/data/review_master.csv
        Contains:
            ANALYSIS.py
            output/

======================================================================
6. IMPORTANT PATH NOTE
======================================================================

Most original scripts use relative paths and should normally be run from their own workspace.

Examples:

Full corpus pipeline:
    cd Full_Corpus/pilot
    python step1.py

Gemini branch:
    cd Full_Corpus/pilot/GEMINI
    python step12.py
    python step13.py

Original thesis corpus outputs:
    cd Full_Corpus/CORPUS
    python corpus1.py

Original thesis chapter analysis:
    cd Full_Corpus/ANA
    python ANALYSIS.py --all

Corrected article corpus outputs:
    cd Full_Corpus/CORPUS_article
    python corpus1.py

Corrected article chapter analysis:
    cd Full_Corpus/ANA_article
    python ANALYSIS.py --all

If a file is missing, check:
    1. whether the preceding step has been run;
    2. whether the script is being run from the correct directory;
    3. whether required inputs exist in the relevant data/ folder;
    4. whether the workflow used is the original thesis layer or the corrected article layer.

======================================================================
7. FULL CORPUS PIPELINE SCRIPTS
======================================================================

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

Main procedures:
    - normalizes article IDs to six-digit strings;
    - derives outlet codes and outlet names;
    - parses dates and records date precision;
    - cleans HTML residues and generic boilerplate;
    - removes duplicated lead text where the body repeats the lead;
    - applies outlet-specific body cleaning;
    - builds full cleaned article text.

Main variables:
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

------------------------------------------------------------
step2.py
------------------------------------------------------------

Purpose:
    Performs relevance screening and target/Mali-context scoring.

Input:
    data/postStep1.csv

Output:
    data/postStep2.csv

Main procedures:
    - detects Wagner, Africa Corps, Russian mercenary, and Russian instructor terminology;
    - detects Mali-context markers;
    - scores target centrality in headlines, leads, and bodies;
    - computes target sentence count, target sentence share, target clustering, and first-third presence;
    - identifies bulletin-style articles;
    - assigns relevance codes:
        1 = not relevant
        2 = marginal mention only
        3 = substantively relevant
        4 = main topic;
    - flags possible manual-review cases.

------------------------------------------------------------
step3.py
------------------------------------------------------------

Purpose:
    Codes actor mention, successor framing, and dominant labels.

Inputs:
    data/postStep2.csv
    data/postStep1.csv

Output:
    data/postStep3.csv

Main procedures:
    - retains relevance 2, 3, and 4 articles;
    - codes Actor_Mention:
        1 = Wagner explicitly mentioned
        2 = Africa Corps explicitly mentioned
        3 = both explicitly mentioned
        4 = indirect Russian contractor terminology
        5 = cannot determine;
    - codes Successor_Frame;
    - codes Dominant_Label:
        mercenaries
        instructors/advisers
        allies/partners
        foreign/occupying forces
        neutral designation
        multiple/no clear dominance;
    - uses weighted local target-context scoring.

------------------------------------------------------------
step4.py
------------------------------------------------------------

Purpose:
    Codes the dominant location of referred Wagner/Africa Corps activity.

Inputs:
    data/postStep3.csv
    data/postStep1.csv

Output:
    data/postStep4.csv

Location categories:
    1 = Mali
    2 = Other African countries
    3 = Ukraine
    4 = Other location
    5 = Mali and other location

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

Actor categories:
    1 = Malian army / junta
    2 = Russia / Russian state
    3 = France
    4 = UN / MINUSMA
    5 = ECOWAS / regional actors
    6 = local civilians
    7 = jihadist / terrorist groups
    8 = Western states broadly
    9 = no clear dominant actor
    10 = other

The script uses headline/lead/body weighting and sentence-level proximity to Wagner/Africa Corps references.

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

Frame variables:
    Counterterrorism
    Sovereignty
    Human_Rights_Abuse
    Anti_or_Neocolonialism
    Western_Failure
    Security_Effectiveness
    Economic_Interests
    Geopolitical_Rivalry

The script uses headline, lead, and local target context, frame-specific thresholds, and suppressors for weak marginal or bulletin-style cases.

------------------------------------------------------------
step7.py
------------------------------------------------------------

Purpose:
    Generates support indicators for stance, ambivalence, and legitimation.

Inputs:
    data/postStep6.csv
    data/postStep1.csv

Output:
    data/postStep7.csv

Variables:
    Stance_Support:
        1 = negative
        2 = neutral
        3 = positive
        4 = mixed/ambivalent
        5 = cannot determine

    Ambivalence_Support:
        0 = no strong ambivalence
        1 = positive and negative signals both present

    Legitimation_Support:
        1 = delegitimized
        2 = normalized / implicitly legitimized
        3 = explicitly legitimized
        4 = cannot determine

These indicators are interpreted cautiously because pilot reliability was lower than for more manifest variables.

------------------------------------------------------------
step8.py
------------------------------------------------------------

Purpose:
    Generates dominant discourse support.

Input:
    data/postStep7.csv

Output:
    data/postStep8.csv

Discourse categories:
    1 = sovereignty and emancipation
    2 = security and stabilization
    3 = violence and abuse
    4 = geopolitical competition
    5 = technocratic / factual reporting
    6 = mixed / no clear dominance

------------------------------------------------------------
step9.py
------------------------------------------------------------

Purpose:
    Consolidates original pipeline outputs.

Inputs:
    data/postStep1.csv through data/postStep8.csv

Output:
    data/postConsolidated.csv

The script merges original pipeline codes, article text, metadata, support notes, and review flags. It creates Review_Flag_Count, Any_Review_Flag, Review_Sources, and Full_Text_For_LLM.

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

The script excludes relevance 1 cases, prepares light or full prompts depending on ambiguity and review flags, extracts local target context, and provides a structured pipeline summary to the model.

------------------------------------------------------------
step11.py
------------------------------------------------------------

Purpose:
    Builds an expanded diagnostic table from rule-based pipeline outputs.

Inputs:
    data/postStep1.csv through data/postStep8.csv

Output:
    data/postDiagnostic.csv

This table preserves detailed notes, support counts, and review flags for audit and inspection.

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

Main procedures:
    - detects explicit external sources;
    - detects agency, foreign media, and author-field source signals;
    - detects references to other Malian media;
    - detects attribution and republication phrases;
    - computes near-duplicate similarity using RapidFuzz;
    - creates duplicate clusters;
    - creates source-attributed and likely-republished indicators.

The source/republication layer is diagnostic. It does not prove editorial alignment or political endorsement.

------------------------------------------------------------
stepBA.py
------------------------------------------------------------

Purpose:
    Supplementary Russian / Russia-attributed source-environment analysis.

Inputs:
    data/postStepB.csv
    data/postConsolidated.csv
    GEMINI/data/final_conservative_adjudicated_table.csv

Outputs:
    data/stepBA/
    data/stepBA/discussion/
    data/stepBA/figures/

The script identifies Russian state-media, official, military, embassy, and attributed-claim signals, then compares frames between Russian-source and non-Russian-source subsets. This is a supplementary source-environment layer and does not measure editorial endorsement.

------------------------------------------------------------
stepC.py
------------------------------------------------------------

Purpose:
    Selects a candidate pool for qualitative critical discourse analysis.

Inputs:
    data/postConsolidated.csv
    data/postStepB.csv
    GEMINI/data/final_conservative_adjudicated_table.csv
    or local equivalent

Outputs:
    postStepC_candidates.csv
    postStepC_review.csv
    postStepC_summary.txt

The script scores eligible relevance 3 and 4 texts according to dominant, strong, anomalous, transitional, and outlet-contrastive value, while penalizing short, bulletin-style, near-duplicate, and likely republished material.

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

The script detects broad Prigozhin references and stricter June 2023 mutiny references, then aggregates them by actor subset and period.

------------------------------------------------------------
stepE.py
------------------------------------------------------------

Purpose:
    Creates an article-specific qualitative review corpus for close reading and discourse interpretation.

Inputs:
    data/postConsolidated.csv
    GEMINI/data/final_conservative_adjudicated_table.csv
    data/postStepB.csv
    data/stepBA/discussion/stepBA_russian_sources_with_final_coding.csv, if available

Outputs:
    data/stepEarticle/
        stepEarticle_candidates_all.csv
        stepEarticle_ranked_candidates.csv
        stepEarticle_selected_review_corpus.csv
        stepEarticle_selected_review_corpus.txt
        stepEarticle_existing_thesis_cases.csv
        stepEarticle_summary.txt

The script identifies potential qualitative cases involving:
    - critical Wagner representation;
    - mercenary labels;
    - human-rights framing;
    - source-attributed critical material;
    - anti-French displacement;
    - Russia-attributed but non-positive material;
    - Africa Corps successor framing.

Final qualitative selection remains researcher-led.

======================================================================
8. GEMINI AND LOCAL LLM BRANCHES
======================================================================

Location:
    Full_Corpus/pilot/GEMINI/
    Full_Corpus/pilot/LOCAL/

GEMINI/step12.py

Purpose:
    Submits payloads to Gemini 2.5 Flash.

Inputs:
    data/gemini_batch_payloads.jsonl
    prompts/gemini_system_instruction_v1.txt
    APIkey.txt

Outputs:
    GEMINI/data/gemini_raw_outputs.jsonl
    GEMINI/data/gemini_submission_log.csv
    GEMINI/data/gemini_api_errors.jsonl

The script uses standardized system instructions, temperature 0, retry/backoff logic, continuation support, and raw API-output logging.

GEMINI/step13.py

Purpose:
    Parses, validates, merges, and adjudicates Gemini outputs.

Outputs:
    GEMINI/data/gemini_parsed_outputs.csv
    GEMINI/data/gemini_parse_errors.csv
    GEMINI/data/final_llm_coding_working.csv
    GEMINI/data/final_llm_merge_issues.csv
    GEMINI/data/final_conservative_adjudicated_table.csv
    GEMINI/data/final_llm_authoritative_table.csv
    GEMINI/data/final_manual_verification_table.csv
    GEMINI/data/final_high_confidence_coding_table.csv

The conservative table contains Adj_V* values and applies rule-based fallback conditions to selected high-risk cases.

LOCAL/step12.py and LOCAL/step13.py

Purpose:
    Run an equivalent local branch using Ollama and qwen2.5:14b-instruct.

The local branch follows the same broad architecture: structured prompts, strict JSON output enforcement, parsing, validation, and conservative adjudication. It is retained primarily for comparison, audit, and robustness purposes.

======================================================================
9. PILOT RELIABILITY WORKFLOW
======================================================================

Location:
    Final_Pilot/RELIABILITY/

Main scripts:
    pilot1.py
    pilot2.py
    pilot3.py
    heatmaps.py
    DBplot.py

The pilot workflow compares:
    - five human coders;
    - benchmark LLM coders;
    - the local rule-based pipeline;
    - Gemini operational outputs;
    - local Ollama outputs;
    - conservative and high-confidence downstream layers.

Reliability procedures include:
    - Krippendorff’s alpha;
    - percent agreement;
    - Cohen’s kappa;
    - comparison with a human-majority reference;
    - comparison with pipeline coding;
    - distance matrices;
    - output-layer selection diagnostics.

The pilot selected the Gemini conservative output as the preferred adjudicated layer for the corrected article workflow because it offered a balanced position between human-majority alignment and pipeline consistency.

======================================================================
10. ORIGINAL THESIS CORPUS WORKSPACE
======================================================================

Location:
    Full_Corpus/CORPUS/

Main scripts:
    corpus1.py through corpus8.py
    synthesis.py

Important note:
    The original CORPUS workspace is preserved as used for the thesis. Its review_master.csv contains original pipeline operational codes and did not merge the conservative adjudicated values into the final downstream coding columns.

Main scripts:

corpus1.py
    Copies selected pilot and LLM outputs into CORPUS/data.
    Builds review_master.csv and descriptive corpus profiles.

corpus2.py
    Creates analytical subsets and evidence packs.

corpus3.py
    Generates keyword-in-context concordance outputs.

corpus4.py
    Produces lexical and KWIC summaries.

corpus5.py
    Produces normalized lexical profiles and word clouds.

corpus6.py
    Produces exploratory BERTopic outputs.

corpus7.py
    Produces exploratory French-language NER outputs.

corpus8.py
    Produces synthesis tables and writing-support files.

synthesis.py
    Exports selected discussion-support materials into a bundle and ZIP archive.

The lexical, BERTopic, and NER layers are supplementary and exploratory. They support interpretation but do not replace codebook-based analysis or close reading.

======================================================================
11. ORIGINAL THESIS ANA WORKSPACE
======================================================================

Location:
    Full_Corpus/ANA/

Main script:
    ANALYSIS.py

The original ANA workflow creates:
    ANA/output/
        timeline/
        chapter_5_2/
        chapter_5_3/
        diagnostics/
        cda_sample/

The script harmonizes coding variables and creates chapter-oriented tables, figures, diagnostics, source/republication profiles, chi-square tests, optional regression outputs, and CDA text bundles.

Important provenance note:
    Because the original ANA workflow read CORPUS/data/review_master.csv and that file did not contain the conservative Adj_V* columns, the original ANA results used pipeline coding values through fallback logic. These outputs correspond to the frozen submitted thesis layer.

======================================================================
12. CORRECTED ARTICLE WORKFLOW
======================================================================

Location:
    Full_Corpus/CORPUS_article/
    Full_Corpus/ANA_article/

Purpose:
    To create a separate article-specific workflow using Gemini conservative adjudication with pipeline fallback as the primary final coding layer.

Creation:
    The article folders were created as copies of the original CORPUS and ANA workspaces. Their generated outputs were cleared, and selected scripts were patched.

CORPUS_article/corpus1.py correction:
    - merges final_conservative_adjudicated_table.csv directly into review_master;
    - preserves original pipeline coding variables as Pipeline_* fields;
    - creates Final_* variables using Adj_V* values with original pipeline fallback;
    - uses corrected Final_* values for operational downstream article coding;
    - records:
        Article_Coding_Layer =
        Gemini conservative adjudication with pipeline fallback.

ANA_article/ANALYSIS.py correction:
    - reads:
        CORPUS_article/data/review_master.csv
      instead of:
        CORPUS/data/review_master.csv;
    - uses corrected Final_* values for article-level analysis;
    - exports corrected article-specific tables and figures to:
        ANA_article/output/.

The corrected article layer produced the following core corpus distribution:

    Total corpus: 2,474 articles
    Relevance 1: 498 articles
    Relevance 2: 947 articles
    Relevance 3: 278 articles
    Relevance 4: 751 articles
    Relevance 2+: 1,976 articles
    Relevance 3+: 1,029 articles

These figures differ from the original thesis output because the original thesis ANA layer used pipeline coding, whereas the article workflow uses conservative Gemini adjudication with pipeline fallback.

Suggested article workflow sequence:

    cd Full_Corpus/CORPUS_article
    python corpus1.py
    python corpus2.py
    python corpus3.py
    python corpus4.py
    python corpus5.py
    python corpus6.py      [optional/exploratory]
    python corpus7.py      [optional/exploratory]
    python corpus8.py      [optional synthesis]

    cd Full_Corpus/ANA_article
    python ANALYSIS.py --all

For the journal article, all descriptive statistics, tables, and figures should be generated from ANA_article/output/, not from ANA/output/.

======================================================================
13. MAIN OUTPUTS
======================================================================

Original thesis-layer outputs:
    Full_Corpus/pilot/data/postConsolidated.csv
    Full_Corpus/pilot/GEMINI/data/final_conservative_adjudicated_table.csv
    Full_Corpus/CORPUS/data/review_master.csv
    Full_Corpus/ANA/output/

Corrected article-layer outputs:
    Full_Corpus/CORPUS_article/data/review_master.csv
    Full_Corpus/ANA_article/output/analysis_master_clean.csv
    Full_Corpus/ANA_article/output/timeline/
    Full_Corpus/ANA_article/output/chapter_5_2/
    Full_Corpus/ANA_article/output/chapter_5_3/
    Full_Corpus/ANA_article/output/diagnostics/
    Full_Corpus/ANA_article/output/cda_sample/

Source and republication layer:
    Full_Corpus/pilot/data/postStepB.csv

Russian-source supplementary layer:
    Full_Corpus/pilot/data/stepBA/

Article qualitative review corpus:
    Full_Corpus/pilot/data/stepEarticle/

Pilot reliability outputs:
    Final_Pilot/REL/data/

======================================================================
14. REPRODUCIBILITY CAUTIONS
======================================================================

1. The corpus is availability-based.
   It should not be interpreted as a statistically representative sample of all Malian media.

2. The retrieval strategy is explicit-term based.
   Articles referring indirectly to Russian instructors or contractors without target terminology may be underrepresented.

3. Online news is only one part of Mali’s information environment.
   Radio, television, social media, WhatsApp, TikTok, oral communication, and local-language media are not directly analysed.

4. Variables differ in reliability.
   Pilot reliability was stronger for manifest variables such as actor mention, successor framing, location, and several frame indicators. Stance, legitimation, geopolitical rivalry, and dominant discourse were less stable and should be interpreted cautiously.

5. LLM outputs are not treated as final truth.
   They are constrained through prompts, parsed, validated, compared with the pipeline, and conservatively adjudicated.

6. Source/republication detection is diagnostic.
   It does not prove editorial endorsement, political alignment, or authorial intent.

7. KWIC, lexical, NER, BERTopic, and word-cloud outputs are exploratory.
   They support interpretation and future research but do not replace close reading or codebook-based analysis.

8. API-based outputs may change over time.
   Gemini API behaviour may not be perfectly reproducible if model versions or API infrastructure change.

9. Local Ollama outputs depend on local hardware and model versions.

10. The original thesis and corrected article workflows use distinct downstream analytical layers.
    Original thesis statistics should be reproduced from:
        Full_Corpus/CORPUS/
        Full_Corpus/ANA/

    Corrected article statistics should be reproduced from:
        Full_Corpus/CORPUS_article/
        Full_Corpus/ANA_article/

======================================================================
15. INTERPRETING OUTPUTS
======================================================================

Primary original thesis outputs:
    postConsolidated.csv
    original CORPUS/data/review_master.csv
    original ANA/output/

Primary corrected article outputs:
    Gemini conservative adjudicated table
    CORPUS_article/data/review_master.csv
    ANA_article/output/

Diagnostic outputs:
    parse errors
    merge issues
    manual verification tables
    review flags
    duplicate clusters
    source/republication profiles
    provenance documentation

Supplementary outputs:
    KWIC
    lexical normalization
    word clouds
    NER
    BERTopic
    synthesis tables

The corrected article layer should be used for future article-specific statistical reporting. The original thesis layer remains preserved for faithful reproduction of the submitted thesis.

======================================================================
16. CITATION
======================================================================

If using or referring to this code package, cite:

    Vrána, P. (2026). Media Discourses of the Wagner Group in Malian Online Press:
    Replication materials [Data set and code repository]. GitHub.
    https://github.com/v0987654321/diploma-thesis

Associated thesis:

    Vrána, P. (2026). Media Discourses of the Wagner Group in Malian Online Press
    [Master’s thesis, Mendel University in Brno]. Theses.cz.
    https://theses.cz/id/ieu87x/

======================================================================
17. FINAL NOTE
======================================================================

This repository is an open empirical and methodological workspace rather than only a collection of final results. It contains the scripts and outputs necessary to inspect corpus construction, preprocessing, coding, pilot validation, model-assisted adjudication, source/republication enrichment, qualitative selection, and chapter-oriented analysis.

The original thesis workflow is retained unchanged. The corrected article workflow was introduced to repair a downstream provenance issue in which conservative Gemini adjudication was not merged into the original CORPUS-to-ANA analytical layer. The article workflow therefore provides a separate, documented, and reproducible basis for future journal-article analysis.

Researchers are encouraged to inspect, reuse, verify, correct, extend, or reanalyse the available materials, subject to appropriate citation and to legal and ethical restrictions concerning copyrighted news texts.

This README was prepared with generative AI assistance and reviewed by the author.
