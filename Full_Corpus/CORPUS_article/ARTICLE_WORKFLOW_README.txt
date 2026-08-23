ARTICLE-SPECIFIC WORKFLOW

Created: 2026-08-17T14:37:00

Purpose
-------
This directory pair was generated from the original CORPUS and ANA workflows
to create an article-specific analytical layer based on the Gemini conservative
adjudicated coding output.

Key correction
--------------
The original CORPUS -> ANA workflow used pipeline coding values from
review_master.csv because the adjudicated Adj_V* values were not merged into
review_master. This article workflow corrects that issue.

CORPUS_article/corpus1.py now:
1. merges final_conservative_adjudicated_table.csv into review_master;
2. preserves original pipeline coding as Pipeline_* variables;
3. creates Final_* variables using Gemini conservative coding with pipeline fallback;
4. overwrites operational downstream coding variables with Final_* values so that
   copied corpus scripts use the article-specific final layer.

ANA_article/ANALYSIS.py now reads:
    CORPUS_article/data/review_master.csv

Run order
---------
1. cd to:
   C:\Users\Petr\Downloads\github_download\diploma-thesis\Full_Corpus\CORPUS_article

2. Run:
   python corpus1.py
   python corpus2.py
   python corpus3.py
   python corpus4.py
   python corpus5.py
   python corpus6.py       [optional; requires BERTopic stack]
   python corpus7.py       [optional; requires French spaCy model]
   python corpus8.py       [optional synthesis layer]

3. cd to:
   C:\Users\Petr\Downloads\github_download\diploma-thesis\Full_Corpus\ANA_article

4. Run:
   python ANALYSIS.py --all

Important
---------
- Original CORPUS and ANA folders are untouched.
- Existing thesis outputs remain preserved in the original folders.
- Before using article results, compare article-specific totals and figures
  with thesis-layer results.
- The article should explicitly state which final coding layer is used.
