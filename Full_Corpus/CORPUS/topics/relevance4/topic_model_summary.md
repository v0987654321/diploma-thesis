CORPUS6 TOPIC CLUSTERING SUMMARY
Subset: relevance4
Generated: 2026-04-25T19:43:43

Methodological note:
- This BERTopic analysis is exploratory and supplementary.
- It is intended for later discussion/triangulation, not as a primary coding layer.
- Topic texts were pre-cleaned and vectorized with French/custom stopword control.
- The script now distinguishes between finer-grained micro-topics and broader topic families.

Input:
- documents in subset after text-length filtering: 585

BERTopic micro-topic results:
- assigned documents: 585
- non-outlier micro-topics: 10
- outlier cluster size (topic -1): 176
- largest micro-topic: 0_wagner_russe_pays_russes (162 documents)

Topic-family results:
- broader topic families: 6
- largest topic family: mercenaires / wagner / russe / russes (251 topic-member documents)

Outputs:
- bertopic_document_topics.csv
- bertopic_topic_info.csv
- bertopic_topic_terms.csv
- bertopic_topic_families.csv
- bertopic_topic_family_members.csv
- bertopic_representative_docs.csv
- bertopic_topic_distribution_by_outlet.csv
- bertopic_topic_distribution_by_year.csv
- topic size and topic family figures (.png)