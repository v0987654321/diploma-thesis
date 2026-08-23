CORPUS6 TOPIC CLUSTERING SUMMARY
Subset: africa_corps_or_both
Generated: 2026-08-17T15:29:45

Methodological note:
- This BERTopic analysis is exploratory and supplementary.
- It is intended for later discussion/triangulation, not as a primary coding layer.
- Topic texts were pre-cleaned and vectorized with French/custom stopword control.
- The script now distinguishes between finer-grained micro-topics and broader topic families.

Input:
- documents in subset after text-length filtering: 143

BERTopic micro-topic results:
- assigned documents: 143
- non-outlier micro-topics: 4
- outlier cluster size (topic -1): 30
- largest micro-topic: 0_pays_afrique_militaire_sahel (47 documents)

Topic-family results:
- broader topic families: 4
- largest topic family: pays / afrique / militaire / sahel (47 topic-member documents)

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