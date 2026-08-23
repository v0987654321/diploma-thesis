# export_discussion_inputs.py
# Place this script inside the CORPUS/ folder and run:
# python export_discussion_inputs.py
#
# Purpose:
# Export the most useful CORPUS workspace outputs for writing the Discussion chapter.
# The script does not create new analytical results. It only collects already generated
# synthesis, lexical, NER, topic, source/republication, and outlet-difference files.

from pathlib import Path
import shutil
import csv
import zipfile
from datetime import datetime


# ---------------------------------------------------------------------
# 1. Base paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "discussion_export"
ZIP_PATH = BASE_DIR / "discussion_export_bundle.zip"

EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Files recommended for Discussion chapter writing
# ---------------------------------------------------------------------
# Each item:
#   label: human-readable description
#   rel_path: path relative to CORPUS/
#   priority: high / medium / optional
#   note: why useful

FILES_TO_EXPORT = [
    # -----------------------------------------------------------------
    # corpus8 synthesis outputs - highest priority
    # -----------------------------------------------------------------
    {
        "label": "Discussion working notes",
        "rel_path": "summary/discussion_working_notes.txt",
        "priority": "high",
        "note": "Main writer-oriented notes for the Discussion chapter."
    },
    {
        "label": "Key findings for discussion",
        "rel_path": "synthesis/key_findings_discussion.csv",
        "priority": "high",
        "note": "Candidate discussion findings produced by integrated synthesis."
    },
    {
        "label": "Corpus-wide synthesis",
        "rel_path": "synthesis/corpus_wide_synthesis.csv",
        "priority": "high",
        "note": "Integrated corpus-level findings."
    },
    {
        "label": "Actor representation synthesis",
        "rel_path": "synthesis/actor_representation_synthesis.csv",
        "priority": "high",
        "note": "Useful for Wagner vs Africa Corps interpretation."
    },
    {
        "label": "Frame and discourse synthesis",
        "rel_path": "synthesis/frame_discourse_synthesis.csv",
        "priority": "high",
        "note": "Useful for discussing thematic frames and discourse-level interpretation."
    },
    {
        "label": "Source and republication synthesis",
        "rel_path": "synthesis/source_republication_synthesis.csv",
        "priority": "high",
        "note": "Useful for circulation-environment discussion."
    },
    {
        "label": "Outlet difference synthesis",
        "rel_path": "synthesis/outlet_difference_synthesis.csv",
        "priority": "medium",
        "note": "Useful for Q5 and outlet-level interpretation."
    },
    {
        "label": "Wagner vs Africa Corps integrated comparison",
        "rel_path": "synthesis/wagner_vs_ac_integrated_comparison.csv",
        "priority": "high",
        "note": "Most useful direct comparison of Wagner-only and AC-or-both patterns."
    },
    {
        "label": "Source-attributed vs non-attributed comparison",
        "rel_path": "synthesis/source_attributed_vs_nonattributed_integrated_comparison.csv",
        "priority": "medium",
        "note": "Useful for discussing sourcing and circulation."
    },
    {
        "label": "Review vs high-confidence comparison",
        "rel_path": "synthesis/review_vs_highconfidence_integrated_comparison.csv",
        "priority": "optional",
        "note": "Useful for robustness/methodological reflection if available."
    },
    {
        "label": "Chapter 5.2 working notes",
        "rel_path": "summary/chapter52_working_notes.txt",
        "priority": "medium",
        "note": "May help connect corpus overview to Discussion."
    },
    {
        "label": "Corpus8 summary",
        "rel_path": "summary/corpus8_summary.txt",
        "priority": "medium",
        "note": "Summary of integrated synthesis layer."
    },

    # -----------------------------------------------------------------
    # NER outputs from corpus7
    # -----------------------------------------------------------------
    {
        "label": "Target entity profile",
        "rel_path": "ner/target_entity_profile.csv",
        "priority": "high",
        "note": "Useful for actor-relational interpretation around Wagner/Africa Corps."
    },
    {
        "label": "Target entity profile sentence level",
        "rel_path": "ner/target_entity_profile_sentence_level.csv",
        "priority": "high",
        "note": "Useful for sentence-level co-occurrence and Q3 interpretation."
    },
    {
        "label": "Top entities overall",
        "rel_path": "ner/top_entities_overall.csv",
        "priority": "medium",
        "note": "Useful for identifying dominant entity environment."
    },
    {
        "label": "Top entities by outlet",
        "rel_path": "ner/top_entities_by_outlet.csv",
        "priority": "medium",
        "note": "Useful for outlet-level differences."
    },
    {
        "label": "Top entities by label",
        "rel_path": "ner/top_entities_by_label.csv",
        "priority": "optional",
        "note": "Useful if discussing entity types."
    },
    {
        "label": "Target/entity co-occurrence article level",
        "rel_path": "ner/target_entity_cooccurrence_article_level.csv",
        "priority": "medium",
        "note": "Useful for article-level actor association patterns."
    },
    {
        "label": "Entity evidence top30",
        "rel_path": "ner/entity_evidence_top30.csv",
        "priority": "optional",
        "note": "Useful for examples and evidence inspection."
    },
    {
        "label": "NER notes",
        "rel_path": "summary/corpus7_ner_notes.txt",
        "priority": "medium",
        "note": "Narrative notes from entity enrichment."
    },

    # -----------------------------------------------------------------
    # KWIC / lexical outputs from corpus3 and corpus4
    # -----------------------------------------------------------------
    {
        "label": "Lexical findings notes",
        "rel_path": "summary/corpus4_lexical_findings.txt",
        "priority": "high",
        "note": "Useful for naming practices and lexical triangulation."
    },
    {
        "label": "KWIC keyword summary",
        "rel_path": "concordance/kwic_keyword_summary.csv",
        "priority": "medium",
        "note": "Keyword frequencies and context coverage."
    },
    {
        "label": "KWIC group summary",
        "rel_path": "concordance/kwic_group_summary.csv",
        "priority": "high",
        "note": "Useful for lexical groups: Wagner, Africa Corps, France, Russia, sovereignty, etc."
    },
    {
        "label": "Keyword group subset profile",
        "rel_path": "lexical/keyword_group_subset_profile.csv",
        "priority": "high",
        "note": "Useful for comparing lexical groups across actor/source/relevance subsets."
    },
    {
        "label": "Outlet keyword profile",
        "rel_path": "lexical/outlet_keyword_profile.csv",
        "priority": "medium",
        "note": "Useful for outlet-level lexical tendencies."
    },
    {
        "label": "Repeated sentence patterns",
        "rel_path": "lexical/repeated_sentence_patterns.csv",
        "priority": "medium",
        "note": "Useful for identifying repeated formulations and republication/circulation traces."
    },
    {
        "label": "KWIC coverage profile",
        "rel_path": "lexical/kwic_coverage_profile.csv",
        "priority": "optional",
        "note": "Useful for checking KWIC coverage."
    },

    # -----------------------------------------------------------------
    # Lexical normalization from corpus5
    # -----------------------------------------------------------------
    {
        "label": "All token profiles",
        "rel_path": "lexical_norm/all_token_profiles.csv",
        "priority": "medium",
        "note": "Useful for normalized lexical inspection."
    },
    {
        "label": "Analytic group profile",
        "rel_path": "lexical_norm/analytic_group_profile.csv",
        "priority": "medium",
        "note": "Useful for broader analytical lexical groupings."
    },
    {
        "label": "Mode comparison summary",
        "rel_path": "lexical_norm/mode_comparison_summary.csv",
        "priority": "optional",
        "note": "Useful for methodological reflection on lexical normalization."
    },
    {
        "label": "Lexical normalization notes",
        "rel_path": "summary/corpus5_normalization_notes.txt",
        "priority": "optional",
        "note": "Notes on normalization choices."
    },

    # -----------------------------------------------------------------
    # BERTopic exploratory outputs from corpus6
    # -----------------------------------------------------------------
    {
        "label": "BERTopic summary overall",
        "rel_path": "summary/corpus6_summary.txt",
        "priority": "medium",
        "note": "Summary of exploratory topic modelling."
    },
    {
        "label": "BERTopic summary table",
        "rel_path": "topics/corpus6_summary.csv",
        "priority": "medium",
        "note": "Overview of topic model runs."
    },

    # relevance4 topic outputs
    {
        "label": "BERTopic relevance4 topic summary",
        "rel_path": "topics/relevance4/topic_model_summary.txt",
        "priority": "high",
        "note": "Exploratory topic summary for main-topic/high-salience subset."
    },
    {
        "label": "BERTopic relevance4 topic info",
        "rel_path": "topics/relevance4/bertopic_topic_info.csv",
        "priority": "medium",
        "note": "Topic info for relevance4 subset."
    },
    {
        "label": "BERTopic relevance4 topic terms",
        "rel_path": "topics/relevance4/bertopic_topic_terms.csv",
        "priority": "medium",
        "note": "Topic terms for relevance4 subset."
    },
    {
        "label": "BERTopic relevance4 representative docs",
        "rel_path": "topics/relevance4/bertopic_representative_docs.csv",
        "priority": "optional",
        "note": "Representative docs for relevance4 topics."
    },

    # africa_corps_or_both topic outputs
    {
        "label": "BERTopic Africa Corps or both topic summary",
        "rel_path": "topics/africa_corps_or_both/topic_model_summary.txt",
        "priority": "high",
        "note": "Exploratory topic summary for Africa Corps / both subset."
    },
    {
        "label": "BERTopic Africa Corps or both topic info",
        "rel_path": "topics/africa_corps_or_both/bertopic_topic_info.csv",
        "priority": "medium",
        "note": "Topic info for Africa Corps / both subset."
    },
    {
        "label": "BERTopic Africa Corps or both topic terms",
        "rel_path": "topics/africa_corps_or_both/bertopic_topic_terms.csv",
        "priority": "medium",
        "note": "Topic terms for Africa Corps / both subset."
    },
    {
        "label": "BERTopic Africa Corps or both representative docs",
        "rel_path": "topics/africa_corps_or_both/bertopic_representative_docs.csv",
        "priority": "optional",
        "note": "Representative docs for Africa Corps / both topics."
    },

    # -----------------------------------------------------------------
    # Source/republication and corpus profiles from corpus1
    # -----------------------------------------------------------------
    {
        "label": "Source republication profile",
        "rel_path": "data/source_republication_profile.csv",
        "priority": "high",
        "note": "Core source/republication descriptive profile."
    },
    {
        "label": "Corpus overview",
        "rel_path": "data/corpus_overview.csv",
        "priority": "medium",
        "note": "Basic corpus overview."
    },
    {
        "label": "Representation profile",
        "rel_path": "data/representation_profile.csv",
        "priority": "medium",
        "note": "Basic representation profile."
    },
    {
        "label": "Frame profile",
        "rel_path": "data/frame_profile.csv",
        "priority": "medium",
        "note": "Basic frame profile."
    },
    {
        "label": "Outlet profile",
        "rel_path": "data/outlet_profile.csv",
        "priority": "medium",
        "note": "Basic outlet profile."
    },
    {
        "label": "Relevance profile",
        "rel_path": "data/relevance_profile.csv",
        "priority": "medium",
        "note": "Basic relevance profile."
    },

    # -----------------------------------------------------------------
    # Main review master, useful but large
    # -----------------------------------------------------------------
    {
        "label": "Review master",
        "rel_path": "data/review_master.csv",
        "priority": "optional",
        "note": "Full review master dataset; useful if deeper checking is needed."
    },
]


# ---------------------------------------------------------------------
# 3. Helper functions
# ---------------------------------------------------------------------

def safe_dest_name(rel_path: str) -> str:
    """
    Convert nested path into a safe filename for flat export directory.
    Example:
        synthesis/key_findings_discussion.csv
        -> synthesis__key_findings_discussion.csv
    """
    return rel_path.replace("/", "__").replace("\\", "__")


def copy_file(src: Path, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return True
    except Exception as e:
        print(f"[ERROR] Could not copy {src} -> {dest}: {e}")
        return False


# ---------------------------------------------------------------------
# 4. Export selected files
# ---------------------------------------------------------------------

manifest_rows = []
found_count = 0
missing_count = 0

for item in FILES_TO_EXPORT:
    rel_path = item["rel_path"]
    src = BASE_DIR / rel_path
    dest_name = safe_dest_name(rel_path)
    dest = EXPORT_DIR / dest_name

    exists = src.exists() and src.is_file()

    if exists:
        copied = copy_file(src, dest)
        status = "copied" if copied else "copy_error"
        if copied:
            found_count += 1
        else:
            missing_count += 1
    else:
        status = "missing"
        missing_count += 1

    manifest_rows.append({
        "label": item["label"],
        "priority": item["priority"],
        "status": status,
        "source_relative_path": rel_path,
        "export_filename": dest_name if exists else "",
        "note": item["note"]
    })


# ---------------------------------------------------------------------
# 5. Write manifest CSV
# ---------------------------------------------------------------------

manifest_path = EXPORT_DIR / "discussion_export_manifest.csv"

with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "label",
            "priority",
            "status",
            "source_relative_path",
            "export_filename",
            "note"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest_rows)


# ---------------------------------------------------------------------
# 6. Write README
# ---------------------------------------------------------------------

readme_path = EXPORT_DIR / "discussion_export_readme.txt"

readme_text = f"""Discussion Export Package
Generated: {datetime.now().isoformat(timespec="seconds")}

Base directory:
{BASE_DIR}

Export directory:
{EXPORT_DIR}

Files copied:
{found_count}

Files missing or not copied:
{missing_count}

Purpose:
This export collects the most relevant CORPUS workspace outputs for writing the Discussion chapter.
The exported files include synthesis tables, discussion working notes, actor comparison tables,
source/republication profiles, lexical/KWIC summaries, NER outputs, and exploratory BERTopic outputs.

Methodological caution:
These files are supporting and triangulating evidence layers. They do not replace the primary
codebook-based pipeline, Gemini adjudication outputs, or qualitative CDA. Exploratory outputs
such as BERTopic clusters, lexical summaries, KWIC profiles, and NER co-occurrence tables should
be interpreted as heuristic aids, not standalone proof of substantive claims.

Recommended use:
1. Start with:
   - summary__discussion_working_notes.txt
   - synthesis__key_findings_discussion.csv
   - synthesis__wagner_vs_ac_integrated_comparison.csv
   - synthesis__frame_discourse_synthesis.csv
   - synthesis__source_republication_synthesis.csv

2. Then inspect:
   - ner__target_entity_profile.csv
   - ner__target_entity_profile_sentence_level.csv
   - summary__corpus4_lexical_findings.txt
   - topics__relevance4__topic_model_summary.txt
   - topics__africa_corps_or_both__topic_model_summary.txt

3. Use the manifest to check which expected files were available.

Manifest:
discussion_export_manifest.csv
"""

readme_path.write_text(readme_text, encoding="utf-8")


# ---------------------------------------------------------------------
# 7. Create ZIP archive
# ---------------------------------------------------------------------

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for file_path in EXPORT_DIR.rglob("*"):
        if file_path.is_file():
            arcname = file_path.relative_to(EXPORT_DIR)
            zf.write(file_path, arcname=arcname)


# ---------------------------------------------------------------------
# 8. Print summary
# ---------------------------------------------------------------------

print("=" * 80)
print("DISCUSSION EXPORT COMPLETE")
print("=" * 80)
print(f"Base directory:   {BASE_DIR}")
print(f"Export directory: {EXPORT_DIR}")
print(f"ZIP archive:      {ZIP_PATH}")
print(f"Files copied:     {found_count}")
print(f"Files missing:    {missing_count}")
print()
print(f"Manifest written to: {manifest_path}")
print(f"README written to:   {readme_path}")
print("=" * 80)

# Print high-priority missing files for quick diagnosis
missing_high = [
    row for row in manifest_rows
    if row["status"] == "missing" and row["priority"] == "high"
]

if missing_high:
    print("\nHigh-priority missing files:")
    for row in missing_high:
        print(f" - {row['source_relative_path']} ({row['label']})")
else:
    print("\nNo high-priority files missing.")