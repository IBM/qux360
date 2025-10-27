"""
Example: Fast iteration on theme analysis using cached interviews + topics

This script demonstrates automatic caching workflow:

FIRST RUN (no cache):
  1. Loads interviews from source files
  2. Identifies interviewees
  3. Extracts topics (expensive LLM calls)
  4. Saves interviews + topics to cache
  5. Runs theme analysis

SUBSEQUENT RUNS (cache exists):
  1. Loads interviews + topics from cache (fast!)
  2. Skips topic extraction
  3. Runs theme analysis (can iterate quickly)
"""

from pathlib import Path
from pyqual.core.study import Study
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(name)s - %(message)s'
)
logging.getLogger("pyqual").setLevel(logging.INFO)

load_dotenv()
m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID")))
logging.getLogger('fancy_logger').setLevel(logging.WARNING)

data_dir = Path(__file__).parent / "data"

print("=" * 60)
print("SMART LOADING: Interviews + Topics")
print("=" * 60)

interview_files = [
    data_dir / "interview_A.csv",
    data_dir / "interview_B.csv",
    data_dir / "interview_C.csv"
]

# Load study with automatic caching (default behavior)
# - First run: parses files, creates .pyqual_cache/ for each interview
# - Subsequent runs: loads from cache
study = Study(
    interview_files,
    study_context="A qualitative study about remote work experiences and challenges",
    use_cache=True  # This is the default
)

print(f"\n✅ Study loaded: {len(study)} interviews")
print(f"   Context: {study.study_context}")

# Check if we need to run topic extraction
needs_topic_extraction = False
for interview in study.documents:
    if interview.topics_top_down is None:
        needs_topic_extraction = True
        break

if needs_topic_extraction:
    print("\n" + "=" * 60)
    print("FIRST RUN: Extracting topics (this will be cached)")
    print("=" * 60)
    print("\n⏳ This may take a few minutes...")

    # Identify interviewees
    print("\n→ Step 1: Identifying interviewees...")
    results = study.identify_interviewees(m)
    for interview_id, result in results.items():
        print(f"   {interview_id}: {result.result} ({result.validation.status})")

    # Extract topics (expensive!)
    print("\n→ Step 2: Extracting topics from all interviews...")
    topics_results = study.suggest_topics_all(m)

    for interview_id, topics_result in topics_results.items():
        if topics_result.passed_validation():
            print(f"   ✅ {interview_id}: {len(topics_result.result)} topics")
        else:
            print(f"   ⚠️ {interview_id}: {topics_result.validation.explanation}")

    # Save interviews with topics to cache
    print("\n→ Step 3: Saving to cache for next run...")
    for interview in study.documents:
        interview.save_state()  # Saves to .pyqual_cache/ next to source file

    print("\n✅ Topics extracted and cached!")
    print("   Next time you run this script, it will load instantly from cache.\n")
else:
    print("\n" + "=" * 60)
    print("CACHE HIT: Topics loaded from cache")
    print("=" * 60)
    print("\n⚡ Skipping topic extraction - using cached results!")

    # Show cached topics with full validation summary using print_summary()
    for interview in study.documents:
        if interview.topics_top_down:
            participant = interview.get_participant_id() or interview.id

            # Reconstruct ValidatedList from cached data
            topics_validated = interview.get_topics_validated()
            if topics_validated:
                # Use the built-in print_summary() method
                topics_validated.print_summary(
                    title=f"Cached Topics for {participant}",
                    item_label="Topic"
                )
                print()  # Extra spacing between interviews

# Now run theme analysis (always runs, not cached in this script)
print("=" * 60)
print("THEME ANALYSIS: Identifying cross-cutting patterns")
print("=" * 60)

print("\n🔍 Analyzing themes across all interviews...")
print("   (Theme analysis always runs - not cached for flexibility)\n")

# Try different theme extraction parameters
themes_result = study.suggest_themes(
    m,
    max_quotes_per_topic=3,  # Fewer quotes for faster processing
    max_quote_length=300
)

# Display results
if themes_result and themes_result.result:
    print(f"\n✅ Theme extraction completed: {themes_result.validation.status}")
    print(f"   {themes_result.validation.explanation}\n")

    for idx, theme in enumerate(themes_result.result.themes, start=1):
        print(f"\n{'=' * 60}")
        print(f"THEME {idx}: {theme.title}")
        print(f"{'=' * 60}")
        print(f"\nDescription: {theme.description}")
        print(f"\nExplanation: {theme.explanation[:200]}...")
        print(f"\nSupporting Topics: {len(theme.topics)}")
        for topic in theme.topics[:3]:  # Show first 3 topics
            print(f"  • {topic.topic} ({len(topic.quotes)} quotes)")
        if len(theme.topics) > 3:
            print(f"  ... and {len(theme.topics) - 3} more")
else:
    print(f"⚠️ Theme extraction failed: {themes_result.validation.explanation if themes_result else 'No result'}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\n📝 Summary:")
print(f"  • {len(study)} interviews processed")
if needs_topic_extraction:
    print(f"  • Topics extracted and cached to .pyqual_cache/")
else:
    print(f"  • Topics loaded from cache (instant!)")
print(f"  • {len(themes_result.result.themes) if themes_result and themes_result.result else 0} themes identified")