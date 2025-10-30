"""
Example: Fast iteration on thematic analysis using cached interviews + topics

This script demonstrates interactive caching workflow:

CACHE CHECK:
  - If cache exists: Prompts user to load from cache (y/n)
  - If cache doesn't exist: Loads from source files

FIRST RUN (no cache) or USER DECLINES CACHE:
  1. Loads interviews from source files
  2. Identifies interviewees
  3. Extracts topics
  4. Saves interviews + topics to cache
  5. Runs thematic analysis

SUBSEQUENT RUNS (cache exists and user accepts):
  1. Loads interviews + topics from cache (fast!)
  2. Skips topic extraction
  3. Runs thematic analysis (can iterate quickly)
"""

from pathlib import Path
from qux360.core.study import Study
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os
import logging

# Configure logging, silence all libs, enable ours
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logging.getLogger("qux360").setLevel(logging.INFO)

load_dotenv()

ROOT_DIR = Path.cwd()
data_dir = ROOT_DIR.joinpath("examples/data")
interview_files = [data_dir.joinpath("interview_A.csv"), data_dir.joinpath("interview_B.csv"), data_dir.joinpath("interview_C.csv")]

m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID")))

# Disable mellea's progress bar
logging.getLogger('fancy_logger').setLevel(logging.WARNING)

print("=" * 60)
print("SMART LOADING: Interviews + Topics")
print("=" * 60)

# Check if cache exists on disk
cache_dir = os.path.join(data_dir, ".qux360_cache")
cache_exists = os.path.exists(cache_dir) and os.path.isdir(cache_dir)

use_cache = True  # Default behavior

if cache_exists:
    print(f"\n📁 Cache directory found: {cache_dir}")
    user_choice = input("🔄 Load interviews from cache? (y/n): ").strip().lower()
    use_cache = (user_choice == 'y')
    if use_cache:
        print("⚡ Loading from cache...")
    else:
        print("🔄 Loading from source files...")
else:
    print("\n📁 No cache found - loading from source files...")

# Load study with caching based on user choice
# - use_cache=True: loads from .qux360_cache/ if available
# - use_cache=False: parses from source files
study = Study(
    interview_files,
    study_context="A qualitative study about remote work experiences and challenges",
    use_cache=use_cache
)

print(f"\n✅ Study loaded: {len(study)} interviews")
print(f"   Context: {study.study_context}")

# Check if we need to run topic extraction
needs_topic_extraction = any(interview.topics_top_down is None for interview in study.documents)

if needs_topic_extraction:
    print("\n" + "=" * 60)
    print("TOPIC EXTRACTION: Extracting topics from interviews (may take a few minutes)")
    print("=" * 60)

    # Identify interviewees
    print("\n→ Step 1: Identifying interviewees...")
    results = study.identify_interviewees(m)
    for interview_id, result in results.items():
        print(f"   {interview_id}: {result.result} ({str(result.validation)})")

    # Extract topics (expensive!)
    print("\n→ Step 2: Extracting topics from all interviews...")
    topics_results = study.suggest_topics_all(m)

    for interview_id, topics_result in topics_results.items():
        if topics_result.passed_validation():
            print(f"   ✅ {interview_id}: {len(topics_result.result)} topics")
        else:
            print(f"   ⚠️ {interview_id}: {topics_result.validation.explanation}")

    # Save interviews with topics to cache
    print("\n→ Step 3: Saving to cache...")
    for interview in study.documents:
        interview.save_state()

    print("\n✅ Topics extracted and cached!")
else:
    print("\n" + "=" * 60)
    print("TOPICS LOADED: Using cached topics")
    print("=" * 60)

    # Show cached topics with condensed validation summary
    for interview in study.documents:
        if interview.topics_top_down:
            participant = interview.get_participant_id() or interview.id

            # Reconstruct ValidatedList from cached data
            topics_validated = interview.get_topics_validated()
            if topics_validated:
                # Use condensed mode for cleaner output
                topics_validated.print_summary(
                    title=f"Cached Topics for {participant}",
                    item_label="Topic",
                    condensed=True
                )
                print()  # Extra spacing between interviews

# Now run theme analysis (always runs, not cached in this script)
print("=" * 60)
print("THEMATIC ANALYSIS: Identifying cross-cutting patterns")
print("=" * 60)

print("\n🔍 Analyzing themes across all interviews...\n")

# Try different theme extraction parameters
themes_result = study.suggest_themes(m)


# Display themes using print_summary
if themes_result and themes_result.result:
    themes_result.print_summary(title="Thematic Analysis Results", item_label="Theme")
else:
    print(f"⚠️ Theme extraction failed: {themes_result.validation.explanation if themes_result else 'No result'}")


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\n📝 Summary:")
print(f"  • {len(study)} interviews processed")
if needs_topic_extraction:
    print(f"  • Topics extracted and cached to .qux360_cache/")
else:
    print(f"  • Topics loaded from cache")
print(f"  • {len(themes_result.result) if themes_result and themes_result.result else 0} themes identified")