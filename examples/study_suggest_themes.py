from pathlib import Path
from pyqual.core.study import Study
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Root: suppress all libraries by default
    format='%(message)s'
)

# Enable INFO logging only for pyqual
logging.getLogger("pyqual").setLevel(logging.INFO)

load_dotenv()
#m = MelleaSession(backend=WatsonxAIBackend(model_id=os.getenv("MODEL_ID_WATSONX")))
m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID")))

# Suppress Mellea's FancyLogger (MelleaSession resets it to DEBUG, so we set it here)
logging.getLogger('fancy_logger').setLevel(logging.WARNING)
data_dir = Path(__file__).parent / "data"

# Load multiple interviews into a study
print("=" * 60)
print("STEP 1: Loading interviews into study")
print("=" * 60)

interview_files = [
    data_dir / "interview_A.csv",
    data_dir / "interview_B.csv",
    data_dir / "interview_C.csv"
]

# Disable caching to show complete pipeline from scratch
study = Study(
    interview_files,
    study_context="A qualitative study about remote work experiences and challenges",
    use_cache=False  # Disable cache to run full pipeline
)
print(f"\nLoaded study: {study}")
print(f"Number of interviews: {len(study)}")
print(f"Study context: {study.study_context}")
print("\n💡 Note: This example runs the full pipeline without caching.")
print("   For fast iteration with caching, see: study_suggest_themes_from_cache.py")

# Identify interviewees for each interview
print("\n" + "=" * 60)
print("STEP 2: Identifying interviewees")
print("=" * 60)

# Use Study method instead of looping manually
results = study.identify_interviewees()
for interview_id, result in results.items():
    print(f"\nInterview {interview_id}:")
    print(f"  Participant: {result.result}")
    print(f"  Validation: {result.validation.status} - {result.validation.explanation}")

# Extract topics from each interview
print("\n" + "=" * 60)
print("STEP 3: Extracting topics from each interview")
print("=" * 60)

# Use Study method to extract topics from all interviews
# This automatically uses study.study_context for all interviews
topics_results = study.suggest_topics_all(m)

for interview_id, topics_result in topics_results.items():
    if topics_result.passed_validation():
        print(f"\n✅ Interview {interview_id}: Extracted {len(topics_result.result)} topics")
        for topic in topics_result.result:
            print(f"  • {topic.topic}")
    else:
        print(f"\n⚠️ Interview {interview_id}: Topic extraction had issues")
        print(f"   {topics_result.validation.explanation}")

# Extract themes across all interviews
print("\n" + "=" * 60)
print("STEP 4: Extracting cross-cutting themes")
print("=" * 60)

themes_result = study.suggest_themes(
    m,
    # study_context is inherited from Study constructor
    max_quotes_per_topic=5,  # Limit quotes per topic to control prompt size
    max_quote_length=500     # Truncate long quotes to 500 characters
)

# Display themes using print_summary
if themes_result and themes_result.result:
    # Use built-in print_summary to show themes with validation
    themes_result.print_summary(title="Theme Analysis Results", item_label="Theme")
else:
    print(f"⚠️ Theme extraction failed: {themes_result.validation.explanation if themes_result else 'No result'}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\n📝 Summary:")
print(f"  • {len(study)} interviews processed")
print(f"  • Topics extracted from all interviews")
print(f"  • {len(themes_result.result) if themes_result and themes_result.result else 0} cross-cutting themes identified")
print("\n💡 For faster iteration:")
print("   Run study_suggest_themes_from_cache.py to cache topics and iterate on theme analysis")
