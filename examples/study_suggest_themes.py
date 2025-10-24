from pathlib import Path
from pyqual.core.study import Study
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Root: suppress all libraries by default
    format='%(levelname)s - %(name)s - %(message)s'
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

study = Study(
    interview_files,
    study_context="A qualitative study about remote work experiences and challenges"
)
print(f"\nLoaded study: {study}")
print(f"Number of interviews: {len(study)}")
print(f"Study context: {study.study_context}")

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

# Display themes
if themes_result and themes_result.result:
    print(f"\n✅ Theme extraction completed: {themes_result.validation.status}")
    print(f"   {themes_result.validation.explanation}\n")

    for idx, theme in enumerate(themes_result.result.themes, start=1):
        print(f"\n{'=' * 60}")
        print(f"THEME {idx}: {theme.title}")
        print(f"{'=' * 60}")
        print(f"\nDescription: {theme.description}")
        print(f"\nExplanation: {theme.explanation}")
        print(f"\nSupporting Topics ({len(theme.topics)}):")

        for topic in theme.topics:
            print(f"\n  • {topic.topic}")
            print(f"    Explanation: {topic.explanation}")
            print(f"    Quotes ({len(topic.quotes)}):")
            for quote in topic.quotes[:3]:  # Show first 3 quotes
                print(f"      [{quote.index}] {quote.timestamp} {quote.speaker}:")
                print(f"      {quote.quote}\n")
            if len(topic.quotes) > 3:
                print(f"      ... and {len(topic.quotes) - 3} more\n")
else:
    print(f"⚠️ Theme extraction failed: {themes_result.validation.explanation if themes_result else 'No result'}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
