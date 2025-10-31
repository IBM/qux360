from pathlib import Path
from qux360.core import Interview
from mellea import MelleaSession
#from mellea.backends.watsonx import WatsonxAIBackend
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Enable INFO logging only for qux360
logging.getLogger("qux360").setLevel(logging.INFO)

load_dotenv()

ROOT_DIR = Path.cwd()
data_dir = ROOT_DIR.joinpath("examples/data")
file = data_dir.joinpath("interview_A.csv")
config_file = ROOT_DIR.joinpath("examples/config.json")

m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID")))

# Suppress Mellea's FancyLogger (MelleaSession resets it to DEBUG, so we set it here)
logging.getLogger('fancy_logger').setLevel(logging.WARNING)


# STEP 1: Load interview
print("=" * 60)
print("STEP 1: Loading interview")
print("=" * 60)

participant_id = "P1"


# [OPTION A] create an instance without headers config (has headers by default)

i = Interview(file)

# [OPTION B] create an instance with headers config
'''
try:
    # loads config file to get the headers names provided by the user
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    # create an instance with headers config
    i = Interview(file, headers=config['headers'])
except ValueError as e:
    raise e
'''

# see what we loaded
print("\nTranscript we are working with:")


print(f"\nLoaded interview: {i}")
print("\nTranscript preview:")
i.show(10)

# STEP 2: Identify interviewee
print("\n" + "=" * 60)
print("STEP 2: Identifying interviewee")
print("=" * 60)

result = i.identify_interviewee(m)
identification = result.result
print(f"\nInterview participant: {identification.interviewee}")
print(f"Confidence: {identification.confidence}")
print(f"Explanation: {identification.explanation}")
print(f"Validation: {result.validation.status} - {result.validation.explanation}")

# STEP 3: Extract topics
print("\n" + "=" * 60)
print("STEP 3: Extracting topics from interview")
print("=" * 60)

topics_result = i.suggest_topics_top_down(m, interview_context="Remote Work")

# Access the validated topics
# if topics_result.passed_validation():
#     print(f"\n✅ Topic extraction validated: {topics_result.validation.status}")
#     for idx, topic in enumerate(topics_result.result, start=1):
#         print(f"\n{idx}. {topic.topic}")
#         print(f"   → {topic.explanation}")
#         print(f"   → Quotes: {len(topic.quotes)}")
#         for q in topic.quotes[:2]:  # Show first 2 quotes
#             print(f"      - [{q.index}] {q.quote[:80]}...")
# else:
#     print(f"⚠️ Topic extraction had issues: {topics_result.validation.explanation}")

# Show topics with validation details
topics_result.print_summary(
    title="Topic Extraction Results",
    item_label="Topic"
)
