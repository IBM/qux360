from pathlib import Path
from pyqual.core.interview import Interview
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend
from mellea.backends.litellm import LiteLLMBackend
from mellea import MelleaSession
from dotenv import load_dotenv
from mellea.backends.types import ModelOption
from ibm_watsonx_ai.foundation_models import ModelInference
import os
import json
import logging
logging.getLogger("pyqual.core.interview").setLevel(logging.INFO)

load_dotenv()
m = MelleaSession(backend=WatsonxAIBackend(model_id=os.getenv("MODEL_ID")))
data_dir = Path(__file__).parent / "data"
file = data_dir / "interview_A.csv"
participant_id = "P1"
config_file = Path(__file__).parent / "config.json"

# loads config file to get the headers names provided by the user
try:
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    # create an instance with headers config
    i = Interview(file, headers=config['headers'])
except ValueError as e:
    raise e
except:
    # create an instance without headers config
    i = Interview(file)

# see what we loaded
print("\nTranscript we are working with:")
i.show(10)

result = i.identify_interviewee()
print(f"\nInterview participant: {result.result}")
print(f"Validation: {result.validation.status} - {result.validation.explanation}")

# conducting top-down thematic analysis
print("\nIdentifying and suggesting themes:")

topics_result = i.suggest_topics_top_down(m, interview_context="Remote Work")

# Access the validated topics
if topics_result.passed_validation():
    print(f"\n✅ Topic extraction validated: {topics_result.validation.status}")
    for idx, topic in enumerate(topics_result.result, start=1):
        print(f"\n{idx}. {topic.topic}")
        print(f"   → {topic.explanation}")
        print(f"   → Quotes: {len(topic.quotes)}")
        for q in topic.quotes[:2]:  # Show first 2 quotes
            print(f"      - [{q.index}] {q.quote[:80]}...")
else:
    print(f"⚠️ Topic extraction had issues: {topics_result.validation.explanation}")

# Show per-topic validation details using the built-in method
topics_result.print_summary(
    title="Per-Topic Validation Details",
    item_label="Topic"
)
