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

load_dotenv()

# m = MelleaSession(
#     backend=LiteLLMBackend(
#         model_id=os.getenv("MODEL_ID"),
#         model_options={"max_new_tokens": 5000, ModelOption.TEMPERATURE: 0.0})
#     )

# m = MelleaSession(
#     backend=WatsonxAIBackend(
#         model_id=os.getenv("MODEL_ID"),
#         model_options={ModelOption.MAX_NEW_TOKENS: 5000, ModelOption.TEMPERATURE: 0.0})
#     )

model = ModelInference(
    model_id=os.getenv("MODEL_ID"),
    credentials={
        "url":  os.getenv("WATSONX_URL"),
        "api_key": os.getenv("WATSONX_API_KEY")
    },
    project_id=os.getenv("WATSONX_PROJECT_ID")
)


data_dir = Path(__file__).parent / "data"
file = data_dir / "interview_A.csv"

participant_id = "P1"

# create an instance
i = Interview(file)

# see what we loaded
print("\nTranscript we are working with:")
i.show(10)

p, _ = i.identify_interviewee()
print(f"\nInterview participant: {p}")

# conducting top-down thematic analysis
print("\nIdentifying and suggesting themes:")

#result = i.suggest_topics_top_down(m, n=15, interview_context="Remote Work")
result = i.suggest_topics_top_down_wx(model, interview_context="Remote Work")

if result:
    for idx, topic in enumerate(result.topics, start=1):
        print(f"{idx}. {topic.topic}")
        print(f"   → {topic.explanation}\n")
else:
    print("⚠️ No topics returned")
