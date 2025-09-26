from pathlib import Path
from pyqual.core.interview import Interview
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend
from mellea import MelleaSession
from dotenv import load_dotenv
import os

load_dotenv()
m = MelleaSession(backend=WatsonxAIBackend(model_id=os.getenv("MODEL_ID")))
file = Path(__file__).parent / "data" / "P5.xlsx"



# create an instance
i = Interview(file)

# see what we loaded
i.show(10)

# look at the speakers
print(i.get_speakers())

# use heuristics and AI to find interviewee
print(i.identify_interviewee(m))

# anonmyze speakers
map = i.anonymize_speakers(i.identify_interviewee(m))
print("Anonymization mapping:")
for original, anon in map.items():
    print(f"  {original} -> {anon}")

# let's see how it looks like
i.show(10)
