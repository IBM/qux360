from pathlib import Path
from pyqual.core.study import Study
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os

load_dotenv()

m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID_LITELLM")))

data_dir = Path(__file__).parent / "data"

file1 = data_dir / "interview_A.csv"
file2 = data_dir / "interview_B.docx"
file3 = data_dir / "interview_C.xlsx"

study = Study([file1, file2, file3])

# compute all participants automatically (data is not anonymized yet)
print("\nIdentifying interviewees across all interviews")
results = study.identify_interviewees(m)
print(results)

# anonymize the speakers across all interviews
print("\nAnonymizing speakers across all interviews")
results = study.anonymize_speakers()
print(results)

# printing study summary
print("\nThe study object:")
print(study)

# iterate through all interviews
for i in study:
    print(f"\nInterview with {i.get_participant_id()}")
    print(i)
    print(f"Speaker Mapping: {i.speaker_mapping}")


