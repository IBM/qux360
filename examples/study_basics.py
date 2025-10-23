from pathlib import Path
from pyqual.core.study import Study
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os
import json
from pyqual.core.interview import Interview

load_dotenv()

m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID_LITELLM")))

data_dir = Path(__file__).parent / "data"

file1 = data_dir / "interview_A.csv"
file2 = data_dir / "interview_B.docx"
file3 = data_dir / "interview_C.xlsx"

config_file = Path(__file__).parent / "config.json"

# [OPTION A] create an instance without headers config (has headers by default)
study = Study([file1, file2, file3])

# [OPTION B] create an instance with headers config
'''
try:
    # loads config file to get the headers names provided by the user
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # [OPTION B1] create an instance from files with headers config
    study = Study([file1, file2, file3], headers=[config['headers_study_1'], None, config['headers_study_3']], has_headers=[True, False, True])

    # [OPTION B2] create an instance from Interview objects
    # (given that headers are provided when creating the Interviews, they are not provided when creating the Study)
    #i1 = Interview(file1, headers=config['headers_study_1'], has_headers=True)
    #i2 = Interview(file2)
    #i3 = Interview(file3, headers=config['headers_study_3'], has_headers=True)
    #study = Study(files_or_docs=[i1, i2, i3])
except ValueError as e:
    raise e
'''

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


