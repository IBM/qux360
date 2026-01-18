from pathlib import Path
from qux360.core import Study
import instructor
from litellm import completion
from dotenv import load_dotenv
import os
import logging

# Configure logging: suppress all libraries by default
logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Enable INFO logging only for qux360
logging.getLogger("qux360").setLevel(logging.INFO)

load_dotenv()

ROOT_DIR = Path.cwd()
data_dir = ROOT_DIR.joinpath("examples/data")
file1 = data_dir.joinpath("interview_A.csv")
file2 = data_dir.joinpath("interview_B.csv")
file3 = data_dir.joinpath("interview_C.csv")
config_file = ROOT_DIR.joinpath("examples/config.json")

# Create Instructor client for structured outputs
client = instructor.from_litellm(completion)


# STEP 1: Load study
print("=" * 60)
print("STEP 1: Loading study with multiple interviews")
print("=" * 60)

# [OPTION A] create an instance without headers config (has headers by default)
study = Study([file1, file2, file3])

# [OPTION B] create an instance with headers config
'''
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
'''

print(f"\nLoaded study: {study}")
print(f"Number of interviews: {len(study.documents)}")
for idx, interview in enumerate(study.documents, start=1):
    print(f"  {idx}. {interview.id}")

# STEP 2: Identify interviewees (using Instructor)
print("\n" + "=" * 60)
print("STEP 2: Identifying interviewees across all interviews")
print("=" * 60)

results = study.identify_interviewees(
    client=client,
    model=os.getenv("MODEL_ID"),
    max_retries=3
)

print(f"\n✅ Identified {len(results)} interviewees:")
for interview_id, validated_result in results.items():
    identification = validated_result.result
    print(f"\n  {interview_id}:")
    print(f"    Interviewee: {identification.interviewee}")
    print(f"    Confidence: {identification.confidence}")
    print(f"    Explanation: {identification.explanation}")
    print(f"\n    QIndex:")
    # Indent each line of the QIndex output
    for line in str(validated_result.validation).split('\n'):
        print(f"      {line}")

# STEP 3: Anonymize speakers
print("\n" + "=" * 60)
print("STEP 3: Anonymizing speakers across all interviews")
print("=" * 60)

mappings = study.anonymize_speakers()

print(f"\n✅ Anonymized speakers across {len(mappings)} interviews:")
for interview_id, mapping in mappings.items():
    print(f"\n  {interview_id}:")
    for original, anon in mapping.items():
        print(f"    {original} → {anon}")

# STEP 4: Study summary
print("\n" + "=" * 60)
print("STEP 4: Study summary")
print("=" * 60)

print(f"\n{study}")

print("\nInterview details:")
for i in study:
    print(f"\n  Interview: {i.get_participant_id()}")
    print(f"    ID: {i.id}")
    print(f"    Rows: {len(i.transcript)}")
    print(f"    Speakers: {i.get_speakers()}")
    print(f"    Speaker Mapping: {i.speaker_mapping}")


