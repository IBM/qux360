from pathlib import Path
from qux360.core import Interview
from mellea import MelleaSession
from mellea.backends.litellm import LiteLLMBackend
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Enable INFO logging only for qux360
logging.getLogger("qux360").setLevel(logging.INFO)

load_dotenv()

ROOT_DIR = Path.cwd()
data_dir = ROOT_DIR.joinpath("examples/data")
file = data_dir.joinpath("interview_A.xlsx")
export_file = data_dir.joinpath("interview_A_exported.xlsx")
config_file = ROOT_DIR.joinpath("examples/config.json")

m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID"))) # type: ignore

# Suppress Mellea's FancyLogger (MelleaSession resets it to DEBUG, so we set it here)
logging.getLogger('fancy_logger').setLevel(logging.WARNING)

participant_id = "P1"

# STEP 1: Load interview
print("=" * 60)
print("STEP 1: Loading interview")
print("=" * 60)

# [OPTION A] create an instance without headers config (has headers by default)
i = Interview(file)

# [OPTION B] create an instance with headers config
'''
# loads config file to get the headers names provided by the user
with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)
# create an instance with headers config
i = Interview(file, headers=config['headers'])
'''

print(f"\nLoaded interview: {i}")
print("\nTranscript preview:")
i.show(10)

# look at the speakers
speakers = i.get_speakers()
print(f"\nSpeakers found: {speakers}")

# STEP 2: Anonymize speakers
print("\n" + "=" * 60)
print("STEP 2: Anonymizing speakers")
print("=" * 60)

map = i.anonymize_speakers_generic()
print("\nSpeaker mapping:")
for original, anon in map.items():
    print(f"  {original} → {anon}")

print("\nTranscript after anonymization:")
i.show(10)

# STEP 3: Detect and anonymize entities
print("\n" + "=" * 60)
print("STEP 3: Detecting entities (PERSON, ORG, GPE)")
print("=" * 60)

entities = i.detect_entities()
print(f"\nDetected {len(entities)} unique entities:")
for e in entities:
    print(f"  • '{e['entity']}' ({e['label']}) in rows {e['rows']}")

print("\nBuilding replacement map:")
replacements = i.build_replacement_map(entities)
print(f"Generated {len(replacements)} replacements:")
for orig, repl in replacements.items():
    print(f"  '{orig}' → '{repl}'")

choice = input("\nUse default replacements? [Y/n]: ").strip().lower()
if choice == 'n':
    print("\n--- Manual replacement selection ---")
    counters = {"PERSON": 0, "ORG": 0, "GPE": 0}
    for e in entities:
        entity, label, rows = e["entity"], e["label"], e["rows"]

        print(f"\nEntity: '{entity}' ({label}) found in rows {rows}")
        choice = input("Replace? [y/N]: ").strip().lower()
        if choice == "y":
            counters[label] += 1
            default_replacement = f"[{label}_{counters[label]}]"
            repl = input(f"Replacement (default: {default_replacement}): ").strip()
            # Prevent 'y' or empty from being stored
            if not repl or repl.lower() == "y":
                repl = default_replacement

            replacements[entity] = repl
            print(f"✅ Will replace '{entity}' with '{repl}'")
        else:
            print(f"⏩ Skipping '{entity}'")

    print(f"\nFinal replacements ({len(replacements)} items):")
    for orig, repl in replacements.items():
        print(f"  '{orig}' → '{repl}'")

print("\nApplying entity replacements to transcript...")
i.anonymize_statements(replacements)
print(f"✅ Anonymized {len(replacements)} entities")

print("\nTranscript after entity anonymization:")
i.show(10)

# STEP 4: Identify interviewee
print("\n" + "=" * 60)
print("STEP 4: Identifying interviewee using AI")
print("=" * 60)

result = i.identify_interviewee(m)
identification = result.result

print("\n*** AI Result ***")
print(f"Identified interviewee: {identification.interviewee}")
print(f"Confidence: {identification.confidence}")
print(f"Explanation: {identification.explanation}")
print(f"\n*** QIndex ***: \n{result.validation}")


# STEP 5: Rename interviewee to participant ID
print("\n" + "=" * 60)
print("STEP 5: Renaming interviewee to participant ID")
print("=" * 60)

interviewee = identification.interviewee
map = i.rename_speaker(interviewee, participant_id)
print(f"\nRenamed: {interviewee} → {participant_id}")

print("\nFinal transcript:")
i.show(10)

print(f"\nInterview summary: {i}")


# STEP 6: Export interview
print("\n" + "=" * 60)
print("STEP 6: Exporting interview")
print("=" * 60)

print(f"\nExporting to: {export_file}")
i.to_xlsx(export_file, include_enriched=False)
print(f"✅ Successfully exported interview")


