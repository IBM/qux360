from pathlib import Path
from pyqual.core.interview import Interview
from mellea import MelleaSession
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

m = MelleaSession(backend=LiteLLMBackend(model_id=os.getenv("MODEL_ID")))

# Suppress Mellea's FancyLogger (MelleaSession resets it to DEBUG, so we set it here)
logging.getLogger('fancy_logger').setLevel(logging.WARNING)


data_dir = Path(__file__).parent / "data"
file = data_dir / "interview_A.xlsx"
export_file = data_dir / "interview_A_exported.xlsx"
participant_id = "P1"
config_file = Path(__file__).parent / "config.json"

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

# see what we loaded
i.show(10)

# look at the speakers
speakers = i.get_speakers()
print(f"\nSpeakers: {speakers}")

# anonmyze speakers
map = i.anonymize_speakers_generic()
print("\nSpeaker anonymization - Mapping:")
for original, anon in map.items():
    print(f"  {original} -> {anon}")

# let's see how it looks like
i.show(10)

# let's find stuff we should hide eventually
print("\nDetecting entities:")
entities = i.detect_entities()
print(entities)

# let's see how it looks if we replaced everything we found
print("\nDefault replacements:")
replacements = i.build_replacement_map(entities)
print(replacements)
choice = input("Use as replacements [Y/n]: ").strip().lower()
if choice == 'n':

    # let's go through it and ask the user what they want to do
    print("\nSelect replacements manually:")
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

    print(f"\nManual replacements: \n {replacements}")

# ok, let's do this and anonymize the statements too
print("\nAnonnymizing statements:")
i.anonymize_statements(replacements)

# let's see how it looks like
i.show(10)


# use heuristics and AI to find interviewee
print("\nIdentifying interviewee:")
result = i.identify_interviewee(m)
interviewee = result.result
print(f"Interviewee: {interviewee}")
print(f"Validation:", str(result.validation))

# rename the interviewee to pa participant name, e.g. P5
print("\nRenaming interviewee:")
map = i.rename_speaker(interviewee, participant_id)
print(map)

# let's see how it looks like
i.show(10)

# show the object
print("\nInterview:")
print(i)


# export as an xlsx 
print("\nExporting as: " + str(export_file))
i.to_xlsx(export_file, include_enriched=False)


