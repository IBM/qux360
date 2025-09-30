from pathlib import Path
from pyqual.core.interview import Interview
from mellea import MelleaSession
from mellea.backends.watsonx import WatsonxAIBackend
from mellea import MelleaSession
from dotenv import load_dotenv
import os

load_dotenv()

m = MelleaSession(backend=WatsonxAIBackend(model_id=os.getenv("MODEL_ID")))
data_dir = Path(__file__).parent / "data"
file = data_dir / "P5.docx"
export_file = data_dir / "P5_exported.xlsx"

# create an instance
i = Interview(file)

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
choice = input("Use as replacements [y/N]: ").strip().lower()
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
interviewee = i.identify_interviewee(m)
print (f"Interviewee: {interviewee}")

# rename the interviewee to pa participant name, e.g. P5
print("\nRenaming interviewee:")
map = i.rename_speaker(interviewee, "P5")
print(map)

# let's see how it looks like
i.show(10)

# show the object
print("\nInterview:")
print(i)


# export as an xlsx 
print("\nExporting as: " + str(export_file))
i.to_xlsx(export_file, include_enriched=False)


