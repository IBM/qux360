# PyQual  

PyQual is an experimental Python library for **AI-assisted qualitative interview analysis**, with support for transcripts in DOCX/XLSX formats, anonymization, entity detection, and AI-assisted coding.  

---

## 📦 Installation  

PyQual can be used in two ways:  
1. **As a user** (install from PyPI — coming soon).  
2. **As a developer/contributor** (working with the source repo).  

---

### 1. Users (PyPI, coming soon)

Once published, install with pip:  

```bash
pip install pyqual
```

Example:  

```python
from pyqual.core.interview import Interview

i = Interview("my_transcript.docx")
i.show()
```

---

### 2. Developers / Contributors  

#### Step 1. Clone the repo

```bash
git clone git@github.ibm.com:AIExperience/pyqual.git
cd pyqual
```

#### Step 2. Install dependencies with Poetry

🔧 Setting up Poetry

PyQual uses Poetry for dependency and environment management. If you don’t already have it installed, follow the instructions here:


👉 Poetry Installation Guide: https://python-poetry.org/docs/#installation 
<br>
<br>

```bash
poetry install
```

This creates a `.venv` and installs all dependencies.

#### Step 3. Activate the environment

Option A (recommended): activate `.venv` directly  
```bash
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate        # Windows
```

Option B (alternative): run with Poetry  
```bash
poetry run python examples/interview_basics.py
poetry run pytest -v
```

#### Step 4. Install a spaCy model (**required for NER/anonymization**)

```bash
python -m spacy download en_core_web_trf   # best quality
# or
python -m spacy download en_core_web_sm    # smaller/faster
```

#### Step 5. Verify installation

Run the included example:

```bash
python examples/interview_basics.py
```

---

## 🚀 Quickstart Example  

```python
from pyqual.core.interview import Interview

# Load a transcript (DOCX, XLSX, or CSV)
i = Interview("examples/data/sample_transcript.docx")

# Preview first few rows
i.show(n=5)

# List speakers
print("Speakers:", i.get_speakers())

# Export to Excel (formatted, with wrapped text + column widths)
i.to_xlsx("output_transcript.xlsx", include_enriched=False)
```

---

## 🤖 Quickstart: Anonymization & Interviewee Identification  

```python
from pyqual.core.interview import Interview

# Load a transcript
i = Interview("examples/data/sample_transcript.xlsx")

# Step 1: Rename speakers (e.g., Speaker 1 → Participant)
mapping = i.rename_speaker("Speaker 1", "Participant")
print("Speaker Mapping:", mapping)

# Step 2: Detect named entities (PERSON, ORG, GPE)
entities = i.detect_entities(model="en_core_web_sm")
print("Entities found:", entities)

# Step 3: Identify interviewee (heuristic + optional LLM check)
interviewee, _ = i.identify_interviewee()
print("Predicted interviewee:", interviewee)
```

---

✅ With these steps, you can load transcripts, anonymize participants, detect sensitive entities, and use LLMs via Mellea to assist in interview analysis.  


---

## 📜 License  

PyQual is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  

