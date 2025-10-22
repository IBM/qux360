# Qux360  

**Qux360** is an experimental Python library for **AI-assisted qualitative analysis**.  

Validation is a **first-class concept** in Qux360 — every use of large language models is designed to be transparent, explainable, and open to scrutiny. The goal is to help developers build **trustworthy, interactive qualitative analysis experiences** while retaining flexibility in how they apply Qux360’s built-in quality assurance mechanisms.  

Qux360 is built on **[Mellea](https://mellea.ai/)**, a **generative computing library** that provides robust and validated prompting techniques. The current Qux360 version supports **interview data with a single participant (interviewee)** only, with plans to expand this scope in future releases.  

**Key capabilities:**  
- Import interview transcripts in **DOCX**, **XLSX**, or **CSV** formats  
- Export processed transcripts in **XLSX** or **CSV** formats  
- **Speaker anonymization**  
- **Statement anonymization** using **local, privacy-preserving entity detection**  
- **AI-assisted interviewee detection**  
- **AI-assisted top-down topic extraction**  
- **Bulk processing** across collections of interviews (e.g., anonymization)

---

## 📦 Installation  

Qux360 can be used in two ways:  
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

Qux360 uses Poetry for dependency and environment management. If you don’t already have it installed, follow the instructions here:


👉 Poetry Installation Guide: https://python-poetry.org/docs/#installation 
<br>
<br>

```bash
poetry install
```

This creates a `.venv` and installs all dependencies.

#### Step 3. Activate the environment

Activate `.venv` directly  
```bash
source .venv/bin/activate     # macOS/Linux
```

```bash
.venv\Scripts\activate        # Windows
```


#### Step 4. Install a spaCy model (**required for NER/anonymization**)

```bash
python -m spacy download en_core_web_trf   # best quality
```
or
```bash
python -m spacy download en_core_web_sm    # smaller/faster
```

#### Step 5. Set up `.env`

Qux360 uses Mellea as a layer to connect to inference services. You will need to create a `.env` file in your project root folder, using keys required by Mellea (depending on what models and services you use Mellea with). For example, the following keys in the `.env` file would allow you to use Mellea with WatsonX directly (through a WatsonXBackend), or through LiteLLM. LiteLLM is supported in Mellea and allows you to use a variety of backends.

```bash
MODEL_ID_LITELLM=watsonx/meta-llama/llama-3-3-70b-instruct
MODEL_ID=meta-llama/llama-3-3-70b-instruct
WATSONX_URL=[your URL]
WATSONX_API_KEY=[your API key]
WATSONX_PROJECT_ID=[your project ID]
DISABLE_AIOHTTP_TRANSPORT=True
```


#### Step 6. Verify installation

Run the included example:

```bash
python examples/interview_basics.py
```

---

## 🚀 Quickstart Example

```python
from pyqual.core.interview import Interview

# Load a transcript (DOCX, XLSX, or CSV)
i = Interview("examples/data/interview_A.docx")

# Preview first few rows
i.show(n=5)

# List speakers
print("Speakers:", i.get_speakers())

# Export to Excel (formatted, with wrapped text + column widths)
i.to_xlsx("output_transcript.xlsx", include_enriched=False)
```


## 🤖 Quickstart: Anonymization & Interviewee Identification

```python
from pyqual.core.interview import Interview

# Load a transcript
i = Interview("examples/data/interview_A.xlsx")

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


✅ With these steps, you can load transcripts, anonymize participants, detect sensitive entities, and use LLMs via Mellea to assist in interview analysis.  

--- 

## ✏️ Contributing

You can contribute to:
* Quax360. Look at the [Contribution Guidelines](CONTRIBUTING.md) for more details.
* [Mellea](https://mellea.ai/). Look at the [Contribution Guidelines](https://github.com/generative-computing/mellea/blob/main/docs/tutorial.md#appendix-contributing-to-mellea) for more details.

## 📚 Documentation

You can find extensive documentation of the system and the API usage in the [Documentation page](../../wiki).


## 📜 License  

Qux360 is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  


## IBM ❤️ Open Source AI