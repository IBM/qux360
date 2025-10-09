import os
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from textwrap import shorten
import uuid
import spacy

from mellea import MelleaSession
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pyqual.core.iffy import IffyIndex
import copy

from .models import TopicList
from ..io.docx_parser import parse_docx
from ..io.xlsx_parser import parse_xlsx
from ..io.csv_parser import parse_csv

logger = logging.getLogger(__name__)

class Interview:

    def __init__(self,
                 file: Optional[Union[str, Path]] = None,
                 metadata: Optional[dict] = None):
        """
        Initialize an Interview.
        
        - Always generates a unique UUID-based id.
        - If a file is provided, it is parsed into a transcript DataFrame.
        - Keeps both raw (immutable) and working (mutable) transcripts.
        """
        self.id = f"interview_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}
        raw = self._init_transcript(file) if file else self._empty_transcript()
        self.transcript_raw = raw
        self.transcript = copy.deepcopy(raw)
        self.speaker_mapping = None

    def __repr__(self):
        return f"<Interview {self.id}, {len(self.transcript)} turns, metadata={self.metadata}>"
    
    def _empty_transcript(self):
        return pd.DataFrame(columns=[
            "timestamp", "speaker_id", "speaker", "statement", "codes", "themes"
        ])

    def _init_transcript(self, file):
        if not file:
            return self._empty_transcript()
        
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

        ext = Path(file).suffix.lower()
        if ext == ".docx":
            return parse_docx(file)
        elif ext in (".xls", ".xlsx"):
            return parse_xlsx(file)
        elif ext == ".csv":
            return parse_csv(file)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


    def _load_spacy_model(self, model: str = "en_core_web_trf"):
        """Try loading a spaCy model, with fallback to lg → sm."""
        candidates = [model, "en_core_web_lg", "en_core_web_sm"]
        for candidate in candidates:
            try:
                nlp = spacy.load(candidate)
                print(f"Using spaCy model: {candidate}")
                return nlp
            except OSError:
                print(f"spaCy model '{candidate}' not found.")
        raise RuntimeError("No suitable spaCy model installed.")
    

    
    def reset_transcript(self):
        """Reset the working transcript back to the raw version."""
        self.transcript = copy.deepcopy(self.transcript_raw)

    def load_file(self, file: str | Path):
        """
        Load a transcript file into the interview (overwrites both raw and working).
        """
        raw = self._init_transcript(file)
        self.transcript_raw = raw
        self.transcript = copy.deepcopy(raw)

    def add_code(self, row: int, code: str):
        """Attach a code to a specific row in the working transcript."""
        current = self.transcript.at[row, "codes"]
        if not isinstance(current, list):
            self.transcript.at[row, "codes"] = []
        self.transcript.at[row, "codes"].append(code)

    def to_xlsx(self, path: str, include_enriched: bool = True):
        """
        Export transcript to a well-formatted Excel file using XlsxWriter.

        Parameters
        ----------
        path : str
            Path to save the Excel file.
        include_enriched : bool, default=True
            If False, excludes speaker_id, codes, themes.
        """
        df = copy.deepcopy(self.transcript)

        # Drop enrichment columns if not requested
        if not include_enriched:
            drop_cols = ["speaker_id", "codes", "themes"]
            df = df[[c for c in df.columns if c not in drop_cols]]

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Transcript", index=False)
            ws = writer.sheets["Transcript"]

            # Define column formats
            workbook = writer.book
            wrap_top = workbook.add_format({"text_wrap": True, "valign": "top"})

            # Column width spec
            col_widths = {
                "timestamp": 10,
                "speaker_id": 10,
                "speaker": 25,
                "statement": 60,
                "codes": 20,
                "themes": 20,
            }

            # Apply formatting
            for idx, col in enumerate(df.columns):
                width = col_widths.get(col, 20)
                ws.set_column(idx, idx, width, wrap_top)

            # Freeze header row
            ws.freeze_panes(1, 0)

    def show(self, n: int = 5, speaker: Optional[str] = None, width: int = 60):
        """Pretty-print transcript preview in a clean 3-column layout."""
        df = self.transcript
        if df.empty:
            print("Transcript is empty.")
            return

        if speaker:
            df = df[df["speaker"] == speaker]

        for _, row in df.head(n).iterrows():
            ts = str(row["timestamp"])
            sp = str(row["speaker"])
            st = str(row["statement"])[:width]
            print(f"{ts:>8} | {sp:<20} | {st}")

    def get_speakers(self) -> list[str]:
        """Return a list of unique speakers in the working transcript."""
        if "speaker" not in self.transcript.columns:
            return []
        speakers = (
            self.transcript["speaker"]
            .dropna()
            .map(str.strip)
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        return speakers

 
    def identify_interviewee(self, m: Optional[MelleaSession] = None) -> tuple[str | None, IffyIndex]:
        """
        Identify the likely interviewee via heuristic and/or LLM.
        Returns a tuple: (predicted interviewee, IffyIndex).
        """

        def compute_iffy_index(predicted1: str, predicted2: str | None, ratio: float) -> IffyIndex:
            """Sub-function to compute reliability index for interviewee prediction."""

            if predicted2 is None:  # heuristic only
                if ratio >= 0.70:
                    status = "ok"
                elif ratio >= 0.60:
                    status = "check"
                else:
                    status = "iffy"
                return IffyIndex(
                    status=status,
                    explanation=f"Heuristic only: {predicted1} has {ratio:.0%} of words."
                )

            if predicted1 == predicted2:  # agreement
                status = "ok" if ratio >= 0.60 else "check"
                return IffyIndex(
                    status=status,
                    explanation=f"Heuristic and LLM agreed on {predicted1} ({ratio:.0%} of words)."
                )

            # disagreement
            if ratio >= 0.80:
                status = "ok"
            elif ratio >= 0.70:
                status = "check"
            else:
                status = "iffy"

            return IffyIndex(
                status=status,
                explanation=f"Heuristic chose {predicted1} ({ratio:.0%}), but LLM suggested {predicted2}."
            )

        # Nothing to work on
        if self.transcript.empty or "speaker" not in self.transcript:
            return None, IffyIndex(
                status="not_assessed",
                explanation="Transcript is empty or missing 'speaker' column."
            )

        # Heuristic: speaker with most words
        counts = self.transcript.groupby("speaker")["statement"].apply(
            lambda x: x.str.split().str.len().sum()
        )
        predicted1 = counts.idxmax()
        total_words = counts.sum()
        ratio = counts[predicted1] / total_words if total_words > 0 else 0.0
        predicted2 = None

        # Optional LLM prediction
        if m:
            snippet = "\n".join(
                f"[{row['timestamp']}] {row['speaker']}: {shorten(str(row['statement']), width=120)}"
                for _, row in self.transcript.head(25).iterrows()
            )
            predicted2 = str(m.instruct(
                """
                You are given an interview transcript snippet with multiple speakers.
                One or more speakers are the interviewers (asking questions).
                One speaker is the interviewee (giving longer answers).
                Based on the transcript snippet, identify the interviewee by ID (from the speaker column).

                Transcript Snippet:
                {{snippet}}

                Question: Who is the interviewee?
                """,
                requirements=[
                    "The answer should ONLY contain the speaker ID exactly as shown in the speaker column",
                    "There should be no explanation"
                ],
                strategy=RejectionSamplingStrategy(loop_budget=2),
                user_variables={"snippet": snippet}
            )).strip()

        # --- Reliability scoring ---
        iffiness = compute_iffy_index(predicted1, predicted2, ratio)

        # --- Metadata update only if trust is sufficient ---
        if iffiness.status in ("ok", "check"):
            self.metadata["participant_id"] = predicted1

        return predicted1, iffiness

    def detect_entities(self, model: str = "en_core_web_trf", verbose: bool = False) -> list[dict]:
        """
        Detect PERSON, ORG, and GPE entities in transcript statements.

        Parameters
        ----------
        model : str, default="en_core_web_trf"
            Preferred spaCy model (falls back to lg → sm).
        verbose : bool, default=False
            If False (default), group rows per entity.
            If True, return every occurrence separately.

        Returns
        -------
        list of dict
            Compact mode: {"entity": "IBM", "label": "ORG", "rows": [3, 7]}
            Verbose mode: {"entity": "IBM", "label": "ORG", "row": 3, "statement": "..."}
        """
        nlp = self._load_spacy_model(model)

        results = []
        if verbose:
            for idx, text in self.transcript["statement"].dropna().items():
                doc = nlp(str(text))
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE"]:
                        results.append({
                            "entity": ent.text,
                            "label": ent.label_,
                            "row": idx,
                            "statement": text
                        })
            return results
        else:
            entities = {}
            for idx, text in self.transcript["statement"].dropna().items():
                doc = nlp(str(text))
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE"]:
                        key = (ent.text, ent.label_)
                        if key not in entities:
                            entities[key] = []
                        entities[key].append(idx)

            return [
                {"entity": ent, "label": label, "rows": rows}
                for (ent, label), rows in entities.items()
            ]
    

    def build_replacement_map(self, entities: list[dict]) -> dict:
        
        replacements = {}
        counters = {"PERSON": 0, "ORG": 0, "GPE": 0}

        for e in entities:
            entity, label = e["entity"], e["label"]

            # Assign a new placeholder only if this entity not seen before
            if entity not in replacements:
                counters[label] += 1
                replacements[entity] = f"[{label}_{counters[label]}]"

        return replacements
    
    def anonymize_speakers_generic(self) -> dict:
        """
        First-pass anonymization:
        Replace speaker names with neutral labels (Speaker 1, Speaker 2, ...).

        Returns
        -------
        dict
            Mapping of original names -> generic anonymized labels.
        """
        speakers = self.get_speakers()
        if not speakers:
            return {}

        mapping = {sp: f"Speaker{i+1}" for i, sp in enumerate(speakers)}

        self.transcript["speaker"] = self.transcript["speaker"].map(mapping)
        self.speaker_mapping = mapping

        # Update participant_id in metadata if already set
        pid = self.metadata.get("participant_id")
        if pid and pid in mapping:
            self.metadata["participant_id"] = mapping[pid]

        return mapping

    def anonymize_statements(self, replacements: dict) -> None:

        if not replacements:
            print("No replacements provided.")
            return

        def replace_text(text: str) -> str:
            anonymized = str(text)
            for original, anon in replacements.items():
                anonymized = anonymized.replace(original, anon)
            return anonymized

        # Replace inside statements
        self.transcript["statement"] = self.transcript["statement"].map(replace_text)


    def rename_speaker(self, old: str, new: str) -> dict:
        """
        Rename a speaker label in the transcript and update mapping.

        Parameters
        ----------
        old : str
            The current speaker label to rename (must exist in transcript).
        new : str
            The new label to assign.
        """
        if old not in self.transcript["speaker"].unique():
            raise ValueError(f"Speaker '{old}' not found in transcript.")

        # Apply renaming in transcript
        self.transcript["speaker"] = self.transcript["speaker"].replace({old: new})

        # Update mapping cumulatively
        if not hasattr(self, "speaker_mapping"):
            self.speaker_mapping = {sp: sp for sp in self.get_speakers()}

        composed = {}
        for original, current in self.speaker_mapping.items():
            if current == old:
                composed[original] = new
            else:
                composed[original] = current
        self.speaker_mapping = composed

         # Update participant_id in metadata if necessary
        pid = self.metadata.get("participant_id")
        if pid == old:
            self.metadata["participant_id"] = new

        return self.speaker_mapping

    def set_participant_id(self, pid: str) -> None:
        """
        Set or update the participant_id in metadata.

        - If `pid` matches a speaker in the transcript, it is stored directly.
        - If a speaker_mapping exists and `pid` matches an *original* name,
        it is translated to the current anonymized/renamed label.
        """
        if "speaker" not in self.transcript.columns:
            raise ValueError("Transcript has no 'speaker' column.")

        # Case 1: pid matches a current label in transcript
        if pid in self.transcript["speaker"].unique():
            self.metadata["participant_id"] = pid
            return

        # Case 2: pid matches an original name in speaker_mapping
        if hasattr(self, "speaker_mapping") and pid in self.speaker_mapping:
            self.metadata["participant_id"] = self.speaker_mapping[pid]
            return

        raise ValueError(
            f"Participant ID '{pid}' not found in transcript or speaker mapping."
        )

    def get_participant_id(self) -> str | None:
        """Return participant_id if set in metadata."""
        return self.metadata.get("participant_id")
    

    def suggest_topics_top_down(self, m: MelleaSession, n: Optional[int] = None, explain: bool = True, interview_context = "General") -> TopicList | None:
        """
        Suggest overarching themes for the interview using an LLM.

        Parameters
        ----------
        m : MelleaSession
            Active Mellea session for prompting.
        n : int, optional
            Desired number of themes to suggest. If None, let the model decide.
        explain : bool, default=True
            If True, also request a short explanation for each theme.

        Returns
        -------
        list of dict
            Each dict contains {"theme": str, "explanation": str | None}
        """

        df = self.transcript
        if df.empty:
            print("Transcript is empty.")
            return None

        if "participant_id" not in self.metadata:
            print("Need participant id for thematic analysis")
            return None

        # Join transcript into one block
        text = "\n".join(
            f"[{row['timestamp']}] {row['speaker']}: {row['statement']}"
            for _, row in df.iterrows()
        )
        
        # --- Build instructions dynamically ---
        interviewee = self.metadata["participant_id"]
        num_req = f"exactly {n} unique, non-generic" if n else "as many as possible unique, non-generic"
        exp_req = "Each topic must include a detailed explanation, why it was chosen. More than 1 sentence." if explain else "Explanations must be omitted. Use 'None'."

        prompt = """
        You are given an interview transcript. Your task is to identify {{num_req}} topics in the statements made by the interviewee {{interviewee}}.

        Interview Transcript:
        {{text}} 
        """

        logger.info("Calling Mellea...")
        requirements=[ 
                f"Each topic should be specific to the context of the overall interview: {interview_context}",
                "Each topic should be 2–5 words, concise, concrete, but nuanced.",
                f"{exp_req}"]
        
        response = m.instruct(
            prompt, 
            strategy=RejectionSamplingStrategy(loop_budget=1), 
            user_variables={"interviewee": interviewee, "num_req": num_req, "exp_req": exp_req, "text": text, "interview_context": interview_context},
            model_options={
                "max_tokens": 5000,
                "temperature": 0.0
            },
            requirements=requirements,
            format=TopicList,
            return_sampling_results=True,
        )

        if logger.isEnabledFor(logging.DEBUG):
            print(("****************"))
            for i, validation_group in enumerate(response.sample_validations, start=1):
                print(f"\n--- Validation Group {i} ---")
                for req, res in validation_group:
                    print(f"Requirement: {req.description or '(no description)'}")
                    print(f".  Result: {res._result}")
                    if res.score is not None:
                        print(f"  🔢 Score: {res.score:.2f}")
                    if res.reason:
                        print(f".  Reason: {res.reason}")
                    if req.check_only:
                        print(f"  ⚙️  (Check-only requirement)")
                    print("-" * 40)

        try:
            topics = TopicList.model_validate_json(response._underlying_value)
            return topics
        except Exception as e:
            print("Could not parse LLM output:", e)
            print("Raw response was:", response)
            return None