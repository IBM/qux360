import pandas as pd
from pathlib import Path
from typing import Optional, Union
from textwrap import shorten
import uuid
import spacy
from mellea import MelleaSession

from ..io.docx_parser import parse_docx
from ..io.xlsx_parser import parse_xlsx


class Interview:

    def __init__(self,
                 file: Optional[Union[str, Path]] = None,
                 metadata: Optional[dict] = None):
        """
        Initialize an Interview.

        - Always generates a unique UUID-based id (no overrides).
        - If a file is provided, it is parsed into a transcript DataFrame.
        - Keeps both raw (immutable) and working (mutable) transcripts.
        """
        self.id = f"interview_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}
        raw = self._init_transcript(file) if file else self._empty_transcript()
        self.transcript_raw = raw
        self.transcript = raw.copy()
        self.speaker_mapping = None

    def __repr__(self):
        return f"<Interview {self.id}, {len(self.transcript)} rows>"
    
    def _empty_transcript(self):
        return pd.DataFrame(columns=[
            "timestamp", "speaker_id", "speaker", "statement", "codes", "themes"
        ])

    def _init_transcript(self, file):
        if not file:
            return self._empty_transcript()

        ext = Path(file).suffix.lower()
        if ext == ".docx":
            return parse_docx(file)
        elif ext in (".xls", ".xlsx"):
            return parse_xlsx(file)
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
        self.transcript = self.transcript_raw.copy()

    def load_file(self, file: str | Path):
        """
        Load a transcript file into the interview (overwrites both raw and working).
        """
        raw = self._init_transcript(file)
        self.transcript_raw = raw
        self.transcript = raw.copy()

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
        df = self.transcript.copy()

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

    def identify_interviewee(self, m: Optional[MelleaSession] = None) -> str | dict | None:
        """
        Identify the likely interviewee via heuristic and/or LLM. 
        """
        if self.transcript.empty or "speaker" not in self.transcript:
            return None

        # Heuristic: speaker with most words
        counts = self.transcript.groupby("speaker")["statement"].apply(
            lambda x: x.str.split().str.len().sum()
        )
        predicted1 = counts.idxmax()
        predicted2 = None

        if m:
            df = self.transcript.head(50)
            lines = [
                f"[{row['timestamp']}] {row['speaker']}: "
                f"{shorten(str(row['statement']), width=120)}"
                for _, row in df.iterrows()
            ]
            snippet = "\n".join(lines)

            prompt = f"""
            You are given an interview transcript snippet with multiple speakers.
            One or more speakers are the interviewers (asking questions).
            One speaker is the interviewee (giving longer answers).
            Based on the transcript, identify the interviewee by ID (from the speaker column).

            Transcript Snippet:
            {snippet}

            Question: Who is the interviewee?
            Answer ONLY with the speaker ID exactly as shown in the snippet in the speaker column. Do not add an explanation.
            """

            predicted2 = m.chat(prompt).content.strip()

            if predicted2:
                if predicted1 == predicted2:
                    return predicted1  # agreement
                else:
                    return {
                        "heuristic": predicted1,
                        "llm": predicted2,
                        "status": "conflict"
                    }
            else:
                return predicted1

        return predicted1

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

        return self.speaker_mapping


