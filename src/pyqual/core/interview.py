import pandas as pd
from pathlib import Path
from typing import Optional, Union
from textwrap import shorten
import uuid
import mellea
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

    def to_csv(self, path: str, raw: bool = False):
        df = self.transcript_raw if raw else self.transcript
        df.to_csv(path, index=False)

    def to_xlsx(self, path: str, raw: bool = False):
        df = self.transcript_raw if raw else self.transcript
        df.to_excel(path, index=False)

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
        """Identify the likely interviewee via heuristic and/or LLM."""
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
            One speaker is the interviewee (providing longer answers).
            Based on the transcript, identify the interviewee by name.

            Transcript Snippet:
            {snippet}

            Question: Who is the interviewee?
            Answer ONLY with the speaker's exact name from the transcript. Do not add an explanation.
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

    def anonymize_speakers(
    self,
    interviewee_name: str,
    interviewee_prefix: str = "P",
    interviewer_prefix: str = "Interviewer"
) -> dict:
        """
        Anonymize speaker names in the working transcript by role.

        Parameters
        ----------
        interviewee_name : str
            The exact name of the interviewee in the transcript.
        interviewee_prefix : str, default="P"
            Prefix for participants/interviewees (P1, P2, ...).
        interviewer_prefix : str, default="Interviewer"
            Prefix for interviewer(s) (Interviewer 1, 2, ...).

        Returns
        -------
        dict
            Mapping of original names -> anonymized names.
        """

        if not interviewee_name:
            print("No interviewee specified — skipping anonymization.")
            return {}

        speakers = self.get_speakers()
        if not speakers:
            return {}

        mapping = {}
        participant_count = 0
        interviewer_count = 0

        for sp in speakers:
            if sp == interviewee_name:
                participant_count += 1
                mapping[sp] = f"{interviewee_prefix}{participant_count}"
            else:
                interviewer_count += 1
                mapping[sp] = f"{interviewer_prefix} {interviewer_count}"

        # Apply mapping to working transcript
        self.transcript["speaker"] = self.transcript["speaker"].map(mapping)
        self.speaker_mapping = mapping

        # Print mapping for inspection


        return mapping

    

    def __repr__(self):
        return f"<Interview {self.id}, {len(self.transcript)} rows>"