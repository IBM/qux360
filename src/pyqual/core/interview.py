import pandas as pd
from pathlib import Path
from typing import Optional, Union
import uuid

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
        - Otherwise, starts as an empty Interview with the standard schema.
        """
        self.id = f"interview_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}
        self.transcript = self._init_transcript(file)

    def _init_transcript(self, file):
        if not file:
            return pd.DataFrame(columns=[
                "timestamp", "speaker_id", "speaker", "statement", "codes", "themes"
            ])

        ext = Path(file).suffix.lower()
        if ext == ".docx":
            return parse_docx(file)
        elif ext in (".xls", ".xlsx"):
            return parse_xlsx(file)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def add_code(self, row: int, code: str):
        """Attach a code to a specific row in the transcript."""
        current = self.transcript.at[row, "codes"]
        if not isinstance(current, list):
            self.transcript.at[row, "codes"] = []
        self.transcript.at[row, "codes"].append(code)

    def to_csv(self, path: str):
        self.transcript.to_csv(path, index=False)

    def to_xlsx(self, path: str):
        self.transcript.to_excel(path, index=False)

    def __repr__(self):
        return f"<Interview {self.id}, {len(self.transcript)} rows>"
    
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
        """Return a list of unique speakers in the transcript."""
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