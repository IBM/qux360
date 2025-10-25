import re
import pandas as pd
from pathlib import Path
from docx import Document

import logging
logger = logging.getLogger(__name__)

def parse_docx(path: str | Path) -> pd.DataFrame:
    """
    Parse a DOCX transcript into the PyQual schema.

    Parameters
    ----------
    path : str or Path
        Path to the .docx transcript file.

    Returns
    -------
    pd.DataFrame
        Transcript with columns: timestamp, speaker_id, speaker, statement, codes, themes.
    """
    
    doc = Document(path)
    
    segments = []
    current_speaker = None
    current_timestamp = None
    current_statement = []

    for para in doc.paragraphs:
        line = para.text.strip()
        if not line:
            continue  # Skip empty lines

        # Match: speaker name + timestamp
        match = re.match(
            r'([A-Za-z\s!@#$%^&*()_+=\-\{\}\[\]\|;\'\",.<>/?~]+)\s+((?:(\d{1,2}):)?\d{1,2}:\d{2})',
            line
        )

        if match:
            # Save previous segment
            if current_speaker and current_statement:
                cleaned_statement = " ".join(current_statement).replace("\n", " ")
                segments.append([current_timestamp, None, current_speaker, cleaned_statement, [], []])

            # Extract new speaker + timestamp
            current_speaker = match.group(1).strip()
            current_timestamp = match.group(2)
            current_statement = [line.split(current_timestamp, 1)[1].strip()]
        else:
            if current_statement is not None:
                current_statement.append(line)

    # Save last segment
    if current_speaker and current_statement:
        cleaned_statement = " ".join(current_statement).replace("\n", " ")
        segments.append([current_timestamp, None, current_speaker, cleaned_statement, [], []])

    df = pd.DataFrame(
        segments,
        columns=["timestamp", "speaker_id", "speaker", "statement", "codes", "themes"]
    )

    return df