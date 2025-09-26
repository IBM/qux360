import pandas as pd
from docx import Document
from .utils import ensure_schema

def parse_docx(path: str) -> pd.DataFrame:
    """
    Parse a Teams-style DOCX transcript into the PyQual schema.
    Assumes table format with timestamp, speaker, statement.
    """
    doc = Document(path)
    rows = []
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 3:
                rows.append({
                    "timestamp": cells[0],
                    "speaker": cells[1],
                    "statement": cells[2],
                })
    df = pd.DataFrame(rows)
    return ensure_schema(df, "docx")