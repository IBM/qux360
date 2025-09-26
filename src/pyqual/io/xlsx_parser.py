import pandas as pd
from .utils import ensure_schema

print(">>> loading xlsx_parser.py")

def parse_xlsx(path: str) -> pd.DataFrame:
    """
    Parse an XLSX transcript into the PyQual schema.
    Must contain at least timestamp, speaker, text.
    """
    df = pd.read_excel(path)
    return ensure_schema(df, "xlsx")

print(">>> defining parse_xlsx")