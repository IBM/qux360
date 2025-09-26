import pandas as pd
from .utils import ensure_schema



def parse_xlsx(path: str) -> pd.DataFrame:
    """
    Parse an XLSX transcript into the PyQual schema.
    Must contain at least timestamp, speaker, statement.
    """
    df = pd.read_excel(path)
    return ensure_schema(df, "xlsx")

