import pandas as pd
from .utils import ensure_schema

def parse_csv(path: str) -> pd.DataFrame:
    """
    Parse a CSV transcript into the PyQual schema.
    Must contain at least timestamp, speaker, statement.
    """
    df = pd.read_csv(path)
    return ensure_schema(df, "csv")