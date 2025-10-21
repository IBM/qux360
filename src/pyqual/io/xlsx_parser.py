import pandas as pd
from typing import Optional
from .utils import ensure_schema

def parse_xlsx(path: str, headers: Optional[dict] = None) -> pd.DataFrame:
    """
    Parse an XLSX transcript into the PyQual schema.
    Must contain at least timestamp, speaker, statement.
    """
    df = pd.read_excel(path)

    if (not headers):
        print(f"⚠️ Headers not provided in config.json file. Using default headers ['timestamp', 'speaker', 'statement']")
        headers = {  
            "timestamp": "timestamp",
            "speaker": "speaker",
            "statement": "statement"
        }
    try:
        df = df.rename(columns={
            headers['timestamp']: "timestamp",
            headers['speaker']: "speaker",
            headers['statement']: "statement"
        })
    except KeyError as e:
        raise ValueError(f"Wrong value for headers configuration. Expected values for 'timestamp', 'speaker', 'statement'. Found {e}")

    return ensure_schema(df, "xlsx")

