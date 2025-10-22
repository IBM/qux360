import pandas as pd

def ensure_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Normalize DataFrame to PyQual schema.
    - DOCX/CSV/XLSX: require timestamp, speaker, statement (case-insensitive).
    - VTT: allow missing speaker (set to 'Unknown').
    """

    # Normalize column names to lowercase and strip whitespace
    df.columns = [col.strip().lower() for col in df.columns]

    if source == "vtt":
        if "timestamp" not in df.columns or "statement" not in df.columns:
            raise ValueError("VTT transcript must have timestamp and statement.")
        if "speaker" not in df.columns:
            df["speaker"] = "Unknown"
    else:
        required = ["timestamp", "speaker", "statement"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"{source.upper()} transcript missing required column: {col}")

    # Fill optional columns if missing
    if "speaker_id" not in df.columns:
        df["speaker_id"] = None
    if "codes" not in df.columns:
        df["codes"] = [[] for _ in range(len(df))]
    if "themes" not in df.columns:
        df["themes"] = [[] for _ in range(len(df))]

    # Reorder into canonical schema
    return df[["timestamp", "speaker_id", "speaker", "statement", "codes", "themes"]]


def process_headers(df, headers: dict) -> dict:
    if (not headers):
        print(f"⚠️ Headers not provided. Using default headers ['timestamp', 'speaker', 'statement']")
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

    return df