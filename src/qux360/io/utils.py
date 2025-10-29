import pandas as pd

import logging
logger = logging.getLogger(__name__)

class SchemaValidationError(Exception):
    """Raised when the input DataFrame does not conform to the expected schema."""
    pass

def ensure_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Normalize DataFrame to Qux360 schema.
    - DOCX/CSV/XLSX: require timestamp, speaker, statement (case-insensitive).
    - VTT: allow missing speaker (set to 'Unknown').
    """

    logger.debug(f"Ensure schema - Source: {source}")

    # Normalize column names to lowercase and strip whitespace
    original_columns = df.columns.tolist()
    df.columns = [col.strip().lower() for col in df.columns]
    if original_columns != df.columns.tolist():
        logger.debug(f"⚠️ Column names were normalized. Original: {original_columns} → Normalized: {df.columns.tolist()}")

    if source == "vtt":
        if "timestamp" not in df.columns or "statement" not in df.columns:
            raise SchemaValidationError("VTT transcript must have timestamp and statement.")
        if "speaker" not in df.columns:
            df["speaker"] = "Unknown"
    else:
        required = ["timestamp", "speaker", "statement"]
        for col in required:
            if col not in df.columns:
                raise SchemaValidationError(f"{source.upper()} transcript missing required column: {col}")

    # Fill optional columns if missing
    if "speaker_id" not in df.columns:
        df["speaker_id"] = None
    if "codes" not in df.columns:
        df["codes"] = [[] for _ in range(len(df))]
    if "themes" not in df.columns:
        df["themes"] = [[] for _ in range(len(df))]

    # Reorder into canonical schema
    expected_columns = ["timestamp", "speaker_id", "speaker", "statement", "codes", "themes"]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing columns after normalization: {missing}")

    return df[expected_columns]


def process_headers(df: pd.DataFrame, headers: dict) -> dict:
    logger.debug(f"Process headers - Headers: {headers}")
    
    if headers is None:
        logger.debug("Custom headers not provided. Using default headers ['timestamp', 'speaker', 'statement']")
        headers = {
            "timestamp": "timestamp",
            "speaker": "speaker",
            "statement": "statement"
        }
    else:
        required_keys = {"timestamp", "speaker", "statement"}
        missing_keys = required_keys - headers.keys()
        if missing_keys:
            raise ValueError(f"Missing required header keys: {missing_keys}")
        
        try:
            df = df.rename(columns={
                headers["timestamp"]: "timestamp",
                headers["speaker"]: "speaker",
                headers["statement"]: "statement"
            })
        except KeyError as e:
            raise ValueError(f"Header mapping failed. Column not found: {e}")
    return df
