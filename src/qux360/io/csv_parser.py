import pandas as pd
from typing import Optional
from .utils import ensure_schema, process_headers

import logging
logger = logging.getLogger(__name__)

def parse_csv(path: str, headers: Optional[dict] = None, has_headers = True) -> pd.DataFrame:
    """
    Parse a CSV transcript into the Qux360 schema.
    Must contain at least timestamp, speaker, statement.
    """
    logger.debug(f"Parse CSV - Path: {path}")
    if has_headers:
        df = pd.read_csv(path) # reads the csv with header
        df = process_headers(df, headers) # process the headers according to the provided ones
    else:
        logger.warning(f"⚠️ File without headers. Adding default headers ['timestamp', 'speaker', 'statement']")
        df = pd.read_csv(path, header=None) # reads the csv without header
        df.columns = ["timestamp", "speaker", "statement"] # add headers

    return ensure_schema(df, "csv")