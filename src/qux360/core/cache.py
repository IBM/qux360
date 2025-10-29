"""
Caching utilities for Interview and Study objects.

Provides transparent save/load functionality for complete interview state
(transcript, metadata, topics) to speed up development and testing.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .interview import Interview
    from .study import Study

from .validated import Validated, ValidatedList
from .models import TopicList, ThemeList
from .iffy import IffyIndex

logger = logging.getLogger(__name__)


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for cache invalidation."""
    if not file_path.exists():
        return ""

    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_cache_path(source_file: Path, cache_dir: Path = None) -> Path:
    """Generate cache file path for a given source file."""
    logger.debug(f"Get cache path - Source file: {source_file}")
    if cache_dir is None:
        cache_dir = source_file.parent / ".qux360_cache"

    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{source_file.stem}_state.json"
    return cache_file


def _is_cache_valid(cache_path: Path, source_file: Path) -> bool:
    """Check if cache exists and is up-to-date with source file."""
    logger.debug(f"Is cache valid - Cache path: {cache_path} | Source file: {source_file}")
    if not cache_path.exists():
        return False

    try:
        with open(cache_path, 'r') as f:
            state = json.load(f)

        # Check if source file hash matches
        cached_hash = state.get('source_file_hash', '')
        current_hash = _compute_file_hash(source_file)

        return cached_hash == current_hash
    except (json.JSONDecodeError, KeyError):
        return False


def save_interview_state(interview: 'Interview', cache_path: Optional[Path] = None) -> Path:
    """
    Save complete interview state to JSON file.

    Parameters
    ----------
    interview : Interview
        The interview instance to save
    cache_path : Path, optional
        Custom cache file path. If None, auto-generates in .qux360_cache/

    Returns
    -------
    Path
        Path to the saved cache file
    """
    from .interview import Interview  # Avoid circular import

    logger.debug(f"Save interview state - Interview ID: {interview.id}")

    if cache_path is None:
        if not hasattr(interview, 'file_path') or interview.file_path is None:
            raise ValueError("Interview has no file_path and no cache_path provided")
        cache_path = _get_cache_path(Path(interview.file_path))

    # Serialize topics_top_down (TopicList) and validation (IffyIndex)
    topics_data = None
    if interview.topics_top_down:
        topics_data = interview.topics_top_down.model_dump()

    # Serialize validation results
    validation_data = None
    if interview.topics_top_down_validation:
        validation_data = interview.topics_top_down_validation.to_dict()

    # Serialize state
    state = {
        'version': '1.0',  # For future compatibility
        'id': interview.id,
        'file_path': str(interview.file_path) if hasattr(interview, 'file_path') else None,
        'source_file_hash': _compute_file_hash(Path(interview.file_path)) if hasattr(interview, 'file_path') and interview.file_path else '',
        'transcript_raw': interview.transcript_raw.to_dict(orient='records'),
        'transcript': interview.transcript.to_dict(orient='records'),
        'speaker_mapping': interview.speaker_mapping,
        'metadata': interview.metadata,
        'topics_top_down': topics_data,
        'topics_top_down_validation': validation_data,
    }

    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(state, f, indent=2)

    logger.debug(f"Saved interview state to {cache_path}")
    return cache_path


def load_interview_state(cache_path: Path, validate_source: bool = True) -> 'Interview':
    """
    Load interview from saved state file.

    Parameters
    ----------
    cache_path : Path
        Path to the cache JSON file
    validate_source : bool, default=True
        If True, validates that source file hash matches (cache freshness check)

    Returns
    -------
    Interview
        Reconstructed interview instance with full state
    """
    from .interview import Interview  # Avoid circular import

    logger.debug(f"Load interview state - Cache path: {cache_path}")

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, 'r') as f:
        state = json.load(f)

    # Validate cache freshness if requested
    if validate_source and state.get('file_path'):
        source_file = Path(state['file_path'])
        if source_file.exists():
            current_hash = _compute_file_hash(source_file)
            if current_hash != state.get('source_file_hash', ''):
                logger.warning(f"Cache is stale for {source_file}. Source file has changed.")

    # Reconstruct Interview object without parsing
    interview = Interview.__new__(Interview)
    interview.id = state['id']
    interview.file_path = Path(state['file_path']) if state['file_path'] else None
    interview.transcript_raw = pd.DataFrame(state['transcript_raw'])
    interview.transcript = pd.DataFrame(state['transcript'])
    interview.speaker_mapping = state['speaker_mapping']
    interview.metadata = state['metadata']

    # Reconstruct topics if present
    if state.get('topics_top_down'):
        interview.topics_top_down = TopicList.model_validate(state['topics_top_down'])
    else:
        interview.topics_top_down = None

    # Reconstruct validation if present
    if state.get('topics_top_down_validation'):
        interview.topics_top_down_validation = IffyIndex.from_dict(state['topics_top_down_validation'])
    else:
        interview.topics_top_down_validation = None

    logger.debug(f"Loaded interview state from {cache_path}")
    if interview.topics_top_down_validation:
        logger.debug(f"  → Loaded validation: {interview.topics_top_down_validation.status}")
    return interview


def try_load_or_parse(file: Path, cache_dir: Optional[Path] = None, **parse_kwargs) -> 'Interview':
    """
    Smart loader: tries cache first, falls back to parsing source file.

    This is the recommended entry point for loading interviews with caching.

    Parameters
    ----------
    file : Path
        Path to source interview file (DOCX/XLSX/CSV)
    cache_dir : Path, optional
        Custom cache directory. Defaults to .qux360_cache/ next to source file
    **parse_kwargs
        Additional arguments passed to Interview constructor (headers, has_headers)

    Returns
    -------
    Interview
        Loaded interview (from cache or freshly parsed)
    """
    from .interview import Interview  # Avoid circular import

    logger.debug(f"Try load or parse\n{file}")

    file = Path(file)
    cache_path = _get_cache_path(file, cache_dir)

    # Try loading from cache
    if _is_cache_valid(cache_path, file):
        logger.debug(f"Loading from cache: {file.name}")
        return load_interview_state(cache_path)

    # Cache miss or stale - parse from source
    logger.debug(f"Cache miss or stale. Parsing from source: {file.name}")
    # IMPORTANT: Disable caching to avoid infinite recursion
    parse_kwargs['use_cache'] = False
    interview = Interview(file=file, **parse_kwargs)
    interview.file_path = file  # Ensure file_path is set

    return interview


def save_study_state(study: 'Study', cache_dir: Path) -> Path:
    """
    Save complete study state (all interviews + metadata).

    Parameters
    ----------
    study : Study
        The study instance to save
    cache_dir : Path
        Directory to save study state and all interview caches

    Returns
    -------
    Path
        Path to the study state file
    """
    from .study import Study  # Avoid circular import

    logger.debug(f"Save study state - Study: {study}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    # Save each interview
    interview_cache_paths = []
    for interview in study.documents:
        interview_cache_path = cache_dir / f"{interview.id}_state.json"
        save_interview_state(interview, interview_cache_path)
        interview_cache_paths.append(str(interview_cache_path))

    # Save study-level state
    study_state = {
        'version': '1.0',
        'study_context': study.study_context,
        'metadata': study.metadata,
        'interview_cache_paths': interview_cache_paths,
        # Note: themes_top_down is NOT cached - theme analysis should be re-run each time
    }

    study_cache_path = cache_dir / "study_state.json"
    with open(study_cache_path, 'w') as f:
        json.dump(study_state, f, indent=2)

    logger.debug(f"Saved study state to {cache_dir}")
    return study_cache_path


def load_study_state(cache_dir: Path) -> 'Study':
    """
    Load study from saved state directory.

    Parameters
    ----------
    cache_dir : Path
        Directory containing study_state.json and interview caches

    Returns
    -------
    Study
        Reconstructed study instance with all interviews
    """
    from .study import Study  # Avoid circular import

    logger.debug(f"Load study state - Cache path: {cache_dir}")

    cache_dir = Path(cache_dir)
    study_cache_path = cache_dir / "study_state.json"

    if not study_cache_path.exists():
        raise FileNotFoundError(f"Study cache not found: {study_cache_path}")

    with open(study_cache_path, 'r') as f:
        study_state = json.load(f)

    # Load all interviews
    interviews = []
    for interview_cache_path in study_state['interview_cache_paths']:
        interview = load_interview_state(Path(interview_cache_path), validate_source=False)
        interviews.append(interview)

    # Reconstruct Study
    study = Study.__new__(Study)
    study.documents = interviews
    study.study_context = study_state['study_context']
    study.metadata = study_state['metadata']

    # Theme caching removed - themes should be regenerated each run
    study.themes_top_down = None

    logger.debug(f"Loaded study state from {cache_dir}")
    return study
