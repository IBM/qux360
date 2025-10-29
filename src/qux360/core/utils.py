"""
Core utility functions for Qux360.
"""

import logging
from typing import List


logger = logging.getLogger(__name__)


def print_mellea_validations(response, title: str = "Validations") -> None:
    """
    Print Mellea validation results to stdout.

    This helper prints validation results from a Mellea response in a
    standardized format, showing requirements, results, scores, and reasons.

    Parameters
    ----------
    response : MelleaResponse
        The response object from m.instruct()
    title : str, default="Validations"
        Title to display at the top of the validation output

    Examples
    --------
    >>> response = m.instruct(prompt, requirements=reqs, format=MyModel)
    >>> print_mellea_validations(response, title="Topic Extraction Validations")
    """
    if not logger.isEnabledFor(logging.INFO):
        return

    if not hasattr(response, 'sample_validations') or not response.sample_validations:
        logger.debug("No sample_validations found in Mellea response")
        return

    print(f"**** {title}")
    for i, validation_group in enumerate(response.sample_validations, start=1):
        print(f"\n--- Validation Group {i} ---")
        for req, res in validation_group:
            print(f"Requirement: {req.description or '(no description)'}")
            print(f".  Result: {res._result}")
            if res.score is not None:
                print(f"  🔢 Score: {res.score:.2f}")
            if res.reason:
                # Truncate reason to 80 characters total (including prefix)
                reason_text = res.reason[:69] + "..." if len(res.reason) > 69 else res.reason
                print(f".  Reason: {reason_text}")
            if req.check_only:
                print(f"  ⚙️  (Check-only requirement)")
            print("-" * 40)


def validate_transcript_columns(transcript, required_columns: List[str]) -> None:
    """
    Validate that required columns exist in transcript.

    Parameters
    ----------
    transcript : DataFrame
        The transcript to validate
    required_columns : List[str]
        List of required column names

    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing = [col for col in required_columns if col not in transcript.columns]
    if missing:
        missing_str = "', '".join(missing)
        raise ValueError(
            f"Transcript is missing required column(s): '{missing_str}'. "
            f"If your transcript names them differently, you can set up the right headers configuration."
        )


def format_quotes_for_display(quotes: List, max_length: int = 100) -> str:
    """
    Format a list of quotes for display in prompts or logs.

    Parameters
    ----------
    quotes : List[Quote]
        List of Quote objects to format
    max_length : int, optional
        Maximum length of quote text (default: 100)

    Returns
    -------
    str
        Formatted quotes as newline-separated string
    """
    return "\n".join(
        f"- [{q.index}] {q.quote[:max_length]}..."
        for q in quotes
    )


def parse_quality_rating(rating: str) -> tuple[str, str]:
    """
    Parse LLM quality rating into status and explanation.

    Maps quality ratings to IffyIndex statuses:
    - "excellent" -> "ok"
    - "acceptable" -> "check"
    - "poor" or other -> "iffy"

    Parameters
    ----------
    rating : str
        The LLM rating response (should start with rating keyword)

    Returns
    -------
    tuple[str, str]
        (status, explanation) where status is "ok", "check", or "iffy"
    """
    rating_lower = rating.strip().lower()

    if rating_lower.startswith("excellent"):
        return "ok", rating
    elif rating_lower.startswith("acceptable"):
        return "check", rating
    elif rating_lower.startswith("poor"):
        return "iffy", rating
    else:
        return "iffy", f"Unexpected LLM response: {rating}"


def parse_coherence_rating(rating: str) -> tuple[str, str]:
    """
    Parse LLM coherence rating into status and prefix.

    Maps coherence ratings to IffyIndex statuses:
    - "strong" -> "ok", "Strong coherence"
    - "acceptable" -> "check", "Acceptable coherence (review recommended)"
    - "weak" or other -> "iffy", "Weak coherence"

    Parameters
    ----------
    rating : str
        The coherence rating ("Strong", "Acceptable", or "Weak")

    Returns
    -------
    tuple[str, str]
        (status, prefix) where status is "ok", "check", or "iffy"
    """
    rating_lower = rating.strip().lower()

    if rating_lower == "strong":
        return "ok", "Strong coherence"
    elif rating_lower == "acceptable":
        return "check", "Acceptable coherence (review recommended)"
    elif rating_lower == "weak":
        return "iffy", "Weak coherence"
    else:
        return "check", "Coherence assessment unclear"
