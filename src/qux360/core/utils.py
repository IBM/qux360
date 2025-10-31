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
                reason_text = res.reason[:300] + "..." if len(res.reason) > 300 else res.reason
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

    Maps quality ratings to QIndex statuses:
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

    Maps coherence ratings to QIndex statuses:
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


def extract_mellea_validation_status(response) -> tuple[str, str, list[str], dict]:
    """
    Extract validation status from Mellea sample_validations and convert to QIndex status.

    Analyzes Mellea's internal validation results to determine overall quality:
    - All requirements passed (score >= 0.5 or _result == True) -> "ok"
    - Some requirements passed, some check-only -> "check"
    - Any requirements failed -> "iffy"

    Parameters
    ----------
    response : MelleaResponse
        Response object from m.instruct() with return_sampling_results=True

    Returns
    -------
    tuple[str, str, list[str], dict]
        (status, explanation, requirement_details, metadata) where:
        - status: "ok", "check", or "iffy"
        - explanation: Human-readable summary of validation results
        - requirement_details: List of strings describing each requirement result
        - metadata: Dict with detailed validation info (passed_count, failed_count, etc.)
    """
    if not hasattr(response, 'sample_validations') or not response.sample_validations:
        return "check", "No Mellea validation data available", [], {}

    # Flatten all validation results across groups
    all_validations = []
    for validation_group in response.sample_validations:
        all_validations.extend(validation_group)

    if not all_validations:
        return "check", "No validation requirements found", [], {}

    # Analyze validation results and build detailed list
    passed_count = 0
    failed_count = 0
    check_only_count = 0
    total_score = 0.0
    scored_count = 0
    requirement_details = []

    for req, res in all_validations:
        req_desc = req.description or "(no description)"

        if req.check_only:
            check_only_count += 1
            requirement_details.append(f"[CHECK-ONLY] {req_desc}")
            continue

        # Consider a validation passed if result is True or score >= 0.5
        is_passed = res._result is True or (res.score is not None and res.score >= 0.5)

        # Build detail string
        status_icon = "✓" if is_passed else "✗"
        detail = f"[{status_icon}] {req_desc}"
        if res.score is not None:
            detail += f" (score: {res.score:.2f})"

        requirement_details.append(detail)

        if is_passed:
            passed_count += 1
        else:
            failed_count += 1

        if res.score is not None:
            total_score += res.score
            scored_count += 1

    # Calculate average score
    avg_score = total_score / scored_count if scored_count > 0 else None

    # Determine status
    total_requirements = passed_count + failed_count

    if failed_count == 0 and passed_count > 0:
        status = "ok"
        explanation = f"All {passed_count} Mellea requirements passed"
        if avg_score is not None:
            explanation += f" (avg score: {avg_score:.2f})"
    elif failed_count > 0 and passed_count > 0:
        status = "check"
        explanation = f"{passed_count}/{total_requirements} Mellea requirements passed"
        if avg_score is not None:
            explanation += f" (avg score: {avg_score:.2f})"
    elif failed_count > 0:
        status = "iffy"
        explanation = f"All {failed_count} Mellea requirements failed"
        if avg_score is not None:
            explanation += f" (avg score: {avg_score:.2f})"
    else:
        status = "check"
        explanation = "Only check-only requirements found"

    metadata = {
        "passed_count": passed_count,
        "failed_count": failed_count,
        "check_only_count": check_only_count,
        "total_requirements": total_requirements,
        "avg_score": avg_score
    }

    return status, explanation, requirement_details, metadata
