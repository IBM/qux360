"""
Core utility functions for PyQual.
"""

import logging


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
