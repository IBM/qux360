"""
Validator classes for AI-generated results.

This module provides a framework for validating AI-generated outputs
through composable validator classes. Each validator implements a specific
validation check and returns a QIndex result.

Example
-------
>>> from qux360.core.validators import HeuristicAgreementValidator
>>> validator = HeuristicAgreementValidator(ok_threshold=0.60)
>>> validation = validator.validate("Speaker1", transcript)
>>> print(validation.status)  # "ok", "check", or "iffy"
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import pandas as pd
import logging

from .qindex import QIndex
from .utils import extract_mellea_validation_status


logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """
    Abstract base class for all validators.

    Validators encapsulate a single validation check that can be applied
    to AI-generated results. Each validator returns a QIndex object
    indicating the validation status.

    Examples
    --------
    >>> class CustomValidator(BaseValidator):
    ...     def validate(self, data):
    ...         if self._check_passes(data):
    ...             return QIndex.from_check(
    ...                 method=self.method_name,
    ...                 status="ok",
    ...                 explanation="Validation passed"
    ...             )
    ...         else:
    ...             return QIndex.from_check(
    ...                 method=self.method_name,
    ...                 status="iffy",
    ...                 explanation="Validation failed"
    ...             )
    """

    @abstractmethod
    def validate(self, *args, **kwargs) -> QIndex:
        """
        Run validation and return QIndex result.

        Returns
        -------
        QIndex
            Validation result with status and explanation
        """
        pass

    @property
    def method_name(self) -> str:
        """
        Default method name derived from class name.

        Converts 'FooBarValidator' to 'foo_bar'.
        Override this property to customize the method name.

        Returns
        -------
        str
            Method name for QIndex
        """
        # Remove 'Validator' suffix and convert to snake_case
        name = self.__class__.__name__
        if name.endswith('Validator'):
            name = name[:-9]  # Remove 'Validator'

        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        return name


class MelleaRequirementsValidator(BaseValidator):
    """
    Validator that extracts and reports Mellea's internal validation results.

    This validator analyzes the sample_validations from a Mellea response
    and converts them to QIndex status. It's typically used as an informational
    check to track LLM requirement compliance.

    This is a generic validator that works with any Mellea response type.

    Parameters
    ----------
    adjust_status : bool, default=True
        If True, sets status to "check" if any requirement fails,
        "ok" if all pass (overriding Mellea's internal scoring)

    Examples
    --------
    >>> # For interviewee identification
    >>> validator = MelleaRequirementsValidator()
    >>> validation = validator.validate(
    ...     mellea_response,
    ...     result_summary=f"LLM identified: {identification.interviewee}"
    ... )
    >>> print(validation.status)  # "ok", "check", or "iffy"

    >>> # For topic extraction
    >>> validation = validator.validate(
    ...     mellea_response,
    ...     result_summary=f"LLM extracted {len(topics)} topics"
    ... )
    """

    def __init__(self, adjust_status: bool = True):
        self.adjust_status = adjust_status

    @property
    def method_name(self) -> str:
        return "mellea_requirements"

    def validate(self, response: Any, result_summary: Optional[str] = None) -> QIndex:
        """
        Validate Mellea response requirements.

        Parameters
        ----------
        response : MelleaResponse
            Response from m.instruct() with return_sampling_results=True
        result_summary : str, optional
            Optional summary to prepend to the explanation
            (e.g., "LLM identified: Speaker1" or "LLM extracted 5 topics")

        Returns
        -------
        QIndex
            Informational validation result based on Mellea requirements
        """
        logger.debug("Running Mellea requirements validation")

        # Extract Mellea's internal validation status
        status, explanation, req_details, metadata = extract_mellea_validation_status(response)

        # Optionally adjust status based on pass/fail counts
        if self.adjust_status:
            if metadata.get("failed_count", 0) > 0:
                status = "check"
            elif metadata.get("passed_count", 0) > 0:
                status = "ok"

        # Optionally prepend result summary
        if result_summary:
            explanation = f"{result_summary}. {explanation}"

        return QIndex.from_check(
            method=self.method_name,
            status=status, # type: ignore
            explanation=explanation,
            metadata=metadata,
            errors=req_details if status != "ok" else None,
            informational=True  # Don't affect overall validation
        )


class HeuristicAgreementValidator(BaseValidator):
    """
    Validator that compares a predicted speaker with word count heuristic.

    This validator calculates which speaker has the most words (typically
    the interviewee) and compares it with a predicted speaker.
    The validation status depends on agreement and word ratio thresholds.

    This is a generic validator that can validate any speaker prediction,
    not just from interviewee identification.

    Parameters
    ----------
    ok_threshold : float, default=0.60
        Minimum word ratio for "ok" status when prediction agrees with heuristic
    check_threshold : float, default=0.50
        Minimum word ratio for "check" status when prediction agrees with heuristic

    Examples
    --------
    >>> # For interviewee identification
    >>> validator = HeuristicAgreementValidator(ok_threshold=0.60)
    >>> validation = validator.validate("Speaker1", transcript)
    >>> if validation.status == "ok":
    ...     print("Prediction agrees with word count heuristic")
    """

    def __init__(self, ok_threshold: float = 0.60, check_threshold: float = 0.50):
        self.ok_threshold = ok_threshold
        self.check_threshold = check_threshold

    @property
    def method_name(self) -> str:
        return "heuristic_agreement"

    def validate(self, predicted_speaker: str, transcript: pd.DataFrame) -> QIndex:
        """
        Validate speaker prediction against word count heuristic.

        Parameters
        ----------
        predicted_speaker : str
            The predicted speaker ID to validate
        transcript : pd.DataFrame
            Interview transcript with 'speaker' and 'statement' columns

        Returns
        -------
        QIndex
            Validation result based on heuristic agreement
        """
        logger.debug(f"Running heuristic agreement validation (thresholds: ok={self.ok_threshold}, check={self.check_threshold})")

        # Calculate word count heuristic
        counts = transcript.groupby("speaker")["statement"].apply(
            lambda x: x.str.split().str.len().sum()
        )
        predicted_heuristic = counts.idxmax()
        total_words = counts.sum()
        heuristic_ratio = counts[predicted_heuristic] / total_words if total_words > 0 else 0.0

        # Get word ratio for predicted speaker
        predicted_word_ratio = counts.get(predicted_speaker, 0) / total_words if total_words > 0 else 0.0

        # Check agreement
        agrees = predicted_speaker == predicted_heuristic

        # Determine validation status
        if agrees and heuristic_ratio >= self.ok_threshold:
            # Prediction agrees with heuristic and speaker has >threshold% of words
            status = "ok"
            explanation = f"Heuristic agrees with prediction: {predicted_speaker} has {heuristic_ratio:.0%} of words"
        elif agrees and heuristic_ratio >= self.check_threshold:
            # Prediction agrees with heuristic but word ratio is moderate
            status = "check"
            explanation = f"Heuristic agrees with prediction but word ratio is moderate: {heuristic_ratio:.0%}"
        else:
            # Prediction disagrees with heuristic
            status = "iffy"
            explanation = (
                f"Prediction: {predicted_speaker} ({predicted_word_ratio:.0%} of words) "
                f"but heuristic suggests {predicted_heuristic} ({heuristic_ratio:.0%})"
            )

        metadata = {
            "predicted_speaker": predicted_speaker,
            "predicted_word_ratio": predicted_word_ratio,
            "heuristic_prediction": predicted_heuristic,
            "heuristic_word_ratio": heuristic_ratio,
            "agreement": agrees,
            "ok_threshold": self.ok_threshold,
            "check_threshold": self.check_threshold
        }

        return QIndex.from_check(
            method=self.method_name,
            status=status,
            explanation=explanation,
            metadata=metadata
        )
