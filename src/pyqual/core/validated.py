from dataclasses import dataclass, field
from typing import TypeVar, Generic, List

from .iffy import IffyIndex


T = TypeVar('T')


@dataclass
class Validated(Generic[T]):
    """
    Generic container for AI-generated results with validation metadata.

    Use this to wrap any AI-generated output that has been validated
    against quality/reliability criteria.

    Attributes
    ----------
    result : T
        The actual AI-generated result
    validation : IffyIndex
        Validation assessment with status and explanation

    Examples
    --------
    >>> # Simple validated result
    >>> result = Validated(
    ...     result="Speaker1",
    ...     validation=IffyIndex(status="ok", explanation="High confidence")
    ... )
    >>> if result.passed_validation():
    ...     print(f"Interviewee: {result.result}")
    """
    result: T
    validation: IffyIndex

    def passed_validation(self) -> bool:
        """
        Check if result has acceptable validation status.

        Returns True for "ok" or "check" status, indicating the result
        can be used (though "check" suggests human review is recommended).

        Returns
        -------
        bool
            True if status is "ok" or "check", False if "iffy"
        """
        return self.validation.status in ("ok", "check")

    def needs_review(self) -> bool:
        """
        Check if result requires human review.

        Returns True for "check" or "iffy" status, indicating the result
        should be reviewed by a human before use.

        Returns
        -------
        bool
            True if status is "check" or "iffy", False if "ok"
        """
        return self.validation.status in ("check", "iffy")

    def has_issues(self) -> bool:
        """
        Check if validation found significant problems.

        Returns True only for "iffy" status, indicating the result
        has significant validation issues and may not be usable.

        Returns
        -------
        bool
            True if status is "iffy", False otherwise
        """
        return self.validation.status == "iffy"

    @property
    def value(self) -> T:
        """
        Alias for result (for convenience).

        Returns
        -------
        T
            The validated result
        """
        return self.result


@dataclass
class ValidatedList(Validated[List[T]]):
    """
    Container for AI-validated lists with per-item validation.

    Extends Validated to include validation for each item in the list,
    enabling granular quality assessment and filtering.

    Attributes
    ----------
    result : List[T]
        List of AI-generated results
    validation : IffyIndex
        Overall validation assessment for the entire list
    item_validations : List[IffyIndex]
        Per-item validation assessments (same length as result)

    Examples
    --------
    >>> # Validated list with per-item assessments
    >>> result = ValidatedList(
    ...     result=[topic1, topic2, topic3],
    ...     validation=IffyIndex(status="check", explanation="1/3 needs review"),
    ...     item_validations=[
    ...         IffyIndex(status="ok", explanation="All quotes validated"),
    ...         IffyIndex(status="check", explanation="1 quote mismatch"),
    ...         IffyIndex(status="ok", explanation="All quotes validated")
    ...     ]
    ... )
    >>> good_topics = result.ok_items
    >>> for topic, iffy in result.items_with_validations():
    ...     print(f"{topic.name}: {iffy.status}")
    """
    item_validations: List[IffyIndex] = field(default_factory=list)

    def __post_init__(self):
        """
        Validate that item_validations length matches result length.

        Raises
        ------
        ValueError
            If item_validations is provided but length doesn't match result
        """
        if self.item_validations and len(self.item_validations) != len(self.result):
            raise ValueError(
                f"Length mismatch: {len(self.result)} items in result but "
                f"{len(self.item_validations)} item validations provided"
            )

    def filter_by_status(self, *statuses: str) -> List[T]:
        """
        Return items matching given validation statuses.

        Parameters
        ----------
        *statuses : str
            One or more status values: "ok", "check", "iffy"

        Returns
        -------
        List[T]
            Items whose validation status matches any of the given statuses

        Examples
        --------
        >>> # Get only items with "ok" status
        >>> good_items = result.filter_by_status("ok")
        >>>
        >>> # Get items needing review
        >>> review_items = result.filter_by_status("check", "iffy")
        """
        if not self.item_validations:
            # If no per-item validations, return all or none based on overall
            return self.result if self.validation.status in statuses else []

        return [
            item for item, iffy in zip(self.result, self.item_validations)
            if iffy.status in statuses
        ]

    @property
    def ok_items(self) -> List[T]:
        """
        Return items with "ok" validation status.

        Returns
        -------
        List[T]
            Items that passed validation without issues
        """
        return self.filter_by_status("ok")

    @property
    def flagged_items(self) -> List[T]:
        """
        Return items with "check" or "iffy" validation status.

        Returns
        -------
        List[T]
            Items that need review or have validation issues
        """
        return self.filter_by_status("check", "iffy")

    def items_with_validations(self) -> List[tuple[T, IffyIndex]]:
        """
        Return list of (item, validation) tuples for easy iteration.

        Returns
        -------
        List[tuple[T, IffyIndex]]
            Zipped pairs of items and their corresponding validations

        Raises
        ------
        ValueError
            If no item validations are available

        Examples
        --------
        >>> for item, validation in result.items_with_validations():
        ...     print(f"{item}: {validation.status}")
        ...     if validation.has_issues():
        ...         print(f"  Errors: {validation.errors}")
        """
        if not self.item_validations:
            raise ValueError("No item validations available")
        return list(zip(self.result, self.item_validations))
