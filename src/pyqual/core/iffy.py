from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Dict


@dataclass
class IffyIndex:
    """
    Validation result that can represent both individual checks and aggregated results.

    Can be used standalone (single validation check) or composite (aggregates multiple checks).
    When used as a composite, the nested checks represent either:
    - Multiple validation methods applied to a single result
    - Per-item validations for a list result

    Attributes
    ----------
    status : Literal["ok", "check", "iffy"]
        Validation status
    explanation : str
        Human-readable explanation of the validation result
    errors : List[str]
        Specific validation errors or issues found
    method : Optional[str]
        Name of validation method (set for individual checks only)
    metadata : Optional[Dict]
        Additional validation metadata (e.g., confidence scores, ratios)
    checks : List[IffyIndex]
        Nested validation checks (set for composite results only)
    informational : bool
        If True, this check is informational only and won't affect aggregated validation status
    """
    status: Literal["ok", "check", "iffy"]
    explanation: str
    errors: List[str] = field(default_factory=list)
    method: Optional[str] = None
    metadata: Optional[Dict] = None
    checks: List[IffyIndex] = field(default_factory=list)
    informational: bool = False

    @classmethod
    def from_check(
        cls,
        method: str,
        status: Literal["ok", "check", "iffy"],
        explanation: str,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        informational: bool = False
    ) -> IffyIndex:
        """
        Create an IffyIndex representing a single validation check.

        Parameters
        ----------
        method : str
            Name of the validation method (e.g., "quote_match", "llm_judge")
        status : Literal["ok", "check", "iffy"]
            Validation status
        explanation : str
            Human-readable explanation
        errors : Optional[List[str]]
            Specific validation errors
        metadata : Optional[Dict]
            Additional metadata (e.g., {"confidence": 0.85})
        informational : bool, default=False
            If True, this check is informational only and won't affect aggregated status

        Returns
        -------
        IffyIndex
            Single validation check result
        """
        return cls(
            status=status,
            explanation=explanation,
            errors=errors or [],
            method=method,
            metadata=metadata,
            checks=[],
            informational=informational
        )

    @classmethod
    def from_checks(
        cls,
        checks: List[IffyIndex],
        aggregation: str = "strictest"
    ) -> IffyIndex:
        """
        Aggregate multiple IffyIndex results into a composite result.

        Informational checks (informational=True) are included in the checks list
        but do not affect the aggregated validation status.

        Parameters
        ----------
        checks : List[IffyIndex]
            Individual validation check results to aggregate
        aggregation : str, default="strictest"
            Strategy for aggregation:
            - "strictest": worst status wins (iffy > check > ok)
            - "consensus": all must agree on ok

        Returns
        -------
        IffyIndex
            Composite validation result with nested checks

        Raises
        ------
        ValueError
            If checks list is empty or aggregation strategy is unknown
        """
        if not checks:
            raise ValueError("Cannot aggregate empty checks list")

        # Separate validation checks from informational checks
        validation_checks = [c for c in checks if not c.informational]

        # If all checks are informational, treat them all as validation checks
        if not validation_checks:
            validation_checks = checks

        if aggregation == "strictest":
            # Priority: iffy > check > ok (only for non-informational checks)
            if any(c.status == "iffy" for c in validation_checks):
                status = "iffy"
                failing = [c.method or "unknown" for c in validation_checks if c.status == "iffy"]
                explanation = f"Failed validation: {', '.join(failing)}"
            elif any(c.status == "check" for c in validation_checks):
                status = "check"
                flagged = [c.method or "unknown" for c in validation_checks if c.status == "check"]
                explanation = f"Needs review: {', '.join(flagged)}"
            else:
                status = "ok"
                explanation = f"All {len(validation_checks)} validation checks passed"

        elif aggregation == "consensus":
            if all(c.status == "ok" for c in validation_checks):
                status = "ok"
                explanation = f"Consensus: all {len(validation_checks)} checks passed"
            else:
                status = "check"
                explanation = "No consensus on validation"

        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")

        return cls(
            status=status,
            explanation=explanation,
            checks=checks,  # Include all checks (validation + informational)
            method=None,
            metadata=None
        )

    def is_composite(self) -> bool:
        """Check if this is an aggregated result (has nested checks)."""
        return len(self.checks) > 0

    def get_check(self, method: str) -> Optional[IffyIndex]:
        """
        Get a specific validation check by method name.

        Parameters
        ----------
        method : str
            Name of the validation method to retrieve

        Returns
        -------
        Optional[IffyIndex]
            The matching check, or None if not found
        """
        for check in self.checks:
            if check.method == method:
                return check
        return None

    @property
    def all_errors(self) -> List[str]:
        """Get all errors including from nested checks."""
        errors = list(self.errors)
        for check in self.checks:
            errors.extend(check.all_errors)
        return errors

    def icon(self) -> str:
        """Get emoji icon representing the validation status."""
        return {
            "ok": "✅",
            "check": "⚠️",
            "iffy": "❌"
        }[self.status]

    def to_dict(self) -> dict:
        """
        Convert to dictionary representation.

        Returns
        -------
        dict
            Dictionary containing status, explanation, icon, and nested data
        """
        d = {
            "status": self.status,
            "explanation": self.explanation,
            "icon": self.icon(),
            "errors": self.errors
        }

        if self.method:
            d["method"] = self.method
        if self.metadata:
            d["metadata"] = self.metadata
        if self.checks:
            d["checks"] = [c.to_dict() for c in self.checks]

        return d

    def __str__(self) -> str:
        if self.method:
            # Individual check format
            base = f"{self.icon()} [{self.method}] {self.status} — {self.explanation}"
        else:
            # Composite format
            base = f"{self.icon()} {self.status} — {self.explanation}"

        if self.errors:
            error_detail = "\n    • " + "\n    • ".join(self.errors)
            base += error_detail

        if self.checks:
            check_detail = "\n  " + "\n  ".join(str(c) for c in self.checks)
            base += check_detail

        return base

    def __repr__(self) -> str:
        return self.__str__()