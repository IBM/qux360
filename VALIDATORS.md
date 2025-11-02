# Validators Documentation

This document describes all validators in Qux360, grouped by AI function. Each validator performs quality checks on LLM outputs to ensure reliability and transparency.

## Overview

Qux360 implements validation as a first-class concept. Every AI operation includes multiple validators that check different aspects of quality. Validators return statuses:
- `ok`: All checks passed
- `check`: Some issues detected, human review recommended
- `iffy`: Significant problems detected

## Validator Types

- **Structural validators**: Check data integrity (quotes exist, references match, etc.)
- **LLM-as-judge validators**: Use the LLM to critique its own output
- **Heuristic validators**: Compare LLM output against simple rules
- **Informational validators**: Provide context but don't affect validation status

## Validation Aggregation Strategies

When multiple validators run on the same item, their results must be combined into a single validation status. Qux360 uses the `QIndex.from_checks()` method with two aggregation strategies:

### Strategy 1: "strictest" (Default)

**Logic**: Worst status wins - uses priority order `iffy > check > ok`

**When to use**: For **per-item validation** where multiple checks validate different aspects of a single result. You want to know if ANY check fails.

**How it works**:
1. Filters out informational checks (they don't affect status)
2. If ANY remaining check returns `iffy`, the aggregated status is `iffy`
3. Else if ANY remaining check returns `check`, the aggregated status is `check`
4. Else all checks are `ok`, so the aggregated status is `ok`

**Example**: Per-topic validation (3 checks per topic)
- Quote validation: `ok`
- LLM quality: `check`
- LLM assessment: `ok` (informational, ignored)
- **Result**: `check` (worst non-informational status wins)

**Implementation**: [src/qux360/core/qindex.py](src/qux360/core/qindex.py) (lines 127-139)

### Strategy 2: "consensus"

**Logic**: All checks must agree on `ok` status, otherwise returns `check`

**When to use**: For **overall/collection-level validation** where you're aggregating already-validated items and want to know if the collection as a whole is reliable.

**How it works**:
1. If ALL checks return `ok`, the aggregated status is `ok`
2. Otherwise (if ANY check returns `check` or `iffy`), the aggregated status is `check`

**Example**: Overall topic validation (aggregating 10 topics)
- 9 topics: `ok`
- 1 topic: `check`
- **Result**: `check` (no consensus, needs review)

**Key difference from "strictest"**: More forgiving - collapses all non-ok statuses into `check` rather than preserving `iffy`. This makes sense for overall results where you want to signal "needs human review" rather than "failed".

**Implementation**: [src/qux360/core/qindex.py](src/qux360/core/qindex.py) (lines 141-147)

### Hierarchical Validation Pattern

Qux360 uses a **two-level hierarchy** for validation:

```
Level 1: Per-item checks (3-4 validators for single item)
         ↓ Aggregate with "strictest" strategy

Level 2: Per-item validation results (one per item)
         ↓ Aggregate with "consensus" strategy

Level 3: Overall validation result
```

**Example - Topic Extraction**:
1. Each topic gets 3 validators → aggregated with "strictest" → per-topic status
2. All topic statuses → aggregated with "consensus" → overall status

**Example - Theme Extraction**:
1. Each theme gets 4 validators → aggregated with "strictest" → per-theme status
2. All theme statuses → aggregated with custom escalation (based on "consensus") → overall status

### Custom Escalation (Theme Validation)

Theme validation uses a **custom escalation strategy** on top of consensus:

**Implementation**: [src/qux360/core/study.py](src/qux360/core/study.py) (lines 233-270)

**Logic**:
1. Start with `consensus` aggregation
2. Count how many themes have each status (`ok`, `check`, `iffy`)
3. If ANY theme has issues, escalate overall status to `check`
4. Otherwise, use the consensus result

**Why?** Ensures users are always flagged to review problematic themes, even if the majority are ok. This is important because themes represent cross-cutting patterns - even one weak theme deserves attention.

### Informational Checks

Validators marked with `informational=True` are included in the checks list but **don't affect aggregated status**:

- They appear in the nested checks for transparency
- They're filtered out before applying aggregation logic
- If ALL checks are informational, they're treated as validation checks (prevents empty validation)

**Example**: LLM assessment in topic extraction
- Always returns `ok` status
- Provides qualitative feedback (strengths/weaknesses)
- Doesn't change topic's overall validation status

**Implementation**: [src/qux360/core/qindex.py](src/qux360/core/qindex.py) (lines 120-125)

### Visual Summary

| Strategy | When to Use | Logic | Typical Use Case |
|----------|-------------|-------|------------------|
| **strictest** | Per-item validation | Worst status wins (`iffy > check > ok`) | Combining multiple validators for a single topic/theme |
| **consensus** | Overall validation | All must be `ok`, else `check` | Aggregating pre-validated items into collection result |
| **custom escalation** | Sensitive collections | Consensus + escalate if ANY issues | Theme validation (cross-cutting patterns) |

---

## 1. Interviewee Identification

**AI Function**: `Interview.identify_interviewee()`
**Location**: [src/qux360/core/interview.py](src/qux360/core/interview.py) (lines 221-381)

Uses LLM to identify which speaker is the interviewee based on conversation patterns and content.

### 1.1 Heuristic Agreement Validator

**Type**: Structural validator (primary)
**What it validates**: Compares LLM's prediction against a simple word-count heuristic (the speaker with the most words is likely the interviewee).

**Logic**:
1. Calculates total words spoken by each speaker
2. Identifies speaker with highest word count
3. Compares with LLM's prediction
4. Checks if predicted speaker's word ratio meets thresholds

**Status rules**:
- `ok`: Prediction agrees with heuristic AND speaker has ≥60% of words
- `check`: Prediction agrees with heuristic AND speaker has 50-60% of words
- `iffy`: Prediction disagrees with heuristic OR speaker has <50% of words

**Implementation**: [src/qux360/core/validators.py](src/qux360/core/validators.py) (lines 174-273)

### 1.2 Mellea Requirements Validator

**Type**: Informational validator
**What it validates**: Checks whether the LLM response met Mellea's internal validation requirements.

**Logic**:
1. Extracts validation scores from Mellea's response metadata
2. Checks if requirements passed (e.g., speaker ID format, confidence score validity)
3. Reports requirement pass/fail counts

**Status rules**:
- `ok`: All Mellea requirements passed
- `check`: Some Mellea requirements failed
- Always marked informational, so doesn't affect overall validation status

**Implementation**: [src/qux360/core/validators.py](src/qux360/core/validators.py) (lines 91-171)

**Aggregation**: Uses `consensus` strategy (both validators must agree on status)

---

## 2. Topic Extraction (Top-Down)

**AI Function**: `Interview.suggest_topics_top_down()`
**Location**: [src/qux360/core/interview.py](src/qux360/core/interview.py) (lines 806-954)

Asks LLM to identify major topics in an interview and provide supporting quotes.

Each topic is validated with 3 checks:

### 2.1 Quote Validation

**Type**: Structural validator
**What it validates**: Verifies that quotes extracted by the LLM actually exist in the interview transcript.

**Logic**:
1. For each quote, checks if the quote index exists in transcript
2. Attempts exact substring match between quote text and transcript statement
3. Falls back to fuzzy matching (80% similarity threshold) if exact match fails
4. Tracks how many quotes validated successfully

**Status rules**:
- `ok`: ALL quotes validated (exact or fuzzy match)
- `check`: SOME quotes failed validation
- `iffy`: ALL quotes failed validation

**Implementation**: [src/qux360/core/interview.py](src/qux360/core/interview.py) (lines 632-658)
Uses helper: `validate_quote()` (lines 570-610)

### 2.2 LLM Quality Validation

**Type**: LLM-as-judge validator
**What it validates**: Asks the LLM to assess whether the topic is relevant to the interview and well-supported by quotes.

**Logic**:
1. Constructs a prompt with: topic title, topic explanation, quotes, interview context
2. Asks LLM to rate quality as "excellent", "acceptable", or "poor"
3. Parses LLM's rating response

**Status rules**:
- `ok`: LLM rated quality as "excellent"
- `check`: LLM rated quality as "acceptable"
- `iffy`: LLM rated quality as "poor" or parsing failed

**Implementation**: [src/qux360/core/interview.py](src/qux360/core/interview.py) (lines 660-711)
Uses helper: `parse_quality_rating()` in [utils.py](src/qux360/core/utils.py) (lines 101-129)

### 2.3 LLM Assessment

**Type**: Informational validator
**What it validates**: Asks the LLM to provide a qualitative assessment of the topic's strengths and weaknesses.

**Logic**:
1. Constructs a prompt asking for structured assessment
2. LLM generates text with "Strengths:" and "Weaknesses:" sections
3. Stores assessment in validation metadata for human review

**Status rules**:
- Always returns `ok` (informational only, doesn't affect validation status)

**Implementation**: [src/qux360/core/interview.py](src/qux360/core/interview.py) (lines 713-776)

**Per-topic aggregation**: Uses `strictest` strategy (worst status wins across 3 checks)
**Overall aggregation**: Uses `consensus` strategy (aggregates all per-topic validations)
**Aggregation logic**: [src/qux360/core/interview.py](src/qux360/core/interview.py) (lines 778-804)

---

## 3. Theme Extraction (Cross-Interview)

**AI Function**: `Study.suggest_themes()`
**Location**: [src/qux360/core/study.py](src/qux360/core/study.py) (lines 552-825)

Analyzes topics across multiple interviews to identify recurring themes. Each theme is validated with 4 checks:

### 3.1 Topic Hydration Validator

**Type**: Structural validator
**What it validates**: Ensures LLM-generated topic references can be matched back to original Topic objects.

**Context**: The LLM works with simplified topic representations (just titles), but validation needs full Topic objects (with quotes, explanations, interview metadata). This validator performs "hydration" — replacing simplified references with original objects.

**Logic**:
1. Builds lookup table: `(interview_id, topic_title) → original Topic object`
2. For each topic reference in theme, attempts to find matching original Topic
3. Replaces LLM-generated topic with original Topic (preserves quotes and metadata)
4. Tracks topics that couldn't be matched (LLM may have changed topic titles)

**Status rules**:
- `ok`: ALL topic references successfully hydrated
- `check`: SOME topic references failed to hydrate

**Implementation**: [src/qux360/core/study.py](src/qux360/core/study.py) (lines 296-308)
Uses helper: `_hydrate_theme_topics()` (lines 428-492)

### 3.2 Cross-Interview Coverage Validator

**Type**: Structural validator
**What it validates**: Themes should represent patterns that span multiple interviews (not just isolated to one interview).

**Logic**:
1. Extracts `interview_id` from each topic in the theme
2. Counts unique interview IDs
3. Checks if theme spans at least 2 interviews

**Status rules**:
- `ok`: Theme spans ≥2 interviews
- `iffy`: Theme only appears in 1 interview

**Implementation**: [src/qux360/core/study.py](src/qux360/core/study.py) (lines 310-323)

### 3.3 Topic Count Validator

**Type**: Structural validator
**What it validates**: Themes should be supported by multiple topics (not just a single topic).

**Logic**:
1. Counts number of topics in the theme
2. Checks if count meets minimum threshold

**Status rules**:
- `ok`: Theme has ≥2 supporting topics
- `check`: Theme has only 1 supporting topic

**Implementation**: [src/qux360/core/study.py](src/qux360/core/study.py) (lines 325-337)

**Note**: Validators 3.2 and 3.3 together determine the `theme.prospective` flag:
- `prospective = True` if theme has <2 interviews OR <2 topics
- Prospective themes are flagged for human review as potentially emerging patterns

### 3.4 LLM Coherence Assessment

**Type**: LLM-as-judge validator
**What it validates**: Asks the LLM to assess whether supporting topics genuinely relate to each other and to the theme.

**Logic**:
1. Constructs a prompt with: theme title, theme explanation, topic summaries, study context
2. Asks LLM to rate coherence as "Strong", "Acceptable", or "Weak"
3. Parses LLM's rating response

**Status rules**:
- `ok`: LLM rated coherence as "Strong" (tight conceptual fit)
- `check`: LLM rated coherence as "Acceptable" (generally related but loose connections)
- `iffy`: LLM rated coherence as "Weak" (disconnected topics) or parsing failed

**Implementation**: [src/qux360/core/study.py](src/qux360/core/study.py) (lines 343-423)
Uses helper: `parse_coherence_rating()` in [utils.py](src/qux360/core/utils.py) (lines 132-160)

**Per-theme aggregation**: Uses `strictest` strategy (worst status wins across 4 checks)
**Overall aggregation**: Custom escalation strategy (lines 209-270)
- Uses `consensus` as base
- Escalates to `check` if ANY theme has issues (even if majority are OK)
- Ensures users are flagged to review problematic themes

---

## Validation Framework

### Base Infrastructure

All validators inherit from `BaseValidator` abstract class:
- **Location**: [src/qux360/core/validators.py](src/qux360/core/validators.py) (lines 28-88)
- **Provides**: `validate()` interface and automatic `method_name` generation from class name

### Reusable Validators

Two validators are designed to be generic and reusable:

1. **MelleaRequirementsValidator**: Works with ANY Mellea response type
   - Currently used in interviewee identification
   - Could be used for topic/theme extraction

2. **HeuristicAgreementValidator**: Works with ANY speaker prediction
   - Currently used for interviewee identification
   - Could validate other speaker-related predictions

### Utility Functions

Location: [src/qux360/core/utils.py](src/qux360/core/utils.py)

- **parse_quality_rating()** (lines 101-129): Maps LLM quality ratings to statuses
- **parse_coherence_rating()** (lines 132-160): Maps LLM coherence ratings to statuses
- **extract_mellea_validation_status()** (lines 163-266): Extracts Mellea validation metadata
- **print_mellea_validations()** (lines 12-51): Pretty-prints Mellea validation results for debugging

---

## Summary by AI Function

| AI Function | Total Validators | Primary | Informational | LLM-as-Judge | Structural |
|-------------|-----------------|---------|---------------|-------------|-----------|
| **Interviewee Identification** | 2 | 1 | 1 | 0 | 1 |
| **Topic Extraction** | 3 per topic | 2 | 1 | 1 | 1 |
| **Theme Extraction** | 4 per theme | 4 | 0 | 1 | 3 |

**Total Unique Validator Types**: 8 distinct validation checks across 3 AI functions

---

## Key Architectural Patterns

### Two-Tier Validation
Per-item validation → Overall aggregation
- Topics: Each topic gets 3 checks → Overall topic validation
- Themes: Each theme gets 4 checks → Overall theme validation

### Informational Checks
Some validators provide context without affecting validation status:
- LLM assessment in topic extraction (always returns `ok`)
- Mellea requirements in interviewee identification (marked informational)

### LLM-as-Judge
Self-validation by asking the LLM to critique its own output:
- Topic quality validation
- Theme coherence assessment

### Graceful Degradation
Operations continue even when validators detect issues:
- Interviewee identification falls back to heuristic if LLM fails
- Theme/topic extraction returns `check` status on LLM errors instead of crashing

### Composable Validators
The `BaseValidator` pattern allows easy addition of new validators without modifying core logic.

---

## Adding New Validators

To add a new validator:

1. **Create a class** inheriting from `BaseValidator`
2. **Implement `validate()`** method returning `QIndex`
3. **Add to AI function**: Append to validation checks in the appropriate function
4. **Update this documentation**: Add description to relevant section

Example:

```python
from qux360.core.validators import BaseValidator
from qux360.core.qindex import QIndex

class MyCustomValidator(BaseValidator):
    def validate(self, item, context):
        # Your validation logic here
        if condition:
            status = "ok"
            message = "Validation passed"
        else:
            status = "check"
            message = "Issue detected"

        return QIndex.from_check(
            status=status,
            message=message,
            method=self.method_name
        )
```

Then add to AI function:

```python
# In suggest_topics_top_down() or similar
validators = [
    QuoteValidator(),
    LLMQualityValidator(),
    MyCustomValidator(),  # Add your validator
]
```
