# Qux360 Caching System

## Overview

Qux360 includes a transparent caching system that dramatically speeds up development and testing by avoiding repeated expensive operations:
- ✅ File parsing (DOCX/XLSX/CSV)
- ✅ LLM-based topic extraction **with full validation results**
- ❌ Theme analysis (NOT cached - regenerated each run for flexibility)

**Key Features:**
- ✅ Automatic cache detection and loading
- ✅ Hash-based cache invalidation (auto-refresh when source files change)
- ✅ Stores both raw and transformed transcripts
- ✅ Caches topic extraction results with complete validation details
- ✅ Minimal code changes required
- ✅ Cache files stored separately in `.qux360_cache/` directories

## How It Works

### Automatic Caching (Default Behavior)

```python
# First run: parses file and creates cache
interview = Interview("interview_A.docx")

# Subsequent runs: loads from cache instantly
interview = Interview("interview_A.docx")  # Fast!
```

The cache is stored in `.qux360_cache/interview_A_state.json` next to the source file.

**Cache invalidation:** If you modify `interview_A.docx`, the cache automatically becomes stale and Qux360 will re-parse the file.

### What Gets Cached

**For Interviews:**
- `transcript_raw` (immutable original)
- `transcript` (working copy with transformations)
- `speaker_mapping` (rename history)
- `metadata` (including participant_id)
- `topics_top_down` (TopicList - the extracted topics)
- `topics_top_down_validation` (IffyIndex - overall validation with nested per-topic validations)
- Source file hash (for invalidation)

**For Studies:**
- All interview states (above)
- `study_context`
- Study metadata
- ❌ **NOT cached:** `themes_top_down` (themes regenerated each run)

**Why themes aren't cached:**
- Theme analysis is relatively fast compared to topic extraction
- Parameters often change during iteration (n, max_quotes_per_topic, etc.)
- Themes depend on which topics are included (easy to experiment)
- Validation framework for themes not yet implemented

## Usage Examples

### 1. Single Interview Caching

```python
from qux360.core.interview import Interview

# Automatic caching (default)
interview = Interview("data/interview.docx")

# Disable caching if needed
interview = Interview("data/interview.docx", use_cache=False)

# Custom cache location
interview = Interview("data/interview.docx", cache_dir=Path("custom_cache"))

# Manually save state after processing
interview.suggest_topics_top_down(m, n=5)
interview.save_state()  # Saves to default location

# Load from specific cache file
interview = Interview.load_from_cache(Path(".qux360_cache/interview_state.json"))
```

### 2. Study-Level Caching

```python
from qux360.core.study import Study

# Automatic caching for all interviews (default)
study = Study(
    files=["interview_A.docx", "interview_B.docx", "interview_C.docx"],
    study_context="Remote work study"
)

# Each interview loads from cache if available
# First run: parses all files
# Second run: loads all from cache (much faster!)

# Save study state (interviews + topics, but NOT themes)
study.save_state(Path("study_cache"))

# Load study state (interviews + topics ready for theme analysis)
study = Study.load_from_cache(Path("study_cache"))

# Themes are NOT cached - run fresh each time
themes = study.suggest_themes(m, n=3)
```

### 3. Fast Iteration on Theme Analysis

The most powerful use case: extract topics once, iterate on themes many times.

See `examples/study_suggest_themes_from_cache.py` for a complete example.

**First run (no cache - extracts topics):**
```python
# Automatic caching enabled (default)
study = Study(files=[...], study_context="...")

# Check if topics need extraction
if any(i.topics_top_down is None for i in study.documents):
    # First run: extract topics (expensive!)
    study.identify_interviewees(m)
    study.suggest_topics_all(m, n=5)

    # Save to cache for next time
    for interview in study.documents:
        interview.save_state()

# Run theme analysis (always runs, not cached)
themes = study.suggest_themes(m, n=3)
```

**Subsequent runs (cache hit - skips topic extraction):**
```python
# Topics automatically loaded from cache
study = Study(files=[...], study_context="...")

# Topics already loaded - skip straight to theme analysis
themes = study.suggest_themes(m, n=3)  # Fast!

# Try different parameters
themes = study.suggest_themes(m, n=5, max_quotes_per_topic=3)
```

**View cached validation results:**
```python
# After loading from cache, reconstruct ValidatedList to see details
for interview in study.documents:
    topics_validated = interview.get_topics_validated()
    if topics_validated:
        topics_validated.print_summary(
            title=f"Topics for {interview.get_participant_id()}",
            item_label="Topic"
        )
```

## Cache File Structure

### Interview Cache (`.qux360_cache/interview_A_state.json`)

```json
{
  "version": "1.0",
  "id": "interview_abc123",
  "file_path": "/path/to/interview_A.docx",
  "source_file_hash": "sha256...",
  "transcript_raw": [...],
  "transcript": [...],
  "speaker_mapping": {"Speaker1": "P1"},
  "metadata": {"participant_id": "P1"},
  "topics_top_down": {
    "topics": [
      {
        "topic": "...",
        "explanation": "...",
        "quotes": [...],
        "interview_id": "interview_abc123"
      }
    ],
    "generated_at": "2025-01-15T10:30:00"
  },
  "topics_top_down_validation": {
    "status": "ok",
    "explanation": "Consensus: all 5 checks passed",
    "checks": [
      {
        "status": "ok",
        "explanation": "All 3 quotes validated successfully",
        "method": "quote_validation",
        "checks": [...]
      },
      ...
    ]
  }
}
```

### Study Cache (`.study_cache/`)

```
.study_cache/
├── study_state.json              # Study-level metadata (NO themes)
├── interview_abc123_state.json   # Interview 1 state with topics + validation
├── interview_def456_state.json   # Interview 2 state with topics + validation
└── interview_ghi789_state.json   # Interview 3 state with topics + validation
```

**Note:** Themes are NOT included in the study cache and must be regenerated each run.

## Validation Caching

A powerful feature of Qux360's caching system is that it preserves **complete validation results** from topic extraction.

### What Validation Data is Cached

When topics are extracted via `suggest_topics_top_down()`, two types of validation are cached:

1. **Overall validation** (`topics_top_down_validation`): Aggregate status across all topics
2. **Per-topic validations** (nested in `.checks`): Individual validation for each topic including:
   - Quote validation (exact/fuzzy match against transcript)
   - LLM quality assessment
   - Informational assessments (strengths/weaknesses)

### Accessing Cached Validation

```python
# Load interview from cache
interview = Interview("file.csv")  # Auto-loads from cache

# Check if validation was cached
if interview.topics_top_down_validation:
    print(f"Validation status: {interview.topics_top_down_validation.status}")
    print(f"Explanation: {interview.topics_top_down_validation.explanation}")

# Reconstruct full ValidatedList to use print_summary()
topics_validated = interview.get_topics_validated()
if topics_validated:
    topics_validated.print_summary(
        title="Cached Topics with Validation",
        item_label="Topic"
    )
```

### Why Cache Validation?

- **Transparency**: Know how good your cached topics were when generated
- **Reproducibility**: Validation results preserved across runs
- **Debugging**: Check topic quality if theme analysis produces poor results
- **Audit trail**: Track quality over time and across different model runs

### Example Output

```
Cached Topics for Alex Morgan (13 items)
============================================================

⚠️ Overall: CHECK
   No consensus on validation
============================================================

Topic 1: Remote Work Challenges

   Explanation: Employees struggle with work-life boundaries...

   Quotes (2):
      [5] 00:02:15 Alex Morgan:
      The biggest challenge is knowing when to stop working...

   ✅ Validation: OK
      ✅ [quote_validation] ok — All 2 quotes validated
      ✅ [llm_validation] ok — excellent: highly relevant
      ℹ️  [llm_assessment] ok — Strengths: ... | Weaknesses: ...
```

## Cache Invalidation

Caches automatically become stale when:
- Source file is modified (detected via SHA256 hash)
- Cache file is deleted manually
- Cache file is corrupted

When stale, Qux360 automatically re-parses the source file and updates the cache.

## Best Practices

### Development Workflow

1. **Initial processing:** Run full pipeline once with expensive operations
   ```python
   study = Study(files=[...])
   study.identify_interviewees(m)
   study.suggest_topics_all(m)
   study.save_state(Path("study_cache"))
   ```

2. **Fast iteration:** Load from cache and iterate
   ```python
   study = Study.load_from_cache(Path("study_cache"))
   # Iterate on theme analysis, visualization, etc.
   ```

3. **Commit test fixtures:** Version control cached states for reproducible tests
   ```bash
   git add tests/fixtures/.study_cache/
   ```

### Testing

Create fixture data with cached topics:
```python
# tests/fixtures/setup_fixtures.py
study = Study(files=[...])
study.suggest_topics_all(m)
study.save_state(Path("tests/fixtures/study_cache"))

# tests/test_themes.py
study = Study.load_from_cache(Path("tests/fixtures/study_cache"))
themes = study.suggest_themes(m)  # Fast - topics already cached
assert len(themes.result.themes) > 0
```

### Version Control

The `.gitignore` is configured to exclude cache directories by default:
```gitignore
# Qux360 cache directories
.qux360_cache/
.study_cache/
```

**Exception:** You may want to commit test fixture caches for reproducibility:
```bash
git add -f tests/fixtures/.study_cache/
```

## Performance Impact

Typical speedups:

| Operation | Without Cache | With Cache | Speedup |
|-----------|--------------|------------|---------|
| Parse DOCX/XLSX | 100-500ms | 10-50ms | 10-20x |
| Load 10 interviews | 2-5s | 100-300ms | 10-20x |
| Topic extraction | 30-60s | Instant | ∞ |
| Full study pipeline | 5-10 min | 30-60s | 10x |

**Real-world impact:**
- Development iteration: Minutes → Seconds
- Theme analysis testing: Can run dozens of experiments quickly
- CI/CD tests: Faster with fixture caches

## Disabling Caching

If you need to disable caching:

```python
# Single interview
interview = Interview("file.docx", use_cache=False)

# Study (disables for all interviews)
study = Study(files=[...], use_cache=False)
```

## Cache Management

### Clear all caches
```bash
find . -type d -name ".qux360_cache" -exec rm -rf {} +
find . -type d -name ".study_cache" -exec rm -rf {} +
```

### Check cache size
```bash
du -sh examples/data/.qux360_cache/
du -sh examples/data/.study_cache/
```

### Inspect cache contents
```python
import json
from pathlib import Path

cache_file = Path(".qux360_cache/interview_A_state.json")
state = json.loads(cache_file.read_text())

print(f"Cached topics: {len(state['topics_top_down']['topics'])}")
print(f"Source hash: {state['source_file_hash']}")
```

## Troubleshooting

### Cache not loading
- Check if cache file exists: `ls .qux360_cache/`
- Check if source file hash changed (file was modified)
- Enable debug logging: `logging.getLogger("qux360").setLevel(logging.DEBUG)`

### Cache taking too much space
- Cache files are JSON (compressed would be smaller)
- Each interview state: typically 100KB - 2MB depending on transcript size
- Consider clearing old caches periodically

### Stale cache not refreshing
- Delete cache manually: `rm -rf .qux360_cache/`
- Verify source file hash: cache invalidation is automatic based on SHA256

## Implementation Details

The caching system is implemented in [src/qux360/core/cache.py](src/qux360/core/cache.py) as a separate module to keep `Interview` and `Study` classes focused.

Key functions:
- `save_interview_state()` - Serialize interview to JSON
- `load_interview_state()` - Deserialize interview from JSON
- `try_load_or_parse()` - Smart loader with cache-first fallback
- `_compute_file_hash()` - SHA256 hash for invalidation
- `_is_cache_valid()` - Freshness check

The `Interview` and `Study` classes expose simple wrappers:
- `interview.save_state()`
- `Interview.load_from_cache(path)`
- `study.save_state(dir)`
- `Study.load_from_cache(dir)`
