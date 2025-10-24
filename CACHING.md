# PyQual Caching System

## Overview

PyQual now includes a transparent caching system that dramatically speeds up development and testing by avoiding repeated expensive operations like:
- File parsing (DOCX/XLSX/CSV)
- LLM-based topic extraction
- LLM-based theme analysis

**Key Features:**
- ✅ Automatic cache detection and loading
- ✅ Hash-based cache invalidation (auto-refresh when source files change)
- ✅ Stores both raw and transformed transcripts
- ✅ Preserves all metadata, topics, and themes
- ✅ Minimal code changes required
- ✅ Cache files stored separately in `.pyqual_cache/` directories

## How It Works

### Automatic Caching (Default Behavior)

```python
# First run: parses file and creates cache
interview = Interview("interview_A.docx")

# Subsequent runs: loads from cache instantly
interview = Interview("interview_A.docx")  # Fast!
```

The cache is stored in `.pyqual_cache/interview_A_state.json` next to the source file.

**Cache invalidation:** If you modify `interview_A.docx`, the cache automatically becomes stale and PyQual will re-parse the file.

### What Gets Cached

**For Interviews:**
- `transcript_raw` (immutable original)
- `transcript` (working copy with transformations)
- `speaker_mapping` (rename history)
- `metadata` (including participant_id)
- `topics_top_down` (cached topic extraction results)
- Source file hash (for invalidation)

**For Studies:**
- All interview states (above)
- `study_context`
- `themes_top_down` (cached theme analysis results)
- Study metadata

## Usage Examples

### 1. Single Interview Caching

```python
from pyqual.core.interview import Interview

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
interview = Interview.load_from_cache(Path(".pyqual_cache/interview_state.json"))
```

### 2. Study-Level Caching

```python
from pyqual.core.study import Study

# Automatic caching for all interviews (default)
study = Study(
    files=["interview_A.docx", "interview_B.docx", "interview_C.docx"],
    study_context="Remote work study"
)

# Each interview loads from cache if available
# First run: parses all files
# Second run: loads all from cache (much faster!)

# Save complete study state (all interviews + themes)
study.save_state(Path("study_cache"))

# Load complete study state
study = Study.load_from_cache(Path("study_cache"))
```

### 3. Fast Iteration on Theme Analysis

The most powerful use case: extract topics once, iterate on themes many times.

**First run (slow - extracts topics):**
```python
study = Study(files=[...], study_context="...")
study.identify_interviewees(m)
study.suggest_topics_all(m, n=5)  # Expensive!
study.save_state(Path("study_cache"))
```

**Subsequent runs (fast - loads cached topics):**
```python
# Load everything from cache
study = Study.load_from_cache(Path("study_cache"))

# Topics already cached - just extract themes
themes = study.suggest_themes(m, n=3)  # Fast!

# Try different parameters
themes = study.suggest_themes(m, n=5, max_quotes_per_topic=3)

# Iterate quickly without re-running expensive topic extraction
```

See [examples/study_themes_from_cache.py](examples/study_themes_from_cache.py) for a complete example.

## Cache File Structure

### Interview Cache (`.pyqual_cache/interview_A_state.json`)

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
  "topics_top_down": {...}
}
```

### Study Cache (`.study_cache/`)

```
.study_cache/
├── study_state.json              # Study-level metadata + themes
├── interview_abc123_state.json   # Interview 1 state
├── interview_def456_state.json   # Interview 2 state
└── interview_ghi789_state.json   # Interview 3 state
```

## Cache Invalidation

Caches automatically become stale when:
- Source file is modified (detected via SHA256 hash)
- Cache file is deleted manually
- Cache file is corrupted

When stale, PyQual automatically re-parses the source file and updates the cache.

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
# PyQual cache directories
.pyqual_cache/
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
find . -type d -name ".pyqual_cache" -exec rm -rf {} +
find . -type d -name ".study_cache" -exec rm -rf {} +
```

### Check cache size
```bash
du -sh examples/data/.pyqual_cache/
du -sh examples/data/.study_cache/
```

### Inspect cache contents
```python
import json
from pathlib import Path

cache_file = Path(".pyqual_cache/interview_A_state.json")
state = json.loads(cache_file.read_text())

print(f"Cached topics: {len(state['topics_top_down']['topics'])}")
print(f"Source hash: {state['source_file_hash']}")
```

## Troubleshooting

### Cache not loading
- Check if cache file exists: `ls .pyqual_cache/`
- Check if source file hash changed (file was modified)
- Enable debug logging: `logging.getLogger("pyqual").setLevel(logging.DEBUG)`

### Cache taking too much space
- Cache files are JSON (compressed would be smaller)
- Each interview state: typically 100KB - 2MB depending on transcript size
- Consider clearing old caches periodically

### Stale cache not refreshing
- Delete cache manually: `rm -rf .pyqual_cache/`
- Verify source file hash: cache invalidation is automatic based on SHA256

## Implementation Details

The caching system is implemented in [src/pyqual/core/cache.py](src/pyqual/core/cache.py) as a separate module to keep `Interview` and `Study` classes focused.

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
