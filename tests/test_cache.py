"""
Tests for PyQual caching functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd

from pyqual.core.interview import Interview
from pyqual.core.study import Study
from pyqual.core.cache import (
    save_interview_state,
    load_interview_state,
    _compute_file_hash,
    _is_cache_valid
)


class TestInterviewCache:
    """Tests for Interview caching."""

    def test_save_and_load_interview_state(self, tmp_path):
        """Test saving and loading interview state."""
        # Create a minimal interview
        interview = Interview()
        interview.file_path = tmp_path / "test.csv"
        interview.metadata = {"participant_id": "P1", "test": "value"}
        interview.speaker_mapping = {"Speaker1": "P1"}

        # Save state
        cache_path = tmp_path / "cache.json"
        save_interview_state(interview, cache_path)

        assert cache_path.exists()

        # Load state
        loaded = load_interview_state(cache_path, validate_source=False)

        assert loaded.id == interview.id
        assert loaded.metadata == interview.metadata
        assert loaded.speaker_mapping == interview.speaker_mapping

    def test_cache_contains_both_transcripts(self, tmp_path):
        """Test that cache stores both raw and working transcripts."""
        interview = Interview()
        interview.file_path = tmp_path / "test.csv"

        # Modify working transcript
        interview.transcript_raw = pd.DataFrame({
            "timestamp": ["00:00:00"],
            "speaker": ["Speaker1"],
            "statement": ["Original"],
            "speaker_id": [None],
            "codes": [[]],
            "themes": [[]]
        })
        interview.transcript = interview.transcript_raw.copy()
        interview.transcript.loc[0, "statement"] = "Modified"

        # Save and load
        cache_path = tmp_path / "cache.json"
        save_interview_state(interview, cache_path)
        loaded = load_interview_state(cache_path, validate_source=False)

        # Verify both transcripts preserved
        assert loaded.transcript_raw.loc[0, "statement"] == "Original"
        assert loaded.transcript.loc[0, "statement"] == "Modified"

    def test_cache_file_hash(self, tmp_path):
        """Test that cache includes source file hash."""
        # Create a source file
        source_file = tmp_path / "test.txt"
        source_file.write_text("test content")

        interview = Interview()
        interview.file_path = source_file

        # Save cache
        cache_path = tmp_path / "cache.json"
        save_interview_state(interview, cache_path)

        # Check hash in cache
        with open(cache_path) as f:
            state = json.load(f)

        assert "source_file_hash" in state
        assert len(state["source_file_hash"]) == 64  # SHA256 hex length

    def test_cache_invalidation_on_file_change(self, tmp_path):
        """Test that cache becomes invalid when source file changes."""
        # Create source file and cache
        source_file = tmp_path / "test.txt"
        source_file.write_text("original content")

        interview = Interview()
        interview.file_path = source_file

        cache_path = tmp_path / "cache.json"
        save_interview_state(interview, cache_path)

        # Cache should be valid
        assert _is_cache_valid(cache_path, source_file)

        # Modify source file
        source_file.write_text("modified content")

        # Cache should now be invalid
        assert not _is_cache_valid(cache_path, source_file)


class TestStudyCache:
    """Tests for Study caching."""

    def test_save_and_load_study_state(self, tmp_path):
        """Test saving and loading study state."""
        # Create study with interviews
        study = Study()
        study.study_context = "Test context"
        study.metadata = {"test": "value"}

        # Add minimal interviews
        for i in range(3):
            interview = Interview()
            interview.id = f"interview_{i}"
            interview.file_path = tmp_path / f"interview_{i}.csv"
            interview.metadata = {"participant_id": f"P{i}"}
            study.documents.append(interview)

        # Save study state
        cache_dir = tmp_path / "study_cache"
        study.save_state(cache_dir)

        assert cache_dir.exists()
        assert (cache_dir / "study_state.json").exists()

        # Load study state
        loaded = load_study_state(cache_dir)

        assert loaded.study_context == study.study_context
        assert loaded.metadata == study.metadata
        assert len(loaded.documents) == 3

    def test_study_cache_preserves_interview_order(self, tmp_path):
        """Test that study cache preserves interview order."""
        study = Study()

        # Add interviews with specific IDs
        interview_ids = ["interview_a", "interview_b", "interview_c"]
        for iid in interview_ids:
            interview = Interview()
            interview.id = iid
            interview.file_path = tmp_path / f"{iid}.csv"
            study.documents.append(interview)

        # Save and load
        cache_dir = tmp_path / "study_cache"
        study.save_state(cache_dir)
        loaded = load_study_state(cache_dir)

        # Verify order preserved
        loaded_ids = [i.id for i in loaded.documents]
        assert loaded_ids == interview_ids


class TestAutomaticCaching:
    """Tests for automatic cache loading in constructors."""

    def test_interview_auto_loads_from_cache(self, tmp_path):
        """Test that Interview constructor auto-loads from cache."""
        # Create a real CSV file
        csv_file = tmp_path / "interview.csv"
        csv_file.write_text(
            "timestamp,speaker,statement\n"
            "00:00:00,Speaker1,Hello world\n"
        )

        # First load: parses file and creates cache
        interview1 = Interview(csv_file, use_cache=True)
        interview1.metadata["test_marker"] = "first_load"
        interview1.save_state()  # Save with marker

        # Second load: should load from cache
        interview2 = Interview(csv_file, use_cache=True)

        # Should have the marker from cache
        assert interview2.metadata.get("test_marker") == "first_load"

    def test_cache_can_be_disabled(self, tmp_path):
        """Test that caching can be disabled."""
        csv_file = tmp_path / "interview.csv"
        csv_file.write_text(
            "timestamp,speaker,statement\n"
            "00:00:00,Speaker1,Hello\n"
        )

        # Create cache
        interview1 = Interview(csv_file, use_cache=True)
        interview1.metadata["cached"] = True
        interview1.save_state()

        # Load with caching disabled - should not have cached metadata
        interview2 = Interview(csv_file, use_cache=False)
        assert "cached" not in interview2.metadata

    def test_validation_results_are_cached(self, tmp_path):
        """Test that topic validation results are cached and restored."""
        from pyqual.core.models import TopicList, Topic, Quote
        from pyqual.core.iffy import IffyIndex

        # Create an interview
        interview = Interview()
        interview.file_path = tmp_path / "test.csv"

        # Manually set topics and validation (simulating suggest_topics_top_down)
        interview.topics_top_down = TopicList(
            topics=[
                Topic(
                    topic="Test Topic",
                    explanation="Test explanation",
                    quotes=[Quote(index=0, timestamp="00:00:00", speaker="P1", quote="Test")]
                )
            ],
            interview_id="test_123"
        )

        interview.topics_top_down_validation = IffyIndex.from_check(
            method="test_validation",
            status="ok",
            explanation="All topics validated successfully"
        )

        # Save to cache
        cache_path = tmp_path / "cache.json"
        save_interview_state(interview, cache_path)

        # Load from cache
        loaded = load_interview_state(cache_path, validate_source=False)

        # Verify validation was restored
        assert loaded.topics_top_down_validation is not None
        assert loaded.topics_top_down_validation.status == "ok"
        assert loaded.topics_top_down_validation.explanation == "All topics validated successfully"
        assert loaded.topics_top_down_validation.method == "test_validation"


def load_study_state(cache_dir):
    """Helper to load study state for tests."""
    from pyqual.core.cache import load_study_state as _load
    return _load(Path(cache_dir))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
