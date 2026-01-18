import os
import difflib
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from textwrap import shorten
import uuid
import spacy
import time
from datetime import datetime

from mellea import MelleaSession
from mellea.stdlib.sampling import RejectionSamplingStrategy
from qux360.core import QIndex
from qux360.core import Validated, ValidatedList
from qux360.core.utils import print_mellea_validations
import copy

from .models import TopicList, Quote, IntervieweeIdentification
from .utils import validate_transcript_columns, format_quotes_for_display, parse_quality_rating
from .validators import MelleaRequirementsValidator, HeuristicAgreementValidator
from ..io.docx_parser import parse_docx
from ..io.xlsx_parser import parse_xlsx
from ..io.csv_parser import parse_csv

logger = logging.getLogger(__name__)

class Interview:

    def __init__(self,
                 file: Optional[Union[str, Path]] = None,
                 metadata: Optional[dict] = None,
                 headers: Optional[dict] = None,
                 has_headers = True,
                 use_cache: bool = True,
                 cache_dir: Optional[Path] = None):
        """
        Initialize an Interview.

        - Always generates a unique UUID-based id.
        - If a file is provided, attempts to load from cache (if use_cache=True)
        - Falls back to parsing if cache is stale or missing
        - Keeps both raw (immutable) and working (mutable) transcripts.

        Parameters
        ----------
        file : str or Path, optional
            Path to interview transcript file (DOCX/XLSX/CSV)
        metadata : dict, optional
            Additional metadata to store with interview
        headers : dict, optional
            Custom column header mapping for XLSX/CSV files
        has_headers : bool, default=True
            Whether the file has a header row (XLSX/CSV only)
        use_cache : bool, default=True
            If True, attempts to load from cache before parsing
        cache_dir : Path, optional
            Custom cache directory. Defaults to .qux360_cache/ next to source file
        """
        logger.debug(f"\nInit Interview - File Path: {file} | Usign cache {use_cache}")
        # Try smart loading with cache if file provided
        if file and use_cache:
            from .cache import try_load_or_parse
            cached_interview = try_load_or_parse(
                Path(file),
                cache_dir=cache_dir,
                metadata=metadata,
                headers=headers,
                has_headers=has_headers
            )
            # Copy all attributes from cached interview
            self.__dict__.update(cached_interview.__dict__)
            return

        # Standard initialization (no cache or cache disabled)
        self.id = f"interview_{uuid.uuid4().hex[:8]}"
        logger.debug(f"Interview ID: {self.id}")
        self.metadata = metadata or {}
        self.file_path = Path(file) if file else None

        self.transcript_raw = self._init_transcript(file, headers=headers, has_headers=has_headers)
        self.transcript = copy.deepcopy(self.transcript_raw)
        self.speaker_mapping = None
        self.topics_top_down = None
        self.topics_top_down_validation = None  # Stores validation results from topic extraction


    def __repr__(self):
        return f"<Interview {self.id}, {len(self.transcript)} turns, metadata={self.metadata}>"
    
    def _empty_transcript(self):
        logger.debug(f"\nEmpty transcript - Interview ID: {self.id}\n")
        return pd.DataFrame(columns=[
            "timestamp", "speaker_id", "speaker", "statement", "codes", "themes"
        ])

    def _init_transcript(self, file, headers: Optional[dict] = None, has_headers = True):
        logger.debug(f"Init transcript - Interview ID: {self.id} | File: {file}")
        if not file:
            return self._empty_transcript()
        
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")

        ext = Path(file).suffix.lower()
        if ext == ".docx":
            return parse_docx(file)
        elif ext in (".xls", ".xlsx"):
            return parse_xlsx(file, headers=headers, has_headers=has_headers)
        elif ext == ".csv":
            return parse_csv(file, headers=headers, has_headers=has_headers)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


    def _load_spacy_model(self, model: str = "en_core_web_trf"):
        """Try loading a spaCy model, with fallback to lg → sm."""
        logger.debug(f"Loading spacy model: {model} - Interview ID: {self.id}")
        candidates = [model, "en_core_web_lg", "en_core_web_sm"]
        for candidate in candidates:
            try:
                nlp = spacy.load(candidate)
                logger.debug(f"Using spaCy model: {candidate} - Interview ID: {self.id}")
                return nlp
            except OSError:
                logger.error(f"spaCy model '{candidate}' not found.")
        raise RuntimeError("No suitable spaCy model installed.")
    

    
    def reset_transcript(self):
        """Reset the working transcript back to the raw version."""
        logger.debug(f"Reset transcript - Interview ID: {self.id}")
        self.transcript = copy.deepcopy(self.transcript_raw)

    def load_file(self, file: str | Path, headers: Optional[dict] = None, has_headers = True):
        """
        Load a transcript file into the interview (overwrites both raw and working).
        """
        logger.debug(f"Load transcript file - Interview ID: {self.id} | File: {file}")
        self.transcript_raw = self._init_transcript(file, headers=headers, has_headers=has_headers)
        self.transcript = copy.deepcopy(self.transcript_raw)

    def add_code(self, row: int, code: str):
        """Attach a code to a specific row in the working transcript."""
        logger.debug(f"Add code - Row: {row} | Code: {code} | Interview ID: {self.id}")
        current = self.transcript.at[row, "codes"]
        if not isinstance(current, list):
            self.transcript.at[row, "codes"] = []
        self.transcript.at[row, "codes"].append(code)

    def to_xlsx(self, path: str, include_enriched: bool = True):
        """
        Export transcript to a well-formatted Excel file using XlsxWriter.

        Parameters
        ----------
        path : str
            Path to save the Excel file.
        include_enriched : bool, default=True
            If False, excludes speaker_id, codes, themes.
        """
        logger.debug(f"Export to XLSX - Interview ID: {self.id} | File: {path}")
        df = copy.deepcopy(self.transcript)

        # Drop enrichment columns if not requested
        if not include_enriched:
            df.drop(columns=["speaker_id", "codes", "themes"], inplace=True)

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Transcript", index=False)
            ws = writer.sheets["Transcript"]

            # Define column formats
            wrap_top = writer.book.add_format({"text_wrap": True, "valign": "top"})

            # Column width spec
            col_widths = {"timestamp": 10, "speaker_id": 10, "speaker": 25, "statement": 60, "codes": 20, "themes": 20}

            # Apply formatting
            for idx, col in enumerate(df.columns):
                ws.set_column(idx, idx, col_widths.get(col, 20), wrap_top)

            # Freeze header row
            ws.freeze_panes(1, 0)

    def show(self, rows: int = 5, speaker: Optional[str] = None, width: int = 60):
        """Pretty-print transcript preview in a clean 3-column layout."""
        logger.debug(f"View transcript (pretty-print) - Interview ID: {self.id} | Speaker: {speaker} | Rows: {rows}")
        df = self.transcript
        if df.empty:
            print("Transcript is empty.")
            return

        if speaker:
            df = df[df["speaker"] == speaker]

        for _, row in df.head(rows).iterrows():
            ts = str(row["timestamp"])
            sp = str(row["speaker"])
            st = str(row["statement"])[:width]
            print(f"{ts:>8} | {sp:<20} | {st}")

    def get_speakers(self) -> list[str]:
        """Return a list of unique speakers in the working transcript."""
        logger.debug(f"Get unique speakers - Interview ID: {self.id}")
        if "speaker" not in self.transcript.columns:
            return []
        speakers = (
            self.transcript["speaker"]
            .dropna()
            .map(str.strip)
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        return speakers

 
    def identify_interviewee(self, client, snippet_size: int = 25, model: str = "watsonx/meta-llama/llama-3-3-70b-instruct", max_retries: int = 3) -> Validated[IntervieweeIdentification]:
        """
        Identify the likely interviewee using AI analysis with validation.

        This method uses graceful degradation: if AI identification fails due to
        network errors, API issues, or LLM formatting problems, it automatically
        falls back to a word-count heuristic and returns a result with "iffy"
        status rather than raising an exception.

        Validation Strategy
        -------------------
        Heuristic validation is applied to check word count agreement with AI result.

        Parameters
        ----------
        client
            Instructor client for AI analysis (required)
        snippet_size : int, default=25
            Number of transcript rows to analyze (from the beginning)
        model : str, default="watsonx/meta-llama/llama-3-3-70b-instruct"
            LLM model to use for identification
        max_retries : int, default=3
            Number of retries if structured output validation fails

        Returns
        -------
        Validated[IntervieweeIdentification]
            Always returns a Validated result, even on AI failure.

            On success:
            - result: LLM identification with confidence and explanation
            - validation.status: "ok", "check", or "iffy" based on heuristic agreement

            On AI failure (network, API, validation errors):
            - result: Heuristic fallback (speaker with most words)
            - validation.status: "iffy" with error details in explanation
            - result.confidence: "low"

            Always check validation.status before using the result.

        Raises
        ------
        ValueError
            If transcript is empty or missing required columns.
            These are data validation errors (caller's responsibility).
            AI/LLM failures do NOT raise exceptions - they return degraded results.
        """
        logger.info(f"Identify Interviewee - Interview ID: {self.id}")

        # Data precondition checks - use exceptions, not QIndex
        if self.transcript.empty:
            raise ValueError("Transcript is empty")
        if 'speaker' not in self.transcript.columns:
            raise ValueError("Transcript is missing 'speaker' column")

        validate_transcript_columns(self.transcript, ['timestamp', 'speaker', 'statement'])

        # Prepare transcript snippet for LLM
        snippet = "\n".join(
            f"[{row['timestamp']}] {row['speaker']}: {shorten(str(row['statement']), width=120)}"
            for _, row in self.transcript.head(snippet_size).iterrows()
        )

        # Get list of speakers for requirements
        speakers_list = ", ".join(self.get_speakers())

        # Call LLM for structured interviewee identification
        logger.info(f"Calling LLM for interviewee identification - Interview ID: {self.id}")

        # Build prompt with requirements embedded
        prompt = f"""You are analyzing an interview transcript to identify the interviewee.

In interviews, typically:
- One or more speakers are interviewers (asking questions, shorter responses)
- One speaker is the interviewee (giving detailed answers, longer responses)

Requirements:
- The interviewee field must be one of these speaker IDs: {speakers_list}
- The confidence field must be 'high', 'medium', or 'low'
- The explanation must explain the conversation patterns that led to this identification (1-2 sentences)

Transcript Snippet:
{snippet}

Identify the interviewee, assess your confidence, and provide an explanation."""

        try:
            # Use Instructor for structured output
            identification = client.chat.completions.create(
                model=model,
                response_model=IntervieweeIdentification,
                messages=[{"role": "user", "content": prompt}],
                max_retries=max_retries,
                temperature=0.0,
            )

            logger.info(f"LLM identified: {identification.interviewee} (self-confidence: {identification.confidence})")

            # Heuristic validation (primary)
            logger.debug(f"Calculating heuristic agreement - Interview ID: {self.id}")
            heuristic_validator = HeuristicAgreementValidator(
                ok_threshold=0.60,
                check_threshold=0.50
            )
            heuristic_check = heuristic_validator.validate(identification.interviewee, self.transcript)

            # Use heuristic check as overall validation
            overall_validation = heuristic_check

            # Update metadata only if validation passed
            if overall_validation.status in ("ok", "check"):
                self.metadata["participant_id"] = identification.interviewee

            return Validated(
                result=identification,
                validation=overall_validation
            )

        except Exception as e:
            logger.error(f"Interviewee identification failed: {str(e)}")

            # Calculate heuristic fallback
            counts = self.transcript.groupby("speaker")["statement"].apply(
                lambda x: x.str.split().str.len().sum()
            )
            predicted_heuristic = counts.idxmax()
            total_words = counts.sum()
            heuristic_ratio = counts[predicted_heuristic] / total_words if total_words > 0 else 0.0

            # Create fallback IntervieweeIdentification object
            fallback_identification = IntervieweeIdentification(interviewee=predicted_heuristic, confidence="low", explanation=f"AI identification failed, using heuristic fallback (word count analysis)")

            # Return with iffy status on failure
            return Validated(
                result=fallback_identification,
                validation=QIndex.from_check(
                    method="identification_error",
                    status="iffy",
                    explanation=f"AI identification failed, using heuristic fallback, speaker: {predicted_heuristic}, word ratio: {heuristic_ratio}, Exception: {str(e)} ",
                    metadata={"heuristic_prediction": predicted_heuristic, "word_ratio": heuristic_ratio}
                )
            )

    def detect_entities(self, model: str = "en_core_web_trf", verbose: bool = False) -> list[dict]:
        """
        Detect PERSON, ORG, and GPE entities in transcript statements.

        Parameters
        ----------
        model : str, default="en_core_web_trf"
            Preferred spaCy model (falls back to lg → sm).
        verbose : bool, default=False
            If False (default), group rows per entity.
            If True, return every occurrence separately.

        Returns
        -------
        list of dict
            Compact mode: {"entity": "IBM", "label": "ORG", "rows": [3, 7]}
            Verbose mode: {"entity": "IBM", "label": "ORG", "row": 3, "statement": "..."}
        """
        logger.debug(f"Detect entities -Interview ID: {self.id} |  Model: {model}")
        nlp = self._load_spacy_model(model)

        results = []
        validate_transcript_columns(self.transcript, ['statement'])

        if verbose:
            for idx, text in self.transcript["statement"].dropna().items():
                doc = nlp(str(text))
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE"]:
                        results.append({
                            "entity": ent.text,
                            "label": ent.label_,
                            "row": idx,
                            "statement": text
                        })
            return results
        else:
            entities = {}
            for idx, text in self.transcript["statement"].dropna().items():
                doc = nlp(str(text))
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "GPE"]:
                        key = (ent.text, ent.label_)
                        if key not in entities:
                            entities[key] = []
                        entities[key].append(idx)

            return [
                {"entity": ent, "label": label, "rows": rows}
                for (ent, label), rows in entities.items()
            ]
    

    def build_replacement_map(self, entities: list[dict]) -> dict:
        logger.debug(f"Building replacement map - Interview ID: {self.id}")
        replacements = {}
        counters = {"PERSON": 0, "ORG": 0, "GPE": 0}

        for i, e in enumerate(entities):
            if 'entity' not in e or 'label' not in e:
                raise KeyError(f"Missing 'entity' or 'label' in entity at index {i}: {e}")

            logger.debug(f"\tEntity: {e['entity']} | Label: {e['label']}")
            entity, label = e["entity"], e["label"]

            # Assign a new placeholder only if this entity not seen before
            if entity not in replacements:
                counters[label] += 1
                replacements[entity] = f"[{label}_{counters[label]}]"

        return replacements
    
    def anonymize_speakers_generic(self) -> dict:
        """
        First-pass anonymization:
        Replace speaker names with neutral labels (Speaker 1, Speaker 2, ...).

        Returns
        -------
        dict
            Mapping of original names -> generic anonymized labels.
        """
        logger.debug(f"Anonymize speakers - Interview ID: {self.id}")
        speakers = self.get_speakers()
        if not speakers:
            return {}

        mapping = {sp: f"Speaker{i+1}" for i, sp in enumerate(speakers)}

        validate_transcript_columns(self.transcript, ['speaker'])

        self.transcript["speaker"] = self.transcript["speaker"].map(mapping)
        self.speaker_mapping = mapping

        # Update participant_id in metadata if already set
        pid = self.metadata.get("participant_id")
        if pid and pid in mapping:
            self.metadata["participant_id"] = mapping[pid]

        return mapping

    def anonymize_statements(self, replacements: dict):
        logger.debug(f"Anonymize statements - Interview ID: {self.id}")
        if not replacements:
            logger.info("No replacements provided.")
            return

        def replace_text(text: str) -> str:
            anonymized = str(text)
            for original, anon in replacements.items():
                anonymized = anonymized.replace(original, anon)
            return anonymized

        # Replace inside statements
        validate_transcript_columns(self.transcript, ['statement'])

        self.transcript["statement"] = self.transcript["statement"].map(replace_text)


    def rename_speaker(self, old: str, new: str) -> dict:
        """
        Rename a speaker label in the transcript and update mapping.

        Parameters
        ----------
        old : str
            The current speaker label to rename (must exist in transcript).
        new : str
            The new label to assign.
        """
        logger.debug(f"Rename speaker - Old: {old} -> New: {new} | Interview ID: {self.id}")
        validate_transcript_columns(self.transcript, ['speaker'])

        if old not in self.transcript["speaker"].unique():
            raise ValueError(f"Speaker '{old}' not found in transcript.")

        # Apply renaming in transcript
        self.transcript["speaker"] = self.transcript["speaker"].replace({old: new})

        # Update mapping cumulatively
        if not hasattr(self, "speaker_mapping"):
            self.speaker_mapping = {sp: sp for sp in self.get_speakers()}

        composed = {}
        for original, current in self.speaker_mapping.items():
            if current == old:
                composed[original] = new
            else:
                composed[original] = current
        self.speaker_mapping = composed

         # Update participant_id in metadata if necessary
        pid = self.metadata.get("participant_id")
        if pid == old:
            self.metadata["participant_id"] = new

        return self.speaker_mapping

    def set_participant_id(self, pid: str):
        """
        Set or update the participant_id in metadata.

        - If `pid` matches a speaker in the transcript, it is stored directly.
        - If a speaker_mapping exists and `pid` matches an *original* name,
        it is translated to the current anonymized/renamed label.
        """
        logger.debug(f"Set participant ID - Interview ID: {self.id} | Participant ID: {pid}")
        validate_transcript_columns(self.transcript, ['speaker'])

        # Case 1: pid matches a current label in transcript
        if pid in self.transcript["speaker"].unique():
            self.metadata["participant_id"] = pid
            return

        # Case 2: pid matches an original name in speaker_mapping
        if hasattr(self, "speaker_mapping") and pid in self.speaker_mapping:
            self.metadata["participant_id"] = self.speaker_mapping[pid]
            return

        raise ValueError(
            f"Participant ID '{pid}' not found in transcript or speaker mapping."
        )

    def get_participant_id(self) -> str | None:
        """Return participant_id from metadata."""
        return self.metadata.get("participant_id")
    
    def validate_quote(self, quote: Quote, topic_name: Optional[str] = None) -> Optional[str]:
        """
        Validate a single Quote against the interview transcript.

        Parameters
        ----------
        quote : Quote
            The Quote object to validate.
        topic_name : str, optional
            The topic name for context (used only in error messages).

        Returns
        -------
        str | None
            Returns None if the quote is valid (exact or near match),
            otherwise returns a descriptive error message.
        """
        logger.debug(f"Validate quote - Interview ID: {self.id} | Quote text: {quote.quote}")
        df = self.transcript

        def similar(a: str, b: str, threshold: float = 0.8) -> bool:
            return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

        idx = quote.index
        quote_text = quote.quote.strip()

        if idx not in df.index:
            return f"❌ Quote index {idx} not found{f' in topic {topic_name!r}' if topic_name else ''}."

        statement_text = str(df.loc[idx, "statement"])

        if quote_text in statement_text:
            return None  # ✅ exact substring match
        elif similar(quote_text, statement_text):
            return None  # ✅ fuzzy near match
        else:
            return (
                f"⚠️ Mismatch{f' in topic {topic_name!r}' if topic_name else ''} at index {idx}:\n"
                f"  Quote: \"{quote_text[:80]}...\"\n"
                f"  Statement: \"{statement_text[:80]}...\""
            )

    def _validate_topic(self, topic, m: MelleaSession, interview_context: str) -> QIndex:
        """
        Validate a single topic based on quote validation, LLM quality check, and assessment.

        Parameters
        ----------
        topic : Topic
            The topic to validate
        m : MelleaSession
            Mellea session for LLM-as-judge validation
        interview_context : str
            Interview context for validation

        Returns
        -------
        QIndex
            Aggregated validation result for this topic
        """
        checks = []

        # Check 1: Quote validation
        quote_errors = []
        for quote in topic.quotes:
            err = self.validate_quote(quote, topic_name=topic.topic)
            if err:
                quote_errors.append(err)

        if not quote_errors:
            checks.append(QIndex.from_check(
                method="quote_validation",
                status="ok",
                explanation=f"All {len(topic.quotes)} quotes validated successfully"
            ))
        elif len(quote_errors) < len(topic.quotes):
            checks.append(QIndex.from_check(
                method="quote_validation",
                status="check",
                explanation=f"{len(quote_errors)}/{len(topic.quotes)} quotes failed validation",
                errors=quote_errors
            ))
        else:
            checks.append(QIndex.from_check(
                method="quote_validation",
                status="iffy",
                explanation=f"All {len(topic.quotes)} quotes failed validation",
                errors=quote_errors
            ))

        # Check 2: LLM validation
        logger.debug(f"  → Running LLM validation for topic: {topic.topic}")
        validation_prompt = """
        You extracted the topic "{{topic_name}}" from an interview.

        Topic: {{topic_name}}
        Explanation: {{explanation}}
        Supporting Quotes: {{quotes}}

        Interview Context: {{interview_context}}

        Given the information above, evaluate whether this topic is:
        1. Relevant to the interview context
        2. Well-supported by the quotes

        Rate the topic quality as: "excellent", "acceptable", or "poor"
        Provide a reason (1 sentence).
        """

        quotes_text = format_quotes_for_display(topic.quotes, max_length=100)

        try:
            llm_rating = str(m.instruct(
                validation_prompt,
                user_variables={
                    "topic_name": topic.topic,
                    "explanation": topic.explanation,
                    "quotes": quotes_text,
                    "interview_context": interview_context
                },
                requirements=[
                    "Answer must start with rating: 'excellent', 'acceptable', or 'poor'",
                    "Answer must include a reason (1 sentence)"
                ]
            )).strip()

            # Parse LLM rating using utility function
            llm_status, llm_explanation = parse_quality_rating(llm_rating)

            checks.append(QIndex.from_check(
                method="llm_validation",
                status=llm_status,
                explanation=llm_explanation
            ))

        except Exception as e:
            logger.warning(f"LLM quality validation failed for topic '{topic.topic}': {str(e)}")
            checks.append(QIndex.from_check(
                method="llm_validation",
                status="check",
                explanation=f"Quality validation failed - manual review required: {str(e)}"
            ))

        # Check 3: Informational LLM assessment (strengths/weaknesses)
        logger.debug(f"  → Running LLM assessment for topic: {topic.topic}")
        assessment_prompt = """
        You extracted the topic "{{topic_name}}" from an interview.

        Topic: {{topic_name}}
        Explanation: {{explanation}}
        Supporting Quotes: {{quotes}}

        Interview Context: {{interview_context}}

        Provide a brief assessment of this topic:
        1. Strengths: What makes this topic valuable or well-extracted? (1-2 sentences)
        2. Weaknesses: What are potential limitations or concerns? (1-2 sentences)

        Format your response as:
        Strengths: [your assessment]
        Weaknesses: [your assessment]
        """

        try:
            assessment = str(m.instruct(
                assessment_prompt,
                user_variables={
                    "topic_name": topic.topic,
                    "explanation": topic.explanation,
                    "quotes": quotes_text,
                    "interview_context": interview_context
                },
                requirements=[
                    "Response must have two sections: 'Strengths:' and 'Weaknesses:'",
                    "Each section should be 1-2 sentences",
                    "Be specific and constructive"
                ]
            )).strip()

            # Parse strengths and weaknesses
            strengths = ""
            weaknesses = ""
            if "Strengths:" in assessment and "Weaknesses:" in assessment:
                parts = assessment.split("Weaknesses:")
                strengths = parts[0].replace("Strengths:", "").strip()
                weaknesses = parts[1].strip() if len(parts) > 1 else ""

            # Store in metadata for informational check
            assessment_metadata = {
                "strengths": strengths,
                "weaknesses": weaknesses,
                "full_assessment": assessment
            }

            # Create informational check (doesn't affect validation status)
            checks.append(QIndex.from_check(
                method="llm_assessment",
                status="ok",  # Status is irrelevant since informational=True
                explanation=f"Strengths: {strengths[:60]}... | Weaknesses: {weaknesses[:60]}...",
                metadata=assessment_metadata,
                informational=True
            ))

        except Exception as e:
            logger.warning(f"LLM assessment failed for topic '{topic.topic}': {str(e)}")
            # Don't add a check if assessment fails - it's purely informational

        # Aggregate all checks (only non-informational checks affect status)
        return QIndex.from_checks(checks, aggregation="strictest")

    def _aggregate_topic_validations(self, topic_validations: list[QIndex]) -> QIndex:
        """
        Aggregate per-topic validations into overall validation.

        Parameters
        ----------
        topic_validations : list[QIndex]
            Per-topic validation results

        Returns
        -------
        QIndex
            Overall validation result
        """
        if topic_validations:
            return QIndex.from_checks(
                topic_validations,
                aggregation="consensus"
            )
        else:
            return QIndex.from_check(
                method="generation",
                status="iffy",
                explanation="No topics were generated"
            )

    def suggest_topics_top_down(self, m: MelleaSession, n: Optional[int] = None, explain: bool = True, interview_context: Optional[str] = "General") -> ValidatedList:
        """
        Suggest overarching topics for the interview using an LLM.

        Returns a ValidatedList containing topics with dual validation per topic:
        1. Quote validation (checks if quotes exist verbatim in transcript)
        2. LLM validation (validates the topic quality/relevance)

        Parameters
        ----------
        m : MelleaSession
            Active Mellea session for prompting.
        n : int, optional
            Desired number of topics to suggest. If None, let the model decide.
        explain : bool, optional, default=True
            If True, also request a short explanation for each theme.
        interview_context : str, optional, default="General"
            Provides context information about the interview to ground topic extraction.

        Returns
        -------
        ValidatedList[Topic]
            List of topics with per-topic and overall validation results.

        Raises
        ------
        ValueError
            If transcript is empty or missing participant_id in metadata
        """
        logger.debug(f"Starting suggest_topics_top_down for interview {self.id} (n={n}, context={interview_context})")
        df = self.transcript

        # Data precondition checks - use exceptions, not QIndex
        if df.empty:
            raise ValueError("Transcript is empty")

        if "participant_id" not in self.metadata:
            raise ValueError("Missing participant_id in metadata (required for thematic analysis)")

        validate_transcript_columns(self.transcript, ['timestamp', 'speaker', 'statement'])

        # Build prompt
        text = "\n".join(
            f"{index} [{row['timestamp']}] {row['speaker']}: {row['statement']}"
            for index, row in df.iterrows()
        )
        interviewee = self.metadata["participant_id"]
        num_req = f"exactly {n} unique, non-generic" if n else "as many as possible unique, non-generic"
        
        prompt = """
        You are given an interview transcript. Your task is to identify {{num_req}} topics in the statements made by the interviewee {{interviewee}}, provide a detailed explanation, and supporting quotes from the interviewee.

        Interview Transcript:
        {{text}} 
        """

        logger.debug("Calling Mellea for initial topic extraction...")

        requirements=[
                f"Each topic should be specific to the context of the overall interview: {interview_context}",
                "Each topic should be descriptive, 2 to 5 words long.",
                "Each topic must include a detailed explanation, why it was chosen. More than 1 sentence." if explain else "Explanations must be omitted. Use 'None'.",
                "Each topic must be supported by one or more quotes",
                "Each quote must include row number (index), timestamp, and speaker."]

        logger.debug(f"Requirements passed in: {requirements}")

        start_time = time.time()
        response = m.instruct(
            prompt,
            strategy=RejectionSamplingStrategy(loop_budget=1),
            user_variables={"interviewee": interviewee, "num_req": num_req, "text": text, "interview_context": interview_context},
            model_options={
                "max_tokens": 10000,
                "temperature": 0.0
            },
            requirements=requirements,
            format=TopicList,
            return_sampling_results=True,
        )
        elapsed_time = time.time() - start_time
        logger.debug(f"Mellea topic extraction completed in {elapsed_time:.2f} seconds")

        # Dump raw response at DEBUG level
        logger.debug(("*** Response"))
        logger.debug(response)

        # Print validation results
        if logger.isEnabledFor(logging.DEBUG): 
            print_mellea_validations(response, title="Topic Extraction Validations")

        try:
            topics = TopicList.model_validate_json(response._underlying_value)
            logger.debug(f"Successfully parsed {len(topics.topics)} topics from LLM response")

            # Populate TopicList metadata
            topics.generated_at = datetime.now().isoformat()

            # Populate interview_id on each Topic
            for topic in topics.topics:
                topic.interview_id = self.id

            # Validate each topic using extracted helper method
            topic_validations = []
            logger.debug("Starting per-topic validation...")

            for idx, topic in enumerate(topics.topics, start=1):
                logger.debug(f"Validating topic {idx}/{len(topics.topics)}: {topic.topic}")
                topic_validation = self._validate_topic(topic, m, interview_context)
                topic_validations.append(topic_validation)

            # Aggregate per-topic validations into overall validation
            logger.debug("Completed per-topic validation. Creating overall validation summary...")
            overall_validation = self._aggregate_topic_validations(topic_validations)

            # Create ValidatedList result
            validated_result = ValidatedList(
                result=topics.topics,
                validation=overall_validation,
                item_validations=topic_validations
            )

            # Cache the TopicList and validation results
            self.topics_top_down = topics
            self.topics_top_down_validation = overall_validation
            logger.debug(f"Cached TopicList in interview.topics_top_down")
            logger.debug(f"Cached validation (status={overall_validation.status}) in interview.topics_top_down_validation")

            # Print summary only when logger is set to DEBUG level
            if logger.isEnabledFor(logging.DEBUG):
                validated_result.print_summary(
                    title="Topic Validation Summary",
                    item_label="Topic"
                )

            return validated_result

        except Exception as e:
            # Parse error - return empty ValidatedList with iffy status
            return ValidatedList(
                result=[],
                validation=QIndex.from_check(
                    method="parsing",
                    status="iffy",
                    explanation=f"Could not parse LLM output: {str(e)}",
                    errors=[f"Raw response: {str(response)[:200]}..."]
                ),
                item_validations=[]
            )

    def save_state(self, cache_path: Optional[Path] = None) -> Path:
        """
        Save complete interview state to cache file.

        Saves transcript (raw + transformed), metadata, speaker_mapping, and topics.

        Parameters
        ----------
        cache_path : Path, optional
            Custom cache file path. If None, auto-generates in .qux360_cache/

        Returns
        -------
        Path
            Path to saved cache file
        """
        from .cache import save_interview_state
        return save_interview_state(self, cache_path)

    @classmethod
    def load_from_cache(cls, cache_path: Path) -> 'Interview':
        """
        Load interview from cache file.

        Parameters
        ----------
        cache_path : Path
            Path to the cache JSON file

        Returns
        -------
        Interview
            Reconstructed interview with full state
        """
        from .cache import load_interview_state
        return load_interview_state(cache_path)

    def get_topics_validated(self) -> Optional[ValidatedList]:
        """
        Reconstruct ValidatedList from cached topics and validation.

        This is useful after loading from cache to get the full validated
        result that can be used with print_summary() and other ValidatedList methods.

        Returns
        -------
        ValidatedList or None
            ValidatedList containing topics and validation, or None if no cached topics
        """
        logger.debug(f"Get topics validated - Interview ID: {self.id}")
        if self.topics_top_down is None:
            return None

        # Default validation if none cached
        validation = self.topics_top_down_validation or QIndex.from_check(
            method="cache",
            status="ok",
            explanation="Loaded from cache (no validation stored)"
        )

        # Extract per-topic validations from nested checks
        # The overall validation contains per-topic validations in its checks list
        item_validations = validation.checks if validation.checks else []

        # Reconstruct ValidatedList from cached components
        return ValidatedList(
            result=self.topics_top_down.topics,
            validation=validation,
            item_validations=item_validations
        )