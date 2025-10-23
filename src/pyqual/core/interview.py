import os
import difflib
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from textwrap import shorten
import uuid
import spacy

from mellea import MelleaSession
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pyqual.core.iffy import IffyIndex
from pyqual.core.validated import Validated, ValidatedList
import copy

from .models import TopicList, Quote
from ..io.docx_parser import parse_docx
from ..io.xlsx_parser import parse_xlsx
from ..io.csv_parser import parse_csv

logger = logging.getLogger(__name__)

class Interview:

    def __init__(self,
                 file: Optional[Union[str, Path]] = None,
                 metadata: Optional[dict] = None,
                 headers: Optional[dict] = None,
                 has_headers = True):
        """
        Initialize an Interview.
        
        - Always generates a unique UUID-based id.
        - If a file is provided, it is parsed into a transcript DataFrame.
        - Keeps both raw (immutable) and working (mutable) transcripts.
        """
        self.id = f"interview_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}

        self.transcript_raw = self._init_transcript(file, headers=headers, has_headers=has_headers)
        self.transcript = copy.deepcopy(self.transcript_raw)
        self.speaker_mapping = None


    def __repr__(self):
        return f"<Interview {self.id}, {len(self.transcript)} turns, metadata={self.metadata}>"
    
    def _empty_transcript(self):
        return pd.DataFrame(columns=[
            "timestamp", "speaker_id", "speaker", "statement", "codes", "themes"
        ])

    def _init_transcript(self, file, headers: Optional[dict] = None, has_headers = True):
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
        candidates = [model, "en_core_web_lg", "en_core_web_sm"]
        for candidate in candidates:
            try:
                nlp = spacy.load(candidate)
                print(f"Using spaCy model: {candidate}")
                return nlp
            except OSError:
                print(f"spaCy model '{candidate}' not found.")
        raise RuntimeError("No suitable spaCy model installed.")
    

    
    def reset_transcript(self):
        """Reset the working transcript back to the raw version."""
        self.transcript = copy.deepcopy(self.transcript_raw)

    def load_file(self, file: str | Path, headers: Optional[dict] = None, has_headers = True):
        """
        Load a transcript file into the interview (overwrites both raw and working).
        """
        self.transcript_raw = self._init_transcript(file, headers=headers, has_headers=has_headers)
        self.transcript = copy.deepcopy(self.transcript_raw)

    def add_code(self, row: int, code: str):
        """Attach a code to a specific row in the working transcript."""
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
            col_widths = {
                "timestamp": 10,
                "speaker_id": 10,
                "speaker": 25,
                "statement": 60,
                "codes": 20,
                "themes": 20,
            }

            # Apply formatting
            for idx, col in enumerate(df.columns):
                ws.set_column(idx, idx, col_widths.get(col, 20), wrap_top)

            # Freeze header row
            ws.freeze_panes(1, 0)

    def show(self, n: int = 5, speaker: Optional[str] = None, width: int = 60):
        """Pretty-print transcript preview in a clean 3-column layout."""
        df = self.transcript
        if df.empty:
            print("Transcript is empty.")
            return

        if speaker:
            df = df[df["speaker"] == speaker]

        for _, row in df.head(n).iterrows():
            ts = str(row["timestamp"])
            sp = str(row["speaker"])
            st = str(row["statement"])[:width]
            print(f"{ts:>8} | {sp:<20} | {st}")

    def get_speakers(self) -> list[str]:
        """Return a list of unique speakers in the working transcript."""
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

 
    def identify_interviewee(self, m: Optional[MelleaSession] = None) -> Validated[str | None]:
        """
        Identify the likely interviewee via heuristic and/or LLM.

        Returns a Validated result containing the predicted interviewee and validation assessment.

        Parameters
        ----------
        m : Optional[MelleaSession]
            If provided, uses LLM to validate heuristic prediction

        Returns
        -------
        Validated[str | None]
            Validated result with predicted interviewee and composite validation
            from heuristic and optional LLM checks

        Examples
        --------
        >>> result = interview.identify_interviewee(m)
        >>> print(result.validation)  # Overall validation
        >>> print(result.result)  # Predicted interviewee
        >>> if result.passed_validation():
        ...     interviewee = result.result
        """
        # Nothing to work on
        if self.transcript.empty or "speaker" not in self.transcript:
            return Validated(
                result=None,
                validation=IffyIndex.from_check(
                    method="data_check",
                    status="iffy",
                    explanation="Transcript is empty or missing 'speaker' column"
                )
            )

        # Check 1: Heuristic (word count)
        counts = self.transcript.groupby("speaker")["statement"].apply(
            lambda x: x.str.split().str.len().sum()
        )
        predicted_heuristic = counts.idxmax()
        total_words = counts.sum()
        ratio = counts[predicted_heuristic] / total_words if total_words > 0 else 0.0

        # Determine heuristic status
        if ratio >= 0.70:
            heuristic_status = "ok"
        elif ratio >= 0.60:
            heuristic_status = "check"
        else:
            heuristic_status = "iffy"

        heuristic_check = IffyIndex.from_check(
            method="heuristic",
            status=heuristic_status,
            explanation=f"{predicted_heuristic} has {ratio:.0%} of words",
            metadata={"word_ratio": ratio, "prediction": predicted_heuristic}
        )

        checks = [heuristic_check]

        # Check 2: Optional LLM prediction
        if m:
            snippet = "\n".join(
                f"[{row['timestamp']}] {row['speaker']}: {shorten(str(row['statement']), width=120)}"
                for _, row in self.transcript.head(25).iterrows()
            )
            predicted_llm = str(m.instruct(
                """
                You are given an interview transcript snippet with multiple speakers.
                One or more speakers are the interviewers (asking questions).
                One speaker is the interviewee (giving longer answers).
                Based on the transcript snippet, identify the interviewee by ID (from the speaker column).

                Transcript Snippet:
                {{snippet}}

                Question: Who is the interviewee?
                """,
                requirements=[
                    "The answer should ONLY contain the speaker ID exactly as shown in the speaker column",
                    "There should be no explanation"
                ],
                strategy=RejectionSamplingStrategy(loop_budget=2),
                user_variables={"snippet": snippet}
            )).strip()

            # Evaluate LLM check based on agreement and heuristic strength
            if predicted_llm == predicted_heuristic:
                llm_status = "ok"
                llm_explanation = f"LLM agreed: {predicted_llm}"
            else:
                # Disagreement - but strong heuristic can override
                if ratio >= 0.80:
                    llm_status = "ok"
                    llm_explanation = f"LLM disagreed (suggested {predicted_llm}), but heuristic is strong ({ratio:.0%})"
                elif ratio >= 0.70:
                    llm_status = "check"
                    llm_explanation = f"LLM disagreed (suggested {predicted_llm}), moderate confidence ({ratio:.0%})"
                else:
                    llm_status = "iffy"
                    llm_explanation = f"LLM disagreed (suggested {predicted_llm}), low confidence ({ratio:.0%})"

            llm_check = IffyIndex.from_check(
                method="llm",
                status=llm_status,
                explanation=llm_explanation,
                metadata={"prediction": predicted_llm, "agreement": predicted_llm == predicted_heuristic}
            )
            checks.append(llm_check)

        # Aggregate validation checks
        overall_validation = IffyIndex.from_checks(checks, aggregation="consensus")

        # Metadata update only if validation passed
        if overall_validation.status in ("ok", "check"):
            self.metadata["participant_id"] = predicted_heuristic

        return Validated(
            result=predicted_heuristic,
            validation=overall_validation
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
        nlp = self._load_spacy_model(model)

        results = []
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
        
        replacements = {}
        counters = {"PERSON": 0, "ORG": 0, "GPE": 0}

        for e in entities:
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
        speakers = self.get_speakers()
        if not speakers:
            return {}

        mapping = {sp: f"Speaker{i+1}" for i, sp in enumerate(speakers)}

        self.transcript["speaker"] = self.transcript["speaker"].map(mapping)
        self.speaker_mapping = mapping

        # Update participant_id in metadata if already set
        pid = self.metadata.get("participant_id")
        if pid and pid in mapping:
            self.metadata["participant_id"] = mapping[pid]

        return mapping

    def anonymize_statements(self, replacements: dict):

        if not replacements:
            print("No replacements provided.")
            return

        def replace_text(text: str) -> str:
            anonymized = str(text)
            for original, anon in replacements.items():
                anonymized = anonymized.replace(original, anon)
            return anonymized

        # Replace inside statements
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
        if "speaker" not in self.transcript.columns:
            raise ValueError("Transcript has no 'speaker' column.")

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
        """
        df = self.transcript

        # Edge case 1: Empty transcript
        if df.empty:
            return ValidatedList(
                result=[],
                validation=IffyIndex.from_check(
                    method="data_check",
                    status="iffy",
                    explanation="Transcript is empty"
                ),
                item_validations=[]
            )

        # Edge case 2: Missing participant_id
        if "participant_id" not in self.metadata:
            return ValidatedList(
                result=[],
                validation=IffyIndex.from_check(
                    method="data_check",
                    status="iffy",
                    explanation="Missing participant_id in metadata (required for thematic analysis)"
                ),
                item_validations=[]
            )


        # Build prompt
        text = "\n".join(
            f"{index} [{row['timestamp']}] {row['speaker']}: {row['statement']}"
            for index, row in df.iterrows()
        )
        interviewee = self.metadata["participant_id"]
        num_req = f"exactly {n} unique, non-generic" if n else "as many as possible unique, non-generic"
        
        prompt = """
        You are given an interview transcript. Your task is to identify {{num_req}} topics in the statements made by the interviewee {{interviewee}}, provide a separate explanation, and separate supporting quotes from the interviewee.

        Interview Transcript:
        {{text}} 
        """

        logger.info("Calling Mellea...")
        
        requirements=[ 
                f"Each topic should be specific to the context of the overall interview: {interview_context}",
                "Each topic should be 2–5 words, concise, concrete, but nuanced.",
                "Each topic must include a detailed explanation, why it was chosen. More than 1 sentence." if explain else "Explanations must be omitted. Use 'None'.",
                "Each quote must include row number (index), timestamp, and speaker."]
               
        print(requirements)

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
        
        if logger.isEnabledFor(logging.INFO):
            print(("*** Response"))
            
            print(response)
            print(("**** Validations"))
            for i, validation_group in enumerate(response.sample_validations, start=1):
                print(f"\n--- Validation Group {i} ---")
                for req, res in validation_group:
                    print(f"Requirement: {req.description or '(no description)'}")
                    print(f".  Result: {res._result}")
                    if res.score is not None:
                        print(f"  🔢 Score: {res.score:.2f}")
                    if res.reason:
                        print(f".  Reason: {res.reason}")
                    if req.check_only:
                        print(f"  ⚙️  (Check-only requirement)")
                    print("-" * 40)

        try:
            topics = TopicList.model_validate_json(response._underlying_value)

            # NEW: Dual validation per topic
            topic_validations = []

            for topic in topics.topics:
                # Validation Check 1: Quote validation
                quote_errors = []
                for quote in topic.quotes:
                    err = self.validate_quote(quote, topic_name=topic.topic)
                    if err:
                        quote_errors.append(err)

                if not quote_errors:
                    quote_check = IffyIndex.from_check(
                        method="quote_validation",
                        status="ok",
                        explanation=f"All {len(topic.quotes)} quotes validated successfully"
                    )
                elif len(quote_errors) < len(topic.quotes):
                    quote_check = IffyIndex.from_check(
                        method="quote_validation",
                        status="check",
                        explanation=f"{len(quote_errors)}/{len(topic.quotes)} quotes failed validation",
                        errors=quote_errors
                    )
                else:
                    quote_check = IffyIndex.from_check(
                        method="quote_validation",
                        status="iffy",
                        explanation=f"All {len(topic.quotes)} quotes failed validation",
                        errors=quote_errors
                    )

                # Validation Check 2: LLM validation (NEW)
                # Ask the LLM to validate the topic quality/relevance
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
                Provide a brief reason (1 sentence).
                """

                quotes_text = "\n".join(
                    f"- [{q.index}] {q.quote[:100]}..."
                    for q in topic.quotes
                )

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
                        "Follow the rating with a brief reason (1 sentence)"
                    ]
                )).strip().lower()

                # Parse LLM rating
                if llm_rating.startswith("excellent"):
                    llm_status = "ok"
                    llm_explanation = llm_rating
                elif llm_rating.startswith("acceptable"):
                    llm_status = "check"
                    llm_explanation = llm_rating
                else:  # "poor" or unparseable
                    llm_status = "iffy"
                    llm_explanation = llm_rating if llm_rating.startswith("poor") else f"Unexpected LLM response: {llm_rating}"

                llm_check = IffyIndex.from_check(
                    method="llm_validation",
                    status=llm_status,
                    explanation=llm_explanation
                )

                # Validation Check 3: Informational LLM assessment (strengths/weaknesses)
                # This check does NOT affect the validation status
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
                info_check = IffyIndex.from_check(
                    method="llm_assessment",
                    status="ok",  # Status is irrelevant since informational=True
                    explanation=f"Strengths: {strengths[:60]}... | Weaknesses: {weaknesses[:60]}...",
                    metadata=assessment_metadata,
                    informational=True
                )

                # Combine all three checks for this topic
                # Only quote_check and llm_check affect the validation status
                # info_check is included but won't impact the aggregated status
                topic_validation = IffyIndex.from_checks(
                    [quote_check, llm_check, info_check],
                    aggregation="strictest"  # Only non-informational checks affect status
                )
                topic_validations.append(topic_validation)

            # Create overall validation from all topic validations
            if topic_validations:
                overall_validation = IffyIndex.from_checks(
                    topic_validations,
                    aggregation="consensus"  # Overall assessment across all topics
                )
            else:
                overall_validation = IffyIndex.from_check(
                    method="generation",
                    status="iffy",
                    explanation="No topics were generated"
                )

            # Create ValidatedList result
            validated_result = ValidatedList(
                result=topics.topics,
                validation=overall_validation,
                item_validations=topic_validations
            )

            # Print summary only when logger is set to INFO level
            if logger.isEnabledFor(logging.INFO):
                validated_result.print_summary(
                    title="Topic Validation Summary",
                    item_label="Topic"
                )

            return validated_result

        except Exception as e:
            # Parse error - return empty ValidatedList with iffy status
            return ValidatedList(
                result=[],
                validation=IffyIndex.from_check(
                    method="parsing",
                    status="iffy",
                    explanation=f"Could not parse LLM output: {str(e)}",
                    errors=[f"Raw response: {str(response)[:200]}..."]
                ),
                item_validations=[]
            )