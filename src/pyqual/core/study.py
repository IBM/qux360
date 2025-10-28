import uuid
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .interview import Interview
from .models import ThemeList, TopicList, CoherenceAssessment
from .validated import Validated, ValidatedList
from .iffy import IffyIndex
from .utils import print_mellea_validations
from mellea import MelleaSession
from mellea.stdlib.sampling import RejectionSamplingStrategy

logger = logging.getLogger(__name__)

class Study:
    """
    A collection of qualitative documents (for now only interviews are supported).
    """

    def __init__(self, files_or_docs=None, metadata=None, doc_cls=Interview, headers: Optional[list[dict]] = None, has_headers: Optional[list[bool]] = None, study_context=None, use_cache: bool = True, cache_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        files : list[str | Path | Interview], optional
            Either a list of file paths OR a list of Interview objects.
        metadata : dict, optional
            Metadata to attach at corpus level.
        doc_cls : class, default=Interview
            Currently must be Interview
        headers: list of dict, optional
            Headers of columns for timestamp, speaker and statements in the documents
        has_headers: list of bool, optional
            Indicates if each file in files has headers or not. The matching with files is positional.
        study_context : str, optional
            Description of the overall study context (e.g., "Remote work experiences").
            Used as default for theme extraction if not overridden.
        use_cache : bool, default=True
            If True, attempts to load interviews from cache before parsing
        cache_dir : Path, optional
            Custom cache directory for interview states
        """
        logger.debug(f"Init Study")
        if doc_cls is not Interview:
            raise ValueError("Study currently only supports Interview documents.")

        self.id = f"study_{uuid.uuid4().hex[:8]}"
        self.doc_cls = doc_cls
        self.metadata = metadata or {}
        self.documents = []
        self.study_context = study_context
        self.themes_top_down = None
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        logger.debug(f"Study ID: {self.id}")

        if files_or_docs:
            num_items = len(files_or_docs)
            if headers and len(headers) != num_items:
                raise ValueError(f"'headers' length ({len(headers)}) does not match number of documents ({num_items}).")

            if has_headers and len(has_headers) != num_items:
                raise ValueError(f"'has_headers' length ({len(has_headers)}) does not match number of documents ({num_items}).")
            
            for i, item in enumerate(files_or_docs):
                logger.debug(f"Processing item {item}")
                if headers:
                    self._add_checked(file_or_doc=item, headers=headers[i], has_headers=has_headers[i])
                elif has_headers:
                    self._add_checked(file_or_doc=item, has_headers=has_headers[i])
                else:
                    self._add_checked(file_or_doc=item)

    def _add_checked(self, file_or_doc, headers: Optional[dict] = None, has_headers: Optional[bool]= True):
        logger.debug(f"Add document {file_or_doc}| Study ID: {self.id}")
        if isinstance(file_or_doc, (str, Path)):
            self.documents.append(
                self.doc_cls(
                    file=file_or_doc,
                    metadata=self.metadata,
                    headers=headers,
                    has_headers=has_headers,
                    use_cache=self.use_cache,
                    cache_dir=self.cache_dir
                )
            )
        elif isinstance(file_or_doc, self.doc_cls):
            self.documents.append(file_or_doc)
        else:
            raise TypeError(
                f"Invalid type {type(file_or_doc)}. "
                f"Corpus only supports {self.doc_cls.__name__} objects or file paths."
            )
        
    def add(self, file_or_doc, headers: Optional[dict] = None, has_headers: Optional[bool]= True):
        """Add an Interview object or a file path."""
        logger.debug(f"Add document {file_or_doc} | Study ID: {self.id}")
        self._add_checked(file_or_doc=file_or_doc, headers=headers, has_headers=has_headers)


    def get_interview_by_id(self, interview_id: str):
        for doc in self.documents:
            if doc.id == interview_id:
                return doc
        return None

    def get_interviews_by_participant(self, participant_id: str):
        return [
            doc for doc in self.documents
            if doc.metadata.get("participant_id") == participant_id
        ]

    def identify_interviewees(self, m=None):
        """
        Identify the likely interviewee for each interview in the study.
        Results are stored in interview.metadata['participant_id'].

        Parameters
        ----------
        m : Optional[MelleaSession]
            If provided, passed through to Interview.identify_interviewee().
        """
        logger.debug(f"Identify interviewees for study {self.id}")
        results = {}
        for doc in self.documents:
            predicted = doc.identify_interviewee(m=m)
            results[doc.id] = predicted

        return results

    def suggest_topics_all(
        self,
        m: MelleaSession,
        n: Optional[int] = None,
        interview_context: Optional[str] = None
    ):
        """
        Extract topics from all interviews in the study.

        This is a convenience method that calls suggest_topics_top_down() on each
        interview, using the study's context as the default interview context.

        Parameters
        ----------
        m : MelleaSession
            Active Mellea session for prompting
        n : int, optional
            Desired number of topics per interview
        interview_context : str, optional
            Context for topic extraction. If None, uses study.study_context.

        Returns
        -------
        dict
            Mapping of interview.id -> ValidatedList[Topic]
        """
        logger.debug(f"Suggest topics for study {self.id}")
        context = interview_context or self.study_context
        logger.debug(f"Extracting topics from {len(self.documents)} interviews with context: {context}")

        results = {}
        for idx, doc in enumerate(self.documents, start=1):
            logger.debug(f"Processing interview {idx}/{len(self.documents)}: {doc.id}")
            topics_result = doc.suggest_topics_top_down(
                m,
                n=n,
                interview_context=context
            )
            results[doc.id] = topics_result

        return results

    def anonymize_speakers(self):
        """
        Anonymize speakers for all interviews in the study.
        Calls Interview.anonymize_speakers_generic() on each interview.

        Returns
        -------
        dict
            Mapping of interview.id -> {original_speaker: anonymized_speaker}
        """
        logger.debug(f"Anonymize speakers - Study ID: {self.id}")
        all_mappings = {}

        for doc in self.documents:
            mapping = doc.anonymize_speakers_generic()
            all_mappings[doc.id] = mapping

        return all_mappings

    def __len__(self):
        return len(self.documents)

    def __iter__(self):
        return iter(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]
    
    def __repr__(self):
        context_str = f", context='{self.study_context}'" if self.study_context else ""
        return (
            f"<Study {self.id}: {len(self)} {self.doc_cls.__name__}(s){context_str}, "
            f"metadata={self.metadata}>"
        )

    def _aggregate_theme_validations(self, theme_validations: List[IffyIndex]) -> IffyIndex:
        """
        Aggregate per-theme validations using consensus with escalation.

        Uses consensus aggregation but escalates to "check" if ANY theme has issues.
        This ensures users are flagged to review even if majority are OK.

        Parameters
        ----------
        theme_validations : List[IffyIndex]
            Per-theme validation results

        Returns
        -------
        IffyIndex
            Overall validation with detailed status counts in explanation
        """
        if not theme_validations:
            return IffyIndex.from_check(
                method="generation",
                status="iffy",
                explanation="No themes were generated"
            )

        # Get consensus result
        consensus_result = IffyIndex.from_checks(
            theme_validations,
            aggregation="consensus"
        )

        # Count statuses for detailed explanation
        status_counts = {"ok": 0, "check": 0, "iffy": 0}
        for val in theme_validations:
            status_counts[val.status] += 1

        # Check if any themes have issues
        has_issues = status_counts["check"] > 0 or status_counts["iffy"] > 0

        # Escalate to "check" if consensus says "ok" but there are issues
        # This ensures users are flagged to review even if majority are OK
        if has_issues and consensus_result.status == "ok":
            final_status = "check"
            explanation = (
                f"{status_counts['ok']}/{len(theme_validations)} themes OK, "
                f"but {status_counts['check']} need review"
            )
            if status_counts['iffy'] > 0:
                explanation += f" and {status_counts['iffy']} have issues"
        else:
            final_status = consensus_result.status
            # Build detailed explanation with counts
            explanation = f"{status_counts['ok']}/{len(theme_validations)} themes validated successfully"
            if status_counts['check'] > 0:
                explanation += f", {status_counts['check']} need review"
            if status_counts['iffy'] > 0:
                explanation += f", {status_counts['iffy']} have issues"

        return IffyIndex.from_check(
            method="theme_validation_summary",
            status=final_status,
            explanation=explanation
        )

    def _validate_theme(self, theme, hydration_result: dict, m: MelleaSession, study_context: str) -> IffyIndex:
        """
        Validate a single theme based on hydration, coverage, and LLM assessment.

        Also sets theme.prospective flag based on coverage and topic count.

        Parameters
        ----------
        theme : Theme
            The theme to validate
        hydration_result : dict
            Hydration results with keys: "total", "hydrated", "failed"
        m : MelleaSession
            Mellea session for LLM-as-judge validation
        study_context : str
            Study context for validation

        Returns
        -------
        IffyIndex
            Aggregated validation result for this theme
        """
        checks = []

        # Check 1: Topic hydration success
        if hydration_result["failed"]:
            checks.append(IffyIndex.from_check(
                method="topic_hydration",
                status="check",
                explanation=f"{len(hydration_result['failed'])}/{hydration_result['total']} topics failed to hydrate: {', '.join(hydration_result['failed'])}"
            ))
        else:
            checks.append(IffyIndex.from_check(
                method="topic_hydration",
                status="ok",
                explanation=f"All {hydration_result['total']} topics successfully hydrated"
            ))

        # Check 2: Cross-interview coverage
        interview_ids = set(t.interview_id for t in theme.topics if t.interview_id)
        if len(interview_ids) >= 2:
            checks.append(IffyIndex.from_check(
                method="cross_interview_coverage",
                status="ok",
                explanation=f"Theme spans {len(interview_ids)} interviews"
            ))
        else:
            checks.append(IffyIndex.from_check(
                method="cross_interview_coverage",
                status="iffy",
                explanation=f"Theme only spans {len(interview_ids)} interview(s)"
            ))

        # Check 3: Minimum topic count
        if len(theme.topics) >= 2:
            checks.append(IffyIndex.from_check(
                method="topic_count",
                status="ok",
                explanation=f"Theme has {len(theme.topics)} supporting topics"
            ))
        else:
            checks.append(IffyIndex.from_check(
                method="topic_count",
                status="check",
                explanation=f"Theme has only {len(theme.topics)} supporting topic(s)"
            ))

        # Derive prospective flag from validation checks
        # Prospective = < 2 interviews OR < 2 topics
        theme.prospective = (len(interview_ids) < 2 or len(theme.topics) < 2)

        # Check 4: LLM-as-judge coherence assessment
        logger.info(f"  → Running LLM coherence check for theme: {theme.title}")

        # Build topics summary for LLM
        topics_summary = []
        for topic in theme.topics:
            interview_label = topic.interview_id or "unknown"
            topics_summary.append(f"- {topic.topic} (from {interview_label})")
            topics_summary.append(f"  {topic.explanation}")
        topics_text = "\n".join(topics_summary)

        coherence_prompt = """
        You identified the theme "{{theme_title}}" in a qualitative study.

        Theme: {{theme_title}}
        Description: {{description}}
        Explanation: {{explanation}}

        Supporting Topics:
        {{topics_summary}}

        Study Context: {{study_context}}

        Evaluate the COHERENCE of this theme: Do the supporting topics genuinely relate to each other and to the theme?

        Consider:
        - Do the topics share a common conceptual thread?
        - Are any topics tangential or weakly related?
        - Does the theme accurately capture what unifies these topics?

        Rate the coherence:
        - **Strong**: Topics clearly relate to each other and to the theme with a tight conceptual fit
        - **Acceptable**: Topics generally relate but some connections may be loose or require interpretation
        - **Weak**: Topics appear disconnected or the theme doesn't accurately capture their commonality
        """

        try:
            assessment = m.instruct(
                coherence_prompt,
                user_variables={
                    "theme_title": theme.title,
                    "description": theme.description,
                    "explanation": theme.explanation,
                    "topics_summary": topics_text,
                    "study_context": study_context
                },
                format=CoherenceAssessment,
                strategy=RejectionSamplingStrategy(loop_budget=2),
                return_sampling_results=True
            )

            # Print Mellea validations if available
            print_mellea_validations(assessment, title=f"Coherence Check Validations for '{theme.title}'")

            # Extract the validated CoherenceAssessment from the SamplingResult
            coherence_result = CoherenceAssessment.model_validate_json(assessment._underlying_value)

            # Map rating to IffyIndex status
            rating_lower = coherence_result.rating.lower()
            if rating_lower == 'strong':
                status = 'ok'
                status_explanation = f"Strong coherence: {coherence_result.explanation}"
            elif rating_lower == 'acceptable':
                status = 'check'
                status_explanation = f"Acceptable coherence (review recommended): {coherence_result.explanation}"
            elif rating_lower == 'weak':
                status = 'iffy'
                status_explanation = f"Weak coherence: {coherence_result.explanation}"
            else:
                # Shouldn't happen with Literal type, but just in case
                status = 'check'
                status_explanation = f"Coherence assessment unclear: {coherence_result.explanation}"
                logger.warning(f"Unexpected coherence rating for theme '{theme.title}': {coherence_result.rating}")

            # Create required coherence check
            coherence_check = IffyIndex.from_check(
                method="llm_coherence",
                status=status,
                explanation=status_explanation
            )
            checks.append(coherence_check)

        except Exception as e:
            logger.warning(f"LLM coherence assessment failed for theme '{theme.title}': {str(e)}")
            # If LLM fails, add a "check" status (requires manual review)
            checks.append(IffyIndex.from_check(
                method="llm_coherence",
                status="check",
                explanation=f"Coherence assessment failed - manual review required: {str(e)}"
            ))

        # Aggregate all checks (only non-informational checks affect status)
        return IffyIndex.from_checks(checks, aggregation="strictest")

    def _hydrate_theme_topics(
        self,
        theme_list: ThemeList,
        topic_lists: List[TopicList]
    ) -> dict:
        """
        Replace LLM-generated topics in themes with original Topic objects.

        This ensures that topic explanations, quotes, and metadata are preserved
        exactly as they were in the original topic extraction, rather than relying
        on the LLM to faithfully copy all fields.

        Parameters
        ----------
        theme_list : ThemeList
            Theme list with LLM-generated topics to hydrate
        topic_lists : List[TopicList]
            Original topic lists to use as the source of truth

        Returns
        -------
        dict
            Hydration results mapping theme.title -> {"total": int, "hydrated": int, "failed": list}
        """
        # Build lookup table: (interview_id, topic_title) -> original Topic
        topic_lookup = {}
        for topic_list in topic_lists:
            for topic in topic_list.topics:
                key = (topic.interview_id, topic.topic)
                topic_lookup[key] = topic

        logger.info(f"Built topic lookup with {len(topic_lookup)} entries")

        # Track hydration results per theme
        hydration_results = {}

        # Hydrate themes: Replace LLM-generated topics with original Topic objects
        for theme in theme_list.themes:
            hydrated_topics = []
            failed_topics = []

            for llm_topic in theme.topics:
                key = (llm_topic.interview_id, llm_topic.topic)
                original_topic = topic_lookup.get(key)

                if original_topic:
                    # Use the original topic (preserves full explanation and quotes)
                    hydrated_topics.append(original_topic)
                else:
                    # Fallback: Keep LLM version if we can't find original
                    # This might happen if LLM slightly changed the topic title
                    hydrated_topics.append(llm_topic)
                    failed_topics.append(f"{llm_topic.topic} ({llm_topic.interview_id})")
                    logger.warning(f"Could not find original topic for '{llm_topic.topic}' "
                                 f"from {llm_topic.interview_id}. Using LLM version.")

            theme.topics = hydrated_topics
            hydration_results[theme.title] = {
                "total": len(theme.topics),
                "hydrated": len(theme.topics) - len(failed_topics),
                "failed": failed_topics
            }

        logger.info("Completed topic hydration for all themes")
        return hydration_results

    def _build_topics_text(
        self,
        topic_lists: List[TopicList],
        max_quotes_per_topic: Optional[int] = None,
        max_quote_length: Optional[int] = None
    ) -> tuple[str, int, int]:
        """
        Build a text representation of topics from multiple interviews.

        Parameters
        ----------
        topic_lists : List[TopicList]
            Topic lists from interviews to format
        max_quotes_per_topic : int, optional
            Maximum number of quotes to include per topic
        max_quote_length : int, optional
            Maximum character length for each quote

        Returns
        -------
        tuple[str, int, int]
            (topics_text, total_quotes_truncated, total_quote_length_truncated)
        """
        topics_text_parts = []
        total_quotes_truncated = 0
        total_quote_length_truncated = 0

        for topic_list in topic_lists:
            # Get interview_id from first topic in the list (they should all be the same)
            current_interview_id = topic_list.topics[0].interview_id if topic_list.topics else "unknown"
            topics_text_parts.append(f"\n=== Interview: {current_interview_id} ===")

            for topic in topic_list.topics:
                interview_id = topic.interview_id or "unknown"
                topics_text_parts.append(f"\nTopic: {topic.topic}")
                topics_text_parts.append(f"Interview: {interview_id}")
                topics_text_parts.append(f"Explanation: {topic.explanation}")

                # Determine how many quotes to include
                quotes_to_include = topic.quotes
                if max_quotes_per_topic is not None and len(topic.quotes) > max_quotes_per_topic:
                    quotes_to_include = topic.quotes[:max_quotes_per_topic]
                    total_quotes_truncated += len(topic.quotes) - max_quotes_per_topic

                topics_text_parts.append(f"Quotes ({len(quotes_to_include)} of {len(topic.quotes)}):")

                for quote in quotes_to_include:
                    # Truncate quote text if needed
                    quote_text = quote.quote
                    if max_quote_length is not None and len(quote_text) > max_quote_length:
                        quote_text = quote_text[:max_quote_length] + "..."
                        total_quote_length_truncated += 1

                    topics_text_parts.append(f"  - [{quote.index}] {quote.timestamp} {quote.speaker}: {quote_text}")

        topics_text = "\n".join(topics_text_parts)
        return topics_text, total_quotes_truncated, total_quote_length_truncated

    def suggest_themes(
        self,
        m: MelleaSession,
        n: Optional[int] = None,
        study_context: Optional[str] = None,
        topic_lists: Optional[List[TopicList]] = None,
        max_quotes_per_topic: Optional[int] = None,
        max_quote_length: Optional[int] = None
    ) -> Validated[ThemeList] | None:
        """
        Suggest themes/patterns across all interviews in the study.

        Themes represent cross-cutting patterns that emerge from topics extracted
        from multiple interviews. Each theme includes supporting topics with their
        associated quotes.

        Parameters
        ----------
        m : MelleaSession
            Active Mellea session for LLM prompting
        n : int, optional
            Number of themes to suggest. If None, LLM decides how many themes to generate.
        study_context : str, optional
            Description of the overall study context to ground theme extraction
        topic_lists : List[TopicList], optional
            If provided, use these TopicLists instead of cached interview.topics_top_down.
            Allows users to maintain multiple versions externally.
        max_quotes_per_topic : int, optional
            Maximum number of quotes to include per topic in the prompt.
            If None, includes all quotes. Use this to control prompt length.
        max_quote_length : int, optional
            Maximum character length for each quote in the prompt.
            If None, includes full quotes. Longer quotes will be truncated with "...".

        Returns
        -------
        Validated[ThemeList] or None
            ThemeList with cross-cutting themes, or None on failure

        Raises
        ------
        ValueError
            If study has fewer than 2 interviews
        ValueError
            If any interview lacks topics and topic_lists is not provided
        ValueError
            If fewer than 2 topic lists are available after filtering empty ones
        ValueError
            If n < 1, max_quotes_per_topic < 1, or max_quote_length < 50
        """
        logger.debug(f"Starting suggest_themes for study {self.id} (n={n}, context={study_context})")

        # Precondition 1: Study must have at least 2 interviews
        if len(self.documents) < 2:
            raise ValueError(
                "suggest_themes() requires at least 2 interviews to identify cross-cutting patterns. "
                f"Current study has {len(self.documents)} interview(s)."
            )

        # Precondition 2: Validate parameters
        if n is not None and n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if max_quotes_per_topic is not None and max_quotes_per_topic < 1:
            raise ValueError(f"max_quotes_per_topic must be at least 1, got {max_quotes_per_topic}")
        if max_quote_length is not None and max_quote_length < 50:
            raise ValueError(f"max_quote_length must be at least 50 characters, got {max_quote_length}")

        # Collect all topics from interviews
        all_topic_lists: List[TopicList] = []

        if topic_lists:
            # User-provided topic lists
            logger.debug(f"Using {len(topic_lists)} user-provided TopicLists")
            all_topic_lists = topic_lists

            # Validate that we have at least 2 topic lists
            if len(all_topic_lists) < 2:
                raise ValueError(
                    "suggest_themes() requires at least 2 TopicLists to identify cross-cutting patterns. "
                    f"Provided {len(all_topic_lists)} TopicList(s)."
                )
        else:
            # Use cached topics from interviews
            logger.debug(f"Collecting topics from {len(self.documents)} interviews")
            for interview in self.documents:
                if interview.topics_top_down is None:
                    raise ValueError(
                        f"Interview {interview.id} has no cached topics. "
                        "Run suggest_topics_top_down() first or provide topic_lists parameter."
                    )
                all_topic_lists.append(interview.topics_top_down)
                logger.debug(f"  → Collected {len(interview.topics_top_down.topics)} topics from {interview.id}")

        # Precondition 3: Check for empty topic lists and warn
        empty_topic_lists = [tl for tl in all_topic_lists if not tl.topics]
        if empty_topic_lists:
            logger.warning(
                f"{len(empty_topic_lists)} interview(s) have no topics. "
                "Theme extraction will only use interviews with topics."
            )
            # Filter out empty topic lists
            all_topic_lists = [tl for tl in all_topic_lists if tl.topics]

            # After filtering, ensure we still have at least 2
            if len(all_topic_lists) < 2:
                raise ValueError(
                    "After filtering empty topic lists, fewer than 2 interviews have topics. "
                    f"Cannot extract cross-cutting themes from {len(all_topic_lists)} interview(s)."
                )

        # Precondition 4: Warn if total topics is very low
        total_topics = sum(len(tl.topics) for tl in all_topic_lists)
        if total_topics < 5:
            logger.warning(
                f"Only {total_topics} topics across {len(all_topic_lists)} interviews. "
                "Theme extraction may not produce meaningful cross-cutting patterns."
            )

        # Build a comprehensive text representation of all topics
        topics_text, total_quotes_truncated, total_quote_length_truncated = self._build_topics_text(
            all_topic_lists,
            max_quotes_per_topic=max_quotes_per_topic,
            max_quote_length=max_quote_length
        )

        # Log the complete topics_text for debugging (DEBUG level to avoid cluttering output)
        logger.debug(f"Topics text for theme extraction:\n{topics_text}")

        # Estimate tokens (rough heuristic: 1 token ≈ 4 characters)
        estimated_tokens = len(topics_text) // 4
        logger.debug(f"Topics summary: ~{estimated_tokens} tokens (estimated)")

        # Warn if approaching common model limits
        if estimated_tokens > 8000:
            logger.warning(
                f"Topics summary is very large (~{estimated_tokens} tokens). "
                "Consider using max_quotes_per_topic and max_quote_length to reduce prompt size."
            )
        elif estimated_tokens > 4000:
            logger.warning(
                f"Topics summary is large (~{estimated_tokens} tokens). "
                "Monitor for potential context length issues."
            )

        # Log truncation warnings
        if total_quotes_truncated > 0:
            logger.warning(
                f"Truncated {total_quotes_truncated} quotes across topics "
                f"(max_quotes_per_topic={max_quotes_per_topic})"
            )
        if total_quote_length_truncated > 0:
            logger.warning(
                f"Truncated {total_quote_length_truncated} quote texts "
                f"(max_quote_length={max_quote_length})"
            )

        # Build prompt for LLM
        # Use provided context, fall back to Study's context, or use generic default
        context_str = study_context or self.study_context or "General qualitative study"

        # Handle n parameter (unlimited if None)
        if n is not None:
            n_instruction = f"{n} cross-cutting, recurring themes"
            n_requirement = f"Generate exactly {n} recurring themes"
        else:
            n_instruction = "all cross-cutting, recurring themes that emerge from the data"
            n_requirement = "Generate all themes that represent significant, recurring patterns across interviews. Focus on quality over quantity - include only themes with strong evidence."

        prompt = """
        You are analyzing topics extracted from multiple interviews in a qualitative study.
        Your task is to identify """ + n_instruction + """ that emerge across topics from these interviews. 
        Make use of all the topics provided. Themes should be supported by multiple topics. You MAY include themes 
        from a single interview if they show strong potential.

        Study Context: {{context}}

        Topics from Interviews:
        {{topics_text}}
        """
        #         For each theme, provide:
        # 1. A descriptive title (2-5 words)
        # 2. A description explaining what the theme represents
        # 3. An explanation of why this theme was chosen and how it manifests across interviews
        # 4. The specific topics (with their quotes) that support this theme

        logger.debug(f"Calling Mellea for theme extraction (n={'unlimited' if n is None else n})...")

        requirements = [
            f"Each theme should be specific to the context of the overall study: {context_str}",
            "Each theme must include a description explaining what the theme represents",
            "Each theme title should be descriptive, 2-5 words",
            "Each theme must include a detailed explanation of why it was chosen (more than 1 sentence) and how it manifests across interviews.",
            "Each theme must reference the specific topics that support it",
            "For each topic in a theme, you MUST include the exact 'topic' title and 'interview_id' as shown in the input",
            "Each theme should be supported by multiple topics",
        ]

        try:
            import time
            
            start_time = time.time()

            response = m.instruct(
                prompt,
                user_variables={
                    "context": context_str,
                    "topics_text": topics_text
                },
                model_options={
                    "max_tokens": 15000,
                    "temperature": 0.0
                },
                requirements=requirements,
                format=ThemeList,
                strategy=RejectionSamplingStrategy(loop_budget=1),
                return_sampling_results=True
            )

            elapsed_time = time.time() - start_time
            logger.info(f"Mellea theme extraction completed in {elapsed_time:.2f} seconds")

            # Print validation results
            print_mellea_validations(response, title="Theme Extraction Validations")

            # Parse response
            theme_list = ThemeList.model_validate_json(response._underlying_value)
            logger.info(f"Successfully parsed {len(theme_list.themes)} themes from LLM response")

            # Hydrate: Replace LLM-generated topics with original Topic objects
            hydration_results = self._hydrate_theme_topics(theme_list, all_topic_lists)

            # Populate metadata
            theme_list.study_id = self.id
            theme_list.generated_at = datetime.now().isoformat()

            # Validate each theme
            theme_validations = []

            logger.info("Starting per-theme validation...")
            for idx, theme in enumerate(theme_list.themes, start=1):
                # Validate this theme
                logger.info(f"Validating theme {idx}/{len(theme_list.themes)}: {theme.title}")
                theme_validation = self._validate_theme(
                    theme,
                    hydration_results[theme.title],
                    m,
                    context_str
                )
                theme_validations.append(theme_validation)
                logger.info(f"  Theme '{theme.title}' validation: {theme_validation.status}")

            # Aggregate per-theme validations into overall validation
            overall_validation = self._aggregate_theme_validations(theme_validations)

            # Cache the ThemeList
            self.themes_top_down = theme_list
            logger.debug(f"Cached ThemeList in study.themes_top_down")

            logger.info(f"Theme generation completed successfully (overall status: {overall_validation.status})")

            # Return ValidatedList with per-theme validations
            return ValidatedList(
                result=theme_list.themes,
                validation=overall_validation,
                item_validations=theme_validations
            )

        except Exception as e:
            logger.error(f"Theme generation failed: {str(e)}")
            validation = IffyIndex.from_check(
                method="theme_generation",
                status="iffy",
                explanation=f"Failed to generate themes: {str(e)}"
            )
            return Validated(result=None, validation=validation)

    def save_state(self, cache_dir: Path) -> Path:
        """
        Save complete study state (all interviews + themes).

        Parameters
        ----------
        cache_dir : Path
            Directory to save study state and all interview caches

        Returns
        -------
        Path
            Path to the study state file
        """
        from .cache import save_study_state
        return save_study_state(self, cache_dir)

    @classmethod
    def load_from_cache(cls, cache_dir: Path) -> 'Study':
        """
        Load study from cache directory.

        Parameters
        ----------
        cache_dir : Path
            Directory containing study_state.json and interview caches

        Returns
        -------
        Study
            Reconstructed study with all interviews and state
        """
        from .cache import load_study_state
        return load_study_state(cache_dir)