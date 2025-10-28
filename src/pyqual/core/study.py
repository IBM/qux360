import uuid
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .interview import Interview
from .models import ThemeList, TopicList
from .validated import Validated
from .iffy import IffyIndex
from mellea import MelleaSession

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
        topics_text_parts = []
        total_quotes_truncated = 0
        total_quote_length_truncated = 0

        for topic_list in all_topic_lists:
            interview_id = topic_list.interview_id or "unknown"
            topics_text_parts.append(f"\n=== Interview: {interview_id} ===")
            for topic in topic_list.topics:
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
            n_instruction = f"{n} cross-cutting themes"
            n_requirement = f"Generate exactly {n} themes"
        else:
            n_instruction = "as many cross-cutting themes as appropriate"
            n_requirement = "Generate as many themes as needed to capture the key patterns"

        prompt = """
        You are analyzing topics extracted from multiple interviews in a qualitative study.
        Your task is to identify """ + n_instruction + """ that emerge across these interviews.

        Study Context: {{context}}

        Topics from Interviews:
        {{topics_text}}

        For each theme, provide:
        1. A concise title (2-5 words)
        2. A description explaining what the theme represents
        3. An explanation of why this theme was chosen and how it manifests across interviews
        4. The specific topics (with their quotes) that support this theme
        """

        logger.debug(f"Calling Mellea for theme extraction (n={'unlimited' if n is None else n})...")

        requirements = [
            f"Each theme should be specific to the context of the overall study: {context_str}",
            "Each theme title should be 2-5 words, concise, concrete, but nuanced",
            "Each theme must include a detailed explanation of why it was chosen (more than 1 sentence)",
            "Each theme must reference the specific topics that support it, including their quotes",
            n_requirement
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
                format=ThemeList
            )

            elapsed_time = time.time() - start_time
            logger.debug(f"Mellea theme extraction completed in {elapsed_time:.2f} seconds")

            # Parse response
            theme_list = ThemeList.model_validate_json(response._underlying_value)
            logger.debug(f"Successfully parsed {len(theme_list.themes)} themes from LLM response")

            # Populate metadata
            theme_list.study_id = self.id
            theme_list.generated_at = datetime.now().isoformat()

            # Cache the ThemeList
            self.themes_top_down = theme_list
            logger.debug(f"Cached ThemeList in study.themes_top_down")

            # Create validation result
            # For now, simple validation - could be enhanced with quote checking, etc.
            validation = IffyIndex.from_check(
                method="theme_generation",
                status="ok",
                explanation=f"Generated {len(theme_list.themes)} themes across {len(all_topic_lists)} interviews"
            )

            logger.debug(f"Theme generation completed successfully")

            return Validated(result=theme_list, validation=validation)

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