import uuid
from pathlib import Path
from typing import Optional

from .interview import Interview

class Study:
    """
    A collection of qualitative documents (for now only interviews are supported).
    """

    def __init__(self, files_or_docs=None, metadata=None, doc_cls=Interview, headers: Optional[dict] = None):
        """
        Parameters
        ----------
        files : list[str | Path | Interview], optional
            Either a list of file paths OR a list of Interview objects.
        metadata : dict, optional
            Metadata to attach at corpus level.
        doc_cls : class, default=Interview
            Currently must be Interview
        headers: dict, optional
            Headers of columns for timestamp, speaker and statements in the documents
        """
        if doc_cls is not Interview:
            raise ValueError("Study currently only supports Interview documents.")

        self.id = f"study_{uuid.uuid4().hex[:8]}"
        self.doc_cls = doc_cls 
        self.metadata = metadata or {}
        self.documents = []

        if files_or_docs:
            for item in files_or_docs:
                self._add_checked(item, headers=headers)

    def _add_checked(self, file_or_doc, headers: dict):

        if isinstance(file_or_doc, (str, Path)):
            self.documents.append(self.doc_cls(file=file_or_doc, headers=headers))
        elif isinstance(file_or_doc, self.doc_cls):
            self.documents.append(file_or_doc)
        else:
            raise TypeError(
                f"Invalid type {type(file_or_doc)}. "
                f"Corpus only supports {self.doc_cls.__name__} objects or file paths."
            )
        
    def add(self, file_or_doc, doc_cls=Interview):
        """Add an Interview object or a file path."""
        self._add_checked(file_or_doc)


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
        results = {}
        for doc in self.documents:
            predicted = doc.identify_interviewee(m=m)
            results[doc.id] = predicted
    
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
        return (
            f"<Study {self.id}: {len(self)} {self.doc_cls.__name__}(s), "
            f"metadata={self.metadata}>"
        )