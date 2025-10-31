from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class IntervieweeIdentification(BaseModel):
    """LLM identification of the interviewee in a transcript."""
    interviewee: str = Field(
        description="The speaker ID of the identified interviewee (must match a speaker in the transcript)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the identification: high (clear patterns), medium (some ambiguity), low (uncertain)"
    )
    explanation: str = Field(
        description="1-2 sentences explaining why this speaker is the interviewee based on conversation patterns"
    )


class CoherenceAssessment(BaseModel):
    """LLM assessment of theme coherence."""
    rating: Literal["Strong", "Acceptable", "Weak"] = Field(
        description="Coherence rating: Strong (tight conceptual fit), Acceptable (generally related), or Weak (disconnected)"
    )
    explanation: str = Field(
        description="1-2 sentences explaining the rating"
    )


class Quote(BaseModel):
    index: int
    timestamp: str
    speaker: str
    quote: str

class Topic(BaseModel):
    topic: str
    explanation: str
    quotes: List[Quote]
    interview_id: Optional[str] = None

class TopicList(BaseModel):
    topics: List[Topic]
    generated_at: Optional[str] = None

class Theme(BaseModel):
    title: str
    description: str
    explanation: str
    topics: List[Topic]
    prospective: bool = False  # True if theme appears in only 1 interview

class ThemeList(BaseModel):
    themes: List[Theme]
    study_id: Optional[str] = None
    generated_at: Optional[str] = None
