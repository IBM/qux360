from pydantic import BaseModel
from typing import List, Optional


class Quote(BaseModel):
    index: int
    timestamp: str
    speaker: str
    quote: str

class Topic(BaseModel):
    topic: str
    explanation: str
    quotes: List[Quote]

class TopicList(BaseModel):
    topics: List[Topic]
    interview_id: Optional[str] = None
    generated_at: Optional[str] = None

class Theme(BaseModel):
    title: str
    description: str
    explanation: str
    topics: List[Topic]

class ThemeList(BaseModel):
    themes: List[Theme]
    study_id: Optional[str] = None
    generated_at: Optional[str] = None
