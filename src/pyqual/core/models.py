from pydantic import BaseModel
from typing import List


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
