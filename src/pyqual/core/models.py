from pydantic import BaseModel
from typing import List

class Topic(BaseModel):
    topic: str
    explanation: str

class TopicList(BaseModel):
    topics: List[Topic]