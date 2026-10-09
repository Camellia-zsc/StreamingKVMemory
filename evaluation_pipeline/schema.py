from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Question:
    question_id: str
    text: str
    options: List[str] = field(default_factory=list)
    timestamp: Optional[float] = None
    task_type: Optional[str] = None
    answer: Any = None
    response_policy: str = "reply_required"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalSample:
    benchmark: str
    sample_id: str
    video_path: str
    duration: Optional[float] = None
    questions: List[Question] = field(default_factory=list)
    mode: str = "offline"
    metadata: Dict[str, Any] = field(default_factory=dict)
