from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


class EvaluationStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class EvaluationMetadata:
    submission_id: str
    puzzle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    status: EvaluationStatus

    def to_dict(self) -> dict:
        return {
            "submission_id": self.submission_id,
            "puzzle_id": self.puzzle_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationMetadata":
        return cls(
            submission_id=data["submission_id"],
            puzzle_id=data["puzzle_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] is not None else None,
            duration_seconds=data["duration_seconds"],
            status=EvaluationStatus(data["status"]),
        )