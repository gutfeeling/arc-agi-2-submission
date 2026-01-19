from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


class EvaluationStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    CANCELLED = "cancelled"
    ERROR = "error"

@dataclass
class Metadata:
    uid: str
    puzzle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    status: EvaluationStatus

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "puzzle_id": self.puzzle_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Metadata":
        return cls(
            uid=data["uid"],
            puzzle_id=data["puzzle_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] is not None else None,
            duration_seconds=data["duration_seconds"],
            status=EvaluationStatus(data["status"]),
        )

def sort_by_majority(outputs: list[Any]) -> list[tuple[Any, int]]:
    result = []
    for output in outputs:
        if output not in [x[0] for x in result]:
            result.append([output, outputs.count(output)])
    return sorted(result, key=lambda x: x[1], reverse=True)
