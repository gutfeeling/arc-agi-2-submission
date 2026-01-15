from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Optional

from arcagi2.utils.utils import read_file


logger = logging.getLogger(__name__)

class SampleStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class SampleMetadata:
    submission_id: str
    puzzle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    status: SampleStatus

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
    def from_dict(cls, data: dict) -> "SampleMetadata":
        return cls(
            submission_id=data["submission_id"],
            puzzle_id=data["puzzle_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] is not None else None,
            duration_seconds=data["duration_seconds"],
            status=SampleStatus(data["status"]),
        )

@dataclass
class PuzzleStatus:
    """Status of submissions for a single puzzle."""
    puzzle_id: str
    success: list[SampleMetadata]
    running: list[SampleMetadata]
    error: list[SampleMetadata]

    def count_samples(self, count_errored_samples: bool) -> int:
        """Count of valid (completed) samples for this puzzle."""
        if count_errored_samples:
            return len(self.success) + len(self.error)
        return len(self.success)

    @classmethod
    def scan(
        cls,
        output_folder: Path,
        exclude_dirs: Optional[set[str]] = None,
    ) -> dict[str, "PuzzleStatus"]:
        """
        Scan output folder and return dict of puzzle_id -> PuzzleStatus.
        
        exclude_dirs: Optional set of directory names to skip (e.g., for excluding resumed tasks).
        """
        if exclude_dirs is None:
            exclude_dirs = set()
        
        # Collect submissions grouped by puzzle_id
        by_puzzle: defaultdict[str, dict[str, list]] = defaultdict(
            lambda: {"success": [], "running": [], "error": []}
        )
        
        if not output_folder.exists():
            return {}
        
        for subfolder in output_folder.iterdir():
            if not subfolder.is_dir():
                continue
            if subfolder.name in exclude_dirs:
                continue
            metadata_file = subfolder / "metadata.json"
            if not metadata_file.exists():
                continue
            try:    # Since it's used by critical methods, we default to being defensive here.
                metadata = SampleMetadata.from_dict(
                    json.loads(read_file(metadata_file))
                )
            except Exception:
                logger.exception(f"Error reading metadata from {metadata_file}")
                continue
            
            if metadata.status == SampleStatus.SUCCESS:
                by_puzzle[metadata.puzzle_id]["success"].append(metadata)
            elif metadata.status == SampleStatus.RUNNING:
                by_puzzle[metadata.puzzle_id]["running"].append(metadata)
            elif metadata.status == SampleStatus.ERROR:
                by_puzzle[metadata.puzzle_id]["error"].append(metadata)
        
        return {
            puzzle_id: cls(
                puzzle_id=puzzle_id,
                success=data["success"],
                running=data["running"],
                error=data["error"],
            )
            for puzzle_id, data in by_puzzle.items()
        }


def choose_best_output_grids(puzzle_id: str, puzzle_json: dict, output_folder: Path) -> dict:
    """
    Choose the best output grids from all samples in the output folder using majority voting.
    
    Uses PuzzleStatus.scan to find successful samples, reads their submission.json files,
    and performs majority voting across all samples for each test case.
    
    Returns a dict in the format: {puzzle_id: [{"attempt_1": grid_or_none, "attempt_2": grid_or_none}, ...]}
    that can be merged with the main submission file.
    """   
    # Get all successful samples for this puzzle
    status_by_puzzle = PuzzleStatus.scan(output_folder)
    puzzle_status = status_by_puzzle.get(puzzle_id)

    if puzzle_status is None:
        return {
            puzzle_id: [
                {"attempt_1": None, "attempt_2": None}
                for _ in range(len(puzzle_json["test"]))
            ]
        }
    
    # Collect all output grids from successful samples
    # Structure: {test_index: [[grid, count], ...]}
    solutions_by_test = {
        test_index: [] for test_index in range(len(puzzle_json["test"]))
    }
    
    for sample_metadata in puzzle_status.success + puzzle_status.error:    # Errored ones may still have some data
        sample_submission_file = output_folder / sample_metadata.submission_id / "submission.json"
        if not sample_submission_file.exists():
            continue
        
        try:
            sample_submission = json.loads(read_file(sample_submission_file))
            # submission_data format: {puzzle_id: [{"attempt_1": grid, "attempt_2": None}, ...]}
            if puzzle_id not in sample_submission:
                continue
            
            submission_data = sample_submission[puzzle_id]
            for test_index, attempts in enumerate(submission_data):
                output_grid = attempts.get("attempt_1")
                if output_grid is None:
                    continue
                
                # Check if this grid already exists in our list (for counting)
                found = False
                for i in range(len(solutions_by_test[test_index])):
                    if solutions_by_test[test_index][i][0] == output_grid:
                        solutions_by_test[test_index][i][1] += 1
                        found = True
                        break
                if not found:
                    solutions_by_test[test_index].append([output_grid, 1])
        except Exception:
            logger.exception(f"Error reading submission from {sample_submission_file}")
            continue
    
    # Sort by majority vote (grids that appear more often are better)
    sorted_solutions = {
        test_index: sorted(grids, key=lambda x: x[1], reverse=True)
        for test_index, grids in solutions_by_test.items()
    }
    
    # Build submission in the expected format
    submission_data = [
        {
            "attempt_1": sorted_solutions[test_index][0][0] if len(sorted_solutions[test_index]) > 0 else None,
            "attempt_2": sorted_solutions[test_index][1][0] if len(sorted_solutions[test_index]) > 1 else None,
        }
        for test_index in range(len(puzzle_json["test"]))
    ]

    return {puzzle_id: submission_data}
