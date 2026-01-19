import asyncio
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from arcagi2.evaluation.utils import PuzzleMetadata, Metadata, EvaluationStatus
from arcagi2.solver.solver import SolverStatus
from arcagi2.utils.utils import read_file

@dataclass
class SampleState:
    """State for a sample, used in dashboard for display and change detection."""
    status: EvaluationStatus
    start_time: datetime
    duration_seconds: Optional[float]
    solver_status: Optional[SolverStatus]

@dataclass
class PuzzleState:
    """State for a puzzle and its samples, used in dashboard for display and change detection."""
    puzzle_id: str
    status: EvaluationStatus
    start_time: datetime
    duration_seconds: Optional[float]
    samples: dict[str, SampleState]

class StatusDashboard:
    """Dashboard that prints full status on state changes only."""
    
    def __init__(
        self,
        output_folder: Path,
        total_puzzles: int,
        eval_start_time: datetime,
        resume: bool,
        lock: asyncio.Lock,
        poll_interval: float = 5,
        max_rows: int = 5,    # Max rows to show per section before "+ N more"
    ):
        self.output_folder = output_folder
        self.total_puzzles = total_puzzles
        self.eval_start_time = eval_start_time
        self.lock = lock
        self.poll_interval = poll_interval
        self.max_rows = max_rows
        
        # Track previous state to detect changes
        self._prev_state: dict[str, PuzzleState] = {}
        
        # If resuming, capture existing dirs to exclude from tracking
        self._exclude_dirs: set[str] = set()
        if resume and output_folder.exists():
            self._exclude_dirs = {d.name for d in output_folder.iterdir() if d.is_dir()}

    async def _scan_status(self) -> dict[str, PuzzleState]:
        """Scan output folder and return current state of all puzzles and their samples."""
        items = {}

        if not self.output_folder.exists():
            return items

        async with self.lock:
            for puzzle_folder in self.output_folder.iterdir():
                if not puzzle_folder.is_dir():
                    continue
                if puzzle_folder.name in self._exclude_dirs:
                    continue
                puzzle_metadata_file = puzzle_folder / "metadata.json"
                if not puzzle_metadata_file.exists():
                    continue
                try:
                    data = json.loads(read_file(puzzle_metadata_file))
                    puzzle_metadata = PuzzleMetadata.from_dict(data)
                    
                    # Scan sample subfolders
                    samples = {}
                    for sample_folder in puzzle_folder.iterdir():
                        if not sample_folder.is_dir():
                            continue
                        sample_metadata_file = sample_folder / "metadata.json"
                        if not sample_metadata_file.exists():
                            continue
                        try:
                            sample_data = json.loads(read_file(sample_metadata_file))
                            sample_metadata = Metadata.from_dict(sample_data)
                            # Get solver status for running samples
                            if sample_metadata.status == EvaluationStatus.RUNNING:
                                solver_status = SolverStatus.from_output_folder(sample_folder)
                            else:
                                solver_status = None
                            samples[sample_folder.name] = SampleState(
                                status=sample_metadata.status,
                                start_time=sample_metadata.start_time,
                                duration_seconds=sample_metadata.duration_seconds,
                                solver_status=solver_status,
                            )
                        except Exception:
                            continue
                    
                    items[puzzle_folder.name] = PuzzleState(
                        puzzle_id=puzzle_metadata.puzzle_id,
                        status=puzzle_metadata.status,
                        start_time=puzzle_metadata.start_time,
                        duration_seconds=puzzle_metadata.duration_seconds,
                        samples=samples,
                    )
                except Exception:
                    continue

        return items

    def _get_changes(self, current_state: dict[str, PuzzleState]) -> list[str]:
        """Get list of change descriptions for puzzles and samples."""
        changes = []
        for puzzle_uid, puzzle_state in current_state.items():
            prev = self._prev_state.get(puzzle_uid)
            if prev is None:
                changes.append(f"started: puzzle {puzzle_state.puzzle_id}")
            else:
                if puzzle_state.status != prev.status:
                    changes.append(f"{puzzle_state.status.value}: puzzle {puzzle_state.puzzle_id}")
            
            # Check sample changes
            prev_samples = prev.samples if prev else {}
            for sample_id, sample_state in puzzle_state.samples.items():
                prev_sample = prev_samples.get(sample_id)
                if prev_sample is None:
                    changes.append(f"sample started: puzzle {puzzle_state.puzzle_id} / {sample_id}")
                else:
                    if sample_state.status != prev_sample.status:
                        changes.append(f"sample {sample_state.status.value}: puzzle {puzzle_state.puzzle_id} / {sample_id}")
                    elif sample_state.solver_status is not None:
                        prev_stage = prev_sample.solver_status.stage if prev_sample.solver_status else None
                        if sample_state.solver_status.stage != prev_stage:
                            changes.append(f"sample {sample_state.solver_status.stage}: puzzle {puzzle_state.puzzle_id} / {sample_id}")
        return changes

    # Formatting helpers for _build_display

    @staticmethod
    def _format_duration(seconds) -> str:
        """Format duration in seconds to Xm Ys format."""
        if seconds is None:
            return "—"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"

    @staticmethod
    def _format_time(dt) -> str:
        """Format datetime to HH:MM:SS."""
        return dt.strftime("%H:%M:%S")

    @staticmethod
    def _format_item_prefix(puzzle_uid, puzzle_state) -> str:
        return (
            f"  puzzle {puzzle_state.puzzle_id} / folder {puzzle_uid} | "
            f"started {StatusDashboard._format_time(puzzle_state.start_time)}"
        )

    @staticmethod
    def _format_sample_statuses(samples, now) -> str:
        """Format sample statuses as a single line with elapsed/duration."""
        if not samples:
            return "    (no samples)"
        parts = []
        for sample_id, sample_state in sorted(samples.items()):
            if sample_state.status == EvaluationStatus.RUNNING:
                # For running samples, show the solver stage and elapsed time
                stage_str = sample_state.solver_status.stage if sample_state.solver_status and sample_state.solver_status.stage else "starting"
                elapsed = (now - sample_state.start_time).total_seconds()
                parts.append(f"{sample_id}: {stage_str} ({StatusDashboard._format_duration(elapsed)})")
            else:
                # For finished samples, show the final status and duration
                duration_str = StatusDashboard._format_duration(sample_state.duration_seconds)
                parts.append(f"{sample_id}: {sample_state.status.value} ({duration_str})")
        return "    " + " | ".join(parts)

    @staticmethod
    def _format_finished_item(puzzle_uid, puzzle_state, now) -> str:
        prefix = StatusDashboard._format_item_prefix(puzzle_uid, puzzle_state)
        line1 = f"{prefix} | duration {StatusDashboard._format_duration(puzzle_state.duration_seconds)}"
        line2 = StatusDashboard._format_sample_statuses(puzzle_state.samples, now)
        return f"{line1}\n{line2}"

    @staticmethod
    def _format_running_item(puzzle_uid, puzzle_state, now) -> str:
        prefix = StatusDashboard._format_item_prefix(puzzle_uid, puzzle_state)
        elapsed_running = (now - puzzle_state.start_time).total_seconds()
        line1 = f"{prefix} | elapsed {StatusDashboard._format_duration(elapsed_running)}"
        line2 = StatusDashboard._format_sample_statuses(puzzle_state.samples, now)
        return f"{line1}\n{line2}"

    def _build_section(self, name: str, items: list, formatter) -> list[str]:
        """Build a section of the display, returns list of lines."""
        if not items:
            return []
        lines = ["", f"{name} ({len(items)}):"]
        for item in items[:self.max_rows]:
            lines.append(formatter(*item))
        if len(items) > self.max_rows:
            lines.append(f"  + {len(items) - self.max_rows} more")
        return lines

    def _build_display(self, current_state: dict[str, PuzzleState], changes: list[str]) -> str:
        """Build the full dashboard display string."""
        now = datetime.now()
        elapsed = (now - self.eval_start_time).total_seconds()
        
        # Categorize puzzles and collect sample errors
        errored = []
        cancelled = []
        running = []
        success = []
        sample_errors = []  # List of (puzzle_id, puzzle_uid, sample_id)
        
        for puzzle_uid, puzzle_state in current_state.items():
            item = (puzzle_uid, puzzle_state)
            if puzzle_state.status == EvaluationStatus.ERROR:
                errored.append(item)
            elif puzzle_state.status == EvaluationStatus.CANCELLED:
                cancelled.append(item)
            elif puzzle_state.status == EvaluationStatus.RUNNING:
                running.append(item)
            elif puzzle_state.status == EvaluationStatus.SUCCESS:
                success.append(item)
            
            # Collect sample errors
            for sample_id, sample_state in puzzle_state.samples.items():
                if sample_state.status == EvaluationStatus.ERROR:
                    sample_errors.append((puzzle_state.puzzle_id, puzzle_uid, sample_id))
        
        # Sort by start_time
        errored.sort(key=lambda x: x[1].start_time)
        cancelled.sort(key=lambda x: x[1].start_time)
        running.sort(key=lambda x: x[1].start_time)
        success.sort(key=lambda x: x[1].start_time)
        
        # Calculate stats
        finished = errored + cancelled + success
        total_duration = sum(
            ps.duration_seconds for _, ps in finished 
            if ps.duration_seconds is not None
        )
        avg_time = total_duration / len(finished) if len(finished) > 0 else 0
        
        lines = []
        
        # Print changes (limited)
        lines.append("─" * 80)
        if len(changes) <= self.max_rows:
            lines.append(f"Changes: {', '.join(changes)}")
        else:
            lines.append(f"Changes: {', '.join(changes[:self.max_rows])} + {len(changes) - self.max_rows} more")
        lines.append("─" * 80)
        
        lines.append(f"Output: {self.output_folder}")
        lines.append(
            f"Time: {self._format_time(now)} | "
            f"Finished: {len(finished)}/{self.total_puzzles} | "
            f"Running: {len(running)} | "
            f"Avg: {self._format_duration(avg_time)} | "
            f"Elapsed: {self._format_duration(elapsed)}"
        )
        
        lines.extend(self._build_section("Errored", errored, lambda uid, ps: self._format_finished_item(uid, ps, now)))
        lines.extend(self._build_section("Cancelled", cancelled, lambda uid, ps: self._format_finished_item(uid, ps, now)))
        lines.extend(self._build_section("Running", running, lambda uid, ps: self._format_running_item(uid, ps, now)))
        lines.extend(self._build_section("Success", success, lambda uid, ps: self._format_finished_item(uid, ps, now)))
        
        # Sample errors section
        lines.extend(self._build_section(
            "Sample Errors",
            sample_errors,
            lambda puzzle_id, puzzle_uid, sample_id: f"  puzzle {puzzle_id} / folder {puzzle_uid} / {sample_id}"
        ))
        
        lines.append("═" * 80)
        return "\n".join(lines)

    async def update(self):
        """Scan status and print display if there are changes."""
        try:
            current_state = await self._scan_status()
            changes = self._get_changes(current_state)
            
            if changes:
                print(self._build_display(current_state, changes))
                print()
                self._prev_state = current_state
        except Exception as e:
            print(f"[Dashboard error: {e}]")

    async def run(self):
        """Run the dashboard polling loop. Cancel the task to stop."""
        print(f"Polling for status changes every {self.poll_interval} seconds...")
        while True:
            await self.update()
            await asyncio.sleep(self.poll_interval)

    def print_error(self, error: Exception):
        """Print a single line indicating an error occurred."""
        print(f"Error: {error}.\nAll tasks have been cancelled. Evaluation failed.")
