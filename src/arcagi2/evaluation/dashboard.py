import asyncio
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from arcagi2.evaluation.utils import Metadata, EvaluationStatus
from arcagi2.solver.solver import SolverStatus
from arcagi2.utils.utils import read_file

@dataclass
class SampleState:
    """State for a sample, used in dashboard for display and change detection."""
    metadata: Metadata
    solver_status: Optional[SolverStatus]

@dataclass
class PuzzleState:
    """State for a puzzle and its samples, used in dashboard for display and change detection."""
    metadata: Metadata
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
                    puzzle_metadata = Metadata.from_dict(data)
                    
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
                                metadata=sample_metadata,
                                solver_status=solver_status,
                            )
                        except Exception:
                            continue
                    
                    items[puzzle_folder.name] = PuzzleState(
                        metadata=puzzle_metadata,
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
                changes.append(f"started: puzzle {puzzle_state.metadata.puzzle_id} / folder {puzzle_uid}")
            else:
                if puzzle_state.metadata.status != prev.metadata.status:
                    changes.append(f"{puzzle_state.metadata.status.value}: puzzle {puzzle_state.metadata.puzzle_id} / folder {puzzle_uid}")
            
            # Check sample changes
            prev_samples = prev.samples if prev else {}
            for sample_id, sample_state in puzzle_state.samples.items():
                prev_sample = prev_samples.get(sample_id)
                if prev_sample is None:
                    changes.append(f"started: puzzle {puzzle_state.metadata.puzzle_id} / folder {puzzle_uid} / {sample_id}")
                else:
                    if sample_state.metadata.status != prev_sample.metadata.status:
                        changes.append(f"{sample_state.metadata.status.value}: puzzle {puzzle_state.metadata.puzzle_id} / folder {puzzle_uid} / {sample_id}")
                    elif sample_state.solver_status is not None:
                        prev_solver = prev_sample.solver_status
                        prev_sample_num = prev_solver.sample_num if prev_solver else None
                        prev_stage = prev_solver.stage if prev_solver else None
                        curr_sample_num = sample_state.solver_status.sample_num
                        curr_stage = sample_state.solver_status.stage
                        if curr_sample_num != prev_sample_num or curr_stage != prev_stage:
                            if curr_sample_num is not None:
                                status_str = f"sample {curr_sample_num} / {curr_stage}"
                            else:
                                status_str = "starting"
                            changes.append(f"{status_str}: puzzle {puzzle_state.metadata.puzzle_id} / folder {puzzle_uid} / {sample_id}")
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
            f"  puzzle {puzzle_state.metadata.puzzle_id} / folder {puzzle_uid} | "
            f"started {StatusDashboard._format_time(puzzle_state.metadata.start_time)}"
        )

    @staticmethod
    def _format_sample_statuses_multiline(samples, now) -> list[str]:
        """Format sample statuses as multiple lines, one per sample."""
        if not samples:
            return ["    (no samples)"]
        lines = []
        for sample_id, sample_state in sorted(samples.items()):
            if sample_state.metadata.status == EvaluationStatus.RUNNING:
                # For running samples, show sample N / stage with start time and elapsed
                solver_status = sample_state.solver_status
                if solver_status and solver_status.sample_num is not None:
                    status_str = f"sample {solver_status.sample_num} / {solver_status.stage}"
                else:
                    status_str = "starting"
                elapsed = (now - sample_state.metadata.start_time).total_seconds()
                lines.append(f"    {sample_id}: {status_str} | started {StatusDashboard._format_time(sample_state.metadata.start_time)} | elapsed {StatusDashboard._format_duration(elapsed)}")
            else:
                # For finished samples, show the final status and duration
                duration_str = StatusDashboard._format_duration(sample_state.metadata.duration_seconds)
                lines.append(f"    {sample_id}: {sample_state.metadata.status.value} ({duration_str})")
        return lines

    @staticmethod
    def _format_finished_item(puzzle_uid, puzzle_state, now) -> str:
        prefix = StatusDashboard._format_item_prefix(puzzle_uid, puzzle_state)
        line1 = f"{prefix} | duration {StatusDashboard._format_duration(puzzle_state.metadata.duration_seconds)}"
        sample_lines = StatusDashboard._format_sample_statuses_multiline(puzzle_state.samples, now)
        return line1 + "\n" + "\n".join(sample_lines)

    @staticmethod
    def _format_running_item(puzzle_uid, puzzle_state, now) -> str:
        # For running puzzles, just show puzzle info without start/elapsed at puzzle level
        line1 = f"  puzzle {puzzle_state.metadata.puzzle_id} / folder {puzzle_uid}"
        sample_lines = StatusDashboard._format_sample_statuses_multiline(puzzle_state.samples, now)
        return line1 + "\n" + "\n".join(sample_lines)

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
            if puzzle_state.metadata.status == EvaluationStatus.ERROR:
                errored.append(item)
            elif puzzle_state.metadata.status == EvaluationStatus.CANCELLED:
                cancelled.append(item)
            elif puzzle_state.metadata.status == EvaluationStatus.RUNNING:
                running.append(item)
            elif puzzle_state.metadata.status == EvaluationStatus.SUCCESS:
                success.append(item)
            
            # Collect sample errors
            for sample_id, sample_state in puzzle_state.samples.items():
                if sample_state.metadata.status == EvaluationStatus.ERROR:
                    sample_errors.append((puzzle_state.metadata.puzzle_id, puzzle_uid, sample_id))
        
        # Sort by start_time
        errored.sort(key=lambda x: x[1].metadata.start_time)
        cancelled.sort(key=lambda x: x[1].metadata.start_time)
        running.sort(key=lambda x: x[1].metadata.start_time)
        success.sort(key=lambda x: x[1].metadata.start_time)
        
        # Calculate puzzle stats
        finished_puzzles = errored + cancelled + success
        total_puzzle_duration = sum(
            ps.metadata.duration_seconds for _, ps in finished_puzzles 
            if ps.metadata.duration_seconds is not None
        )
        avg_puzzle_time = total_puzzle_duration / len(finished_puzzles) if len(finished_puzzles) > 0 else 0
        total_samples_in_finished = sum(len(ps.samples) for _, ps in finished_puzzles)
        avg_samples_per_puzzle = total_samples_in_finished / len(finished_puzzles) if len(finished_puzzles) > 0 else 0
        
        # Calculate sample stats
        samples_finished = 0
        samples_running = 0
        total_sample_duration = 0.0
        for _, puzzle_state in current_state.items():
            for _, sample_state in puzzle_state.samples.items():
                if sample_state.metadata.status == EvaluationStatus.RUNNING:
                    samples_running += 1
                else:
                    samples_finished += 1
                    if sample_state.metadata.duration_seconds is not None:
                        total_sample_duration += sample_state.metadata.duration_seconds
        avg_sample_time = total_sample_duration / samples_finished if samples_finished > 0 else 0
        
        lines = []
        
        # Print changes (limited)
        lines.append("─" * 80)
        if len(changes) <= self.max_rows:
            lines.append(f"Changes: {', '.join(changes)}")
        else:
            lines.append(f"Changes: {', '.join(changes[:self.max_rows])} + {len(changes) - self.max_rows} more")
        lines.append("─" * 80)
        
        lines.append(f"Output: {self.output_folder}")
        lines.append(f"Time: {self._format_time(now)} | Elapsed: {self._format_duration(elapsed)}")
        lines.append(
            f"Puzzle Stats -> Finished: {len(finished_puzzles)}/{self.total_puzzles} | "
            f"Running: {len(running)} | "
            f"Avg time: {self._format_duration(avg_puzzle_time)} | "
            f"Avg samples: {avg_samples_per_puzzle:.1f}"
        )
        lines.append(
            f"Sample Stats -> Finished: {samples_finished} | "
            f"Running: {samples_running} | "
            f"Avg time: {self._format_duration(avg_sample_time)}"
        )
        
        lines.extend(self._build_section(
            "Puzzle Errors",
            errored,
            lambda uid, ps: f"  puzzle {ps.metadata.puzzle_id} / folder {uid}"
        ))
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
