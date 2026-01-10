import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from arcagi2.evaluation.utils import EvaluationMetadata, EvaluationStatus
from arcagi2.solver.solver import SolverStatus
from arcagi2.utils.utils import read_file


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds to Xm Ys format."""
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def format_time(dt: datetime) -> str:
    """Format datetime to HH:MM:SS."""
    return dt.strftime("%H:%M:%S")


class StatusDashboard:
    """Dashboard that prints full status on state changes only."""
    
    def __init__(
        self,
        output_folder: Path,
        total_puzzles: int,
        eval_start_time: datetime,
        poll_interval: float = 5,
        max_rows: int = 5,    # Max rows to show per section before "+ N more"
    ):
        self.output_folder = output_folder
        self.total_puzzles = total_puzzles
        self.eval_start_time = eval_start_time
        self.poll_interval = poll_interval
        self.max_rows = max_rows
        
        # Track previous state to detect changes
        # Key: submission_id, Value: (status, solver_status)
        self._prev_state: dict[str, tuple[EvaluationStatus, SolverStatus]] = {}

    def _scan_status(self) -> dict[str, tuple[EvaluationMetadata, SolverStatus]]:
        """Scan output folder and return current state of all items."""
        items = {}

        if not self.output_folder.exists():
            return items

        for subfolder in self.output_folder.iterdir():
            if not subfolder.is_dir():
                continue
            metadata_file = subfolder / "metadata.json"
            if not metadata_file.exists():
                continue
            try:
                data = json.loads(read_file(metadata_file))
                metadata = EvaluationMetadata.from_dict(data)
                solver_status = SolverStatus.from_output_folder(subfolder)
                items[metadata.submission_id] = (metadata, solver_status)
            except Exception:
                continue

        return items

    def _get_changes(self, current_state: dict[str, tuple[EvaluationMetadata, SolverStatus]]) -> list[str]:
        """Get list of change descriptions."""
        changes = []
        for submission_id, (metadata, solver_status) in current_state.items():
            prev = self._prev_state.get(submission_id)
            if prev is None:
                changes.append(f"started: puzzle {metadata.puzzle_id}")
            else:
                prev_status, prev_solver_status = prev
                if metadata.status != prev_status:
                    changes.append(f"{metadata.status.value}: puzzle {metadata.puzzle_id}")
                elif solver_status.sample_num != prev_solver_status.sample_num:
                    changes.append(f"sample {solver_status.sample_num}: puzzle {metadata.puzzle_id}")
                elif solver_status.stage != prev_solver_status.stage:
                    changes.append(f"{solver_status.stage}: puzzle {metadata.puzzle_id}")
        return changes

    def _update_prev_state(self, current_state: dict[str, tuple[EvaluationMetadata, SolverStatus]]):
        """Update the previous state tracking."""
        for submission_id, (metadata, solver_status) in current_state.items():
            self._prev_state[submission_id] = (metadata.status, solver_status)

    # Formatting helpers for _build_display

    @staticmethod
    def _format_item_prefix(metadata: EvaluationMetadata) -> str:
        return (
            f"  {metadata.submission_id} / puzzle {metadata.puzzle_id} | "
            f"started {format_time(metadata.start_time)}"
        )

    @staticmethod
    def _format_finished_item(metadata: EvaluationMetadata, _: SolverStatus) -> str:
        return f"{StatusDashboard._format_item_prefix(metadata)} | duration {format_duration(metadata.duration_seconds)}"

    @staticmethod
    def _format_running_item(metadata: EvaluationMetadata, solver_status: SolverStatus, now: datetime) -> str:
        elapsed_running = (now - metadata.start_time).total_seconds()
        if solver_status.sample_num is not None:
            status_str = f"sample {solver_status.sample_num} / {solver_status.stage}"
        else:
            status_str = "starting"
        return f"{StatusDashboard._format_item_prefix(metadata)} | elapsed {format_duration(elapsed_running)} | {status_str}"

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

    def _build_display(self, current_state: dict[str, tuple[EvaluationMetadata, SolverStatus]], changes: list[str]) -> str:
        """Build the full dashboard display string."""
        now = datetime.now()
        elapsed = (now - self.eval_start_time).total_seconds()
        
        # Categorize items
        errored = []
        timeout = []
        running = []
        success = []
        
        for submission_id, (metadata, solver_status) in current_state.items():
            item = (metadata, solver_status)
            if metadata.status == EvaluationStatus.ERROR:
                errored.append(item)
            elif metadata.status == EvaluationStatus.TIMEOUT:
                timeout.append(item)
            elif metadata.status == EvaluationStatus.RUNNING:
                running.append(item)
            elif metadata.status == EvaluationStatus.SUCCESS:
                success.append(item)
        
        # Sort by start_time
        errored.sort(key=lambda x: x[0].start_time)
        timeout.sort(key=lambda x: x[0].start_time)
        running.sort(key=lambda x: x[0].start_time)
        success.sort(key=lambda x: x[0].start_time)
        
        # Calculate stats
        finished = errored + timeout + success
        total_duration = sum(
            m.duration_seconds for m, _ in finished 
            if m.duration_seconds is not None
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
            f"Time: {format_time(now)} | "
            f"Finished: {len(finished)}/{self.total_puzzles} | "
            f"Running: {len(running)} | "
            f"Avg: {format_duration(avg_time)} | "
            f"Elapsed: {format_duration(elapsed)}"
        )
        
        lines.extend(self._build_section("Errored", errored, self._format_finished_item))
        lines.extend(self._build_section("Timeout", timeout, self._format_finished_item))
        lines.extend(self._build_section("Running", running, lambda m, s: self._format_running_item(m, s, now)))
        lines.extend(self._build_section("Success", success, self._format_finished_item))
        
        lines.append("═" * 80)
        return "\n".join(lines)

    def update(self):
        """Scan status and print display if there are changes."""
        try:
            current_state = self._scan_status()
            changes = self._get_changes(current_state)
            
            if changes:
                print(self._build_display(current_state, changes))
                print()
                self._update_prev_state(current_state)
        except Exception as e:
            print(f"[Dashboard error: {e}]")

    async def run(self):
        """Run the dashboard polling loop. Cancel the task to stop."""
        print(f"Polling for status changes every {self.poll_interval} seconds...")
        while True:
            self.update()
            await asyncio.sleep(self.poll_interval)

    def print_error(self, error: Exception):
        """Print a single line indicating an error occurred."""
        print(f"Error: {error}.\nAll tasks have been cancelled. Evaluation failed.")
