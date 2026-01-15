import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from arcagi2.evaluation.utils import PuzzleStatus


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds to Xm Ys format."""
    if seconds is None:
        return "—"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


class StatusDashboard:
    """Dashboard that prints status on sample count changes per puzzle."""
    
    def __init__(
        self,
        output_folder: Path,
        task_data: list[tuple[str, dict]],  # List of (puzzle_id, puzzle_json) tuples
        num_samples: int,
        eval_start_time: datetime,
        resume: bool,
        file_lock: asyncio.Lock,
        poll_interval: float = 5,
        max_rows: int = 5,    # Max rows to show per section before "+ N more"
    ):
        self.output_folder = output_folder
        self.task_data = task_data
        self.puzzle_ids = list(dict.fromkeys(puzzle_id for puzzle_id, _ in task_data))  # Unique puzzle IDs, preserving order
        self.total_tasks = len(task_data)
        self.num_samples = num_samples
        self.eval_start_time = eval_start_time
        self.file_lock = file_lock
        self.poll_interval = poll_interval
        self.max_rows = max_rows
        
        # Track previous sample counts per puzzle to detect changes
        self._prev_sample_counts: dict[str, int] = {}
        
        # If resuming, capture existing dirs to exclude from tracking
        self._exclude_dirs: set[str] = set()
        if resume and output_folder.exists():
            self._exclude_dirs = {d.name for d in output_folder.iterdir() if d.is_dir()}

    def _get_num_samples_for_puzzle(self, puzzle_id: str) -> int:
        """Get the number of samples for a specific puzzle from task_data."""
        return sum(1 for pid, _ in self.task_data if pid == puzzle_id)

    def _build_display(self, status_by_puzzle: dict[str, PuzzleStatus], changes: list[str]) -> str:
        """Build the full dashboard display string."""
        now = datetime.now()
        elapsed = (now - self.eval_start_time).total_seconds()
        
        # Categorize puzzles
        running_puzzles = []    # Puzzles with running samples
        successful_puzzles = [] # Puzzles with all samples successful
        errored_puzzles = []    # Puzzles with errored submissions

        completed = 0
        
        for puzzle_id, ps in status_by_puzzle.items():
            num_samples = self._get_num_samples_for_puzzle(puzzle_id)

            if len(ps.success) == num_samples:
                successful_puzzles.append(ps)
            if len(ps.error) > 0:
                errored_puzzles.append(ps)
            if len(ps.running) > 0:
                running_puzzles.append(ps)

            completed += len(ps.success) + len(ps.error)
        
        lines = []
        
        # Changes section
        lines.append("─" * 80)
        if len(changes) <= self.max_rows:
            lines.append(f"Changes: {', '.join(changes)}")
        else:
            lines.append(f"Changes: {', '.join(changes[:self.max_rows])} + {len(changes) - self.max_rows} more")
        lines.append("─" * 80)
        
        # Status line
        lines.append(f"Output: {self.output_folder}")
        lines.append(
            f"Time: {now.strftime('%H:%M:%S')} | "
            f"Completed: {completed}/{self.total_tasks} | "
            f"Running: {len(running_puzzles)} | "
            f"Elapsed: {format_duration(elapsed)}"
        )
        
        # Running section
        if running_puzzles:
            lines.append("")
            lines.append(f"Running ({len(running_puzzles)}):")
            for ps in running_puzzles[:self.max_rows]:
                num_samples = self._get_num_samples_for_puzzle(ps.puzzle_id)
                all_subs = ps.success + ps.running
                earliest_start = min(m.start_time for m in all_subs)
                puzzle_elapsed = (now - earliest_start).total_seconds()
                
                sample_progress = f"{len(ps.success) + len(ps.running)}/{num_samples}"
                done_ids = ", ".join(m.submission_id for m in ps.success) if ps.success else "none"
                running_ids = ", ".join(m.submission_id for m in ps.running) if ps.running else "none"
                
                lines.append(
                    f"  puzzle {ps.puzzle_id} | elapsed {format_duration(puzzle_elapsed)} | "
                    f"sample {sample_progress} | samples done: {done_ids} | samples running: {running_ids}"
                )
            if len(running_puzzles) > self.max_rows:
                lines.append(f"  + {len(running_puzzles) - self.max_rows} more")
        
        # Successful section
        if successful_puzzles:
            lines.append("")
            lines.append(f"Successful ({len(successful_puzzles)}):")
            for ps in successful_puzzles[:self.max_rows]:
                num_samples = self._get_num_samples_for_puzzle(ps.puzzle_id)
                submission_ids = ", ".join(m.submission_id for m in ps.success)
                lines.append(
                    f"  puzzle {ps.puzzle_id} | sample {len(ps.success)}/{num_samples} | "
                    f"samples: {submission_ids}"
                )
            if len(successful_puzzles) > self.max_rows:
                lines.append(f"  + {len(successful_puzzles) - self.max_rows} more")
        
        # Errored section
        if errored_puzzles:
            lines.append("")
            lines.append(f"Errored ({len(errored_puzzles)}):")
            for ps in errored_puzzles[:self.max_rows]:
                errored_ids = ", ".join(m.submission_id for m in ps.error)
                lines.append(f"  puzzle {ps.puzzle_id} | errored: {errored_ids}")
            if len(errored_puzzles) > self.max_rows:
                lines.append(f"  + {len(errored_puzzles) - self.max_rows} more")
        
        lines.append("═" * 80)
        return "\n".join(lines)

    def _compute_changes(self, current_counts: dict[str, int]) -> list[str]:
        """Compute list of change descriptions from previous to current counts."""
        changes = []
        for puzzle_id, count in current_counts.items():
            prev_count = self._prev_sample_counts.get(puzzle_id, 0)
            if count != prev_count:
                num_samples = self._get_num_samples_for_puzzle(puzzle_id)
                changes.append(f"puzzle {puzzle_id}: {count}/{num_samples}")
        return changes

    def update(self):
        """Scan status and print display if sample counts have changed."""
        try:
            status_by_puzzle = PuzzleStatus.scan(self.output_folder, exclude_dirs=self._exclude_dirs)
            current_counts = {
                puzzle_id: status_by_puzzle[puzzle_id].count_samples(count_errored_samples=True)
                for puzzle_id in self.puzzle_ids
                if puzzle_id in status_by_puzzle
            }
            
            if current_counts != self._prev_sample_counts:
                changes = self._compute_changes(current_counts)
                print(self._build_display(status_by_puzzle, changes))
                print()
                self._prev_sample_counts = current_counts
        except Exception as e:
            print(f"[Dashboard error: {e}]")

    async def run(self):
        """Run the dashboard polling loop. Cancel the task to stop."""
        print(f"Polling for status changes every {self.poll_interval} seconds...")
        while True:
            async with self.file_lock:
                await asyncio.to_thread(self.update)  # Run in thread pool to avoid blocking event loop
            await asyncio.sleep(self.poll_interval)

    def print_error(self, error: Exception):
        """Print a single line indicating an error occurred."""
        print(f"Error: {error}.\nAll tasks have been cancelled. Evaluation failed.")
