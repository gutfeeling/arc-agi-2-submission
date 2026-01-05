import asyncio
import logging
from pathlib import Path
import sys

from contextlib import contextmanager
from contextvars import ContextVar


task_id_var = ContextVar("task_id_var", default=None)

class TaskFilter(logging.Filter):
    def __init__(self, task_id):
        super().__init__()
        self.task_id = task_id

    def filter(self, record: logging.LogRecord) -> bool:
        return task_id_var.get() == self.task_id
    
@contextmanager
def per_task_file_logger(log_file: Path, level: int = logging.INFO):
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    fh.addFilter(TaskFilter(task_id_var.get()))
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)
    try:
        yield
    finally:
        root_logger.removeHandler(fh)
        fh.close()

def setup_logger_for_parallel_processing_script(logger: logging.Logger, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Since parallel processing can produce busy logs, we don't want any console logging. We just want to see progress.
    Therefore, we don't give the root logger any console handlers in this script and don't propagate either.
    """
    logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False  # Don't propagate to root logger

    return logger

class ProgressCounter:
    def __init__(self, total_items: int, logger: logging.Logger):
        self.completed = 0
        self.total = total_items
        self.lock = asyncio.Lock()
        self.logger = logger

    async def increment(self) -> int:
        async with self.lock:
            self.completed += 1
            percentage = self.completed / self.total * 100
            self.logger.info(
                f"Progress: {self.completed}/{self.total} completed ({percentage:.1f}%)"
            )
            return self.completed
