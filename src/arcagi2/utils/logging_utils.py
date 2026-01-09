import logging
from pathlib import Path

from contextlib import contextmanager
from contextvars import ContextVar


task_id_var = ContextVar("task_id_var", default=None)

# Dedicated logger for infrastructure issues (rate limits, connection errors, sandbox failures).
# Used across modules (turn.py, base.py, sandbox implementations) to log retries with exponential
# backoff and sandbox deletion failures. Unlike regular loggers, this one can have a console
# handler attached to surface infrastructure issues during evaluation runs.
infra_logger = logging.getLogger("arcagi2.infra")


def setup_infra_logger_console_handler(level: int):
    """
    Set up a console handler for infra_logger to display infrastructure issues in the terminal.
    
    This should be called at the start of evaluation runs (e.g., in evaluate()) to ensure
    infrastructure errors like rate limits, connection errors, and sandbox failures are
    visible in the console output, not just in log files.
    
    Safe to call multiple times - will not add duplicate handlers.
    """
    # Check if infra_logger already has a StreamHandler to avoid duplicates
    for handler in infra_logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            return
    
    infra_handler = logging.StreamHandler()
    infra_handler.setLevel(level)
    infra_handler.setFormatter(logging.Formatter("%(asctime)s - INFRA - %(message)s"))
    infra_logger.addHandler(infra_handler)

class TaskFilter(logging.Filter):
    def __init__(self, task_id):
        super().__init__()
        self.task_id = task_id

    def filter(self, record: logging.LogRecord) -> bool:
        return task_id_var.get() == self.task_id
    
@contextmanager
def per_task_file_logger(log_file: Path, level: int = logging.INFO):
    """
    Context manager that logs messages from the current asyncio context/task to a dedicated file.
    
    Useful for asyncio parallel processing where multiple tasks run concurrently and
    logs from the same module would otherwise interleave. Uses task_id_var to filter
    logs - only messages emitted in the current context/task will be logged to the file.
    
    Usage:
        tok = task_id_var.set("task_1")
        try:
            with per_task_file_logger(Path("task_1.log")):
                logger.info("This goes to task_1.log")
        finally:
            task_id_var.reset(tok)
    
    Note: Ensure root logger level is set to INFO or lower, otherwise INFO-level
    messages will be filtered before reaching this handler.
    """
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
