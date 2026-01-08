import logging
from pathlib import Path

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
