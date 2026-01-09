"""
Abstract base classes for code execution sandboxes.

Defines the minimal interface that all execution backends must implement:
- Sandbox: Lifecycle management for the execution environment
- REPL: Stateful code execution within a sandbox
- ExecutionResult: Result of code execution
- ExecutionError: Exception for user code failures
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import logging
from typing import Optional, Self

from arcagi2.exceptions import MaxRetriesExceeded
from arcagi2.sandbox.exceptions import SandboxInfrastructureError
from arcagi2.utils.logging_utils import infra_logger


logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """
    Result of code execution.
    
    Attributes:
        text: The output text from execution, or None if no output was produced.
    """
    text: Optional[str]


class ExecutionError(Exception):
    """
    Exception raised when user code execution fails.
    
    This exception is for errors in the user's code (syntax errors, runtime
    exceptions like ZeroDivisionError, etc.), NOT for infrastructure failures
    (network issues, sandbox crashes). Infrastructure errors are propagated
    as-is so callers can implement retry logic.
    
    Attributes:
        trace: The rich IPython-style error traceback from the user's code.
    """
    def __init__(self, trace: Optional[str]):
        self.trace = trace
        super().__init__(trace)


class REPL(ABC):
    """
    Async context manager for executing code in a stateful session.
    
    REPL = Read-Eval-Print-Loop. One REPL represents one kernel/session.
    State (variables, imports, etc.) persists across execute() calls within
    the same REPL session.
    
    Usage:
        async with sandbox.create_repl() as repl:
            await repl.execute("x = 1")
            result = await repl.execute("print(x)")  # prints 1
    """
    
    @abstractmethod
    async def __aenter__(self) -> Self:
        """Initialize the REPL session."""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the REPL session."""
        pass
    
    @abstractmethod
    async def execute(self, code: str, timeout: float) -> ExecutionResult:
        """
        Execute code and return the result.
        
        Args:
            code: Python code to execute.
            timeout: Maximum execution time in seconds.
            
        Returns:
            ExecutionResult with .text containing output (or None if no output).
            
        Raises:
            ExecutionError: If the user's code fails (syntax error, runtime error).
                The .trace attribute contains the IPython-style rich error traceback.
            asyncio.TimeoutError: If the timeout is exceeded.
            Other exceptions: Infrastructure errors (network, API, etc.) are
                propagated as-is. Callers should handle these separately,
                typically with retry logic.
        """
        pass


class Sandbox(ABC):
    """
    Async context manager for the execution environment lifecycle.
    
    A Sandbox represents an isolated execution environment (e.g., a Docker
    container or cloud sandbox). A sandbox can spawn multiple REPL sessions.
    
    Infrastructure errors during sandbox creation/destruction should raise
    SandboxInfrastructureError so callers can retry with backoff.
    
    Usage:
        async with IPyBoxSandbox(tag="ipybox:solver") as sandbox:
            async with sandbox.create_repl() as repl:
                result = await repl.execute("print('hello')")
    """
    
    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        Create and start the sandbox environment.
        
        Raises:
            Infrastructure errors are propagated as-is for caller to handle.
        """
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Destroy and clean up the sandbox environment.
        
        Raises:
            Infrastructure errors are propagated as-is for caller to handle.
        """
        pass
    
    @abstractmethod
    def create_repl(self) -> REPL:
        """
        Create a new REPL for this sandbox.
        
        The returned REPL should be used as an async context manager.
        
        Returns:
            A REPL instance (not yet entered).
        """
        pass

    @classmethod
    async def run_cells(
            cls, 
            cells: list[str], 
            timeout: int, 
            base_delay: int, 
            max_delay: int,
            delay_multiplier: float,
            max_backoff_retries: int,
            **kwargs
            ) -> list[ExecutionResult]:
        """
        Run a list of cells in the sandbox. If any infrastructure error (other than sandbox deletion error) occurs, we will retry with exponential backoff.
        
        Args:
            cells: List of cells to run.
            timeout: Timeout in seconds for each cell execution.
            base_delay: Base delay in seconds between retries.
            max_delay: Maximum delay in seconds between retries.
            delay_multiplier: Multiplier for exponential backoff.
            max_backoff_retries: Maximum number of retries for infrastructure errors.
            kwargs: Keyword arguments to pass to the sandbox class constructor.
            
        Returns:
            List of ExecutionResult objects.
        """

        delay = base_delay
        backoff_attempt = 0
        while True:
            results = []
            try:
                async with cls(**kwargs) as sandbox:
                    async with sandbox.create_repl() as repl:
                        for cell in cells:
                            results.append(await repl.execute(code=cell, timeout=timeout))
            except SandboxInfrastructureError as e:
                if backoff_attempt < max_backoff_retries:
                    backoff_attempt += 1
                    infra_logger.exception("Sandbox infrastructure error, retrying with backoff (attempt %d/%d, delay %.1fs)", backoff_attempt, max_backoff_retries, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * delay_multiplier, max_delay)
                    continue
                else:
                    infra_logger.exception(f"Failed to run cells after {max_backoff_retries} retries (for infrastructure error)")
                    raise MaxRetriesExceeded(f"Failed to run cells after {max_backoff_retries} retries (for infrastructure error)") from e
            else:
                break
        
        return results