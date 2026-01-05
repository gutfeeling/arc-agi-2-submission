"""
Daytona-based execution client that provides an IPyBox-compatible interface.

This module wraps the Daytona SDK's AsyncCodeInterpreter to provide the same
interface as ipybox, making it a drop-in replacement. Uses IPython's
InteractiveShellEmbed for proper REPL behavior including magic commands.

Exports (drop-in replacements for ipybox):
    - ExecutionClient: Stateful code execution client using Daytona sandboxes
    - ExecutionContainer: Manages Daytona sandbox lifecycle
    - ExecutionResult: Result object with .text attribute
    - ExecutionError: Exception with .trace attribute

Usage:
    async with ExecutionContainer(api_key="...") as container:
        async with ExecutionClient(sandbox=container.sandbox) as client:
            result = await client.execute(code="print('hello')")

References:
    - https://www.daytona.io/docs/en/python-sdk/async/async-code-interpreter/
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

# Import Daytona SDK
try:
    from daytona import AsyncDaytona, DaytonaConfig, Image, CreateSandboxFromImageParams
except ImportError:
    AsyncDaytona = None
    DaytonaConfig = None
    Image = None
    CreateSandboxFromImageParams = None

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution, mimics ipybox's result interface."""
    text: Optional[str]


class ExecutionError(Exception):
    """Exception raised when code execution fails, mimics ipybox's ExecutionError."""
    def __init__(self, trace: str):
        self.trace = trace
        super().__init__(trace)


class ExecutionClient:
    """
    Async context manager that provides a stateful code execution interface
    using IPython's InteractiveShellEmbed for proper REPL behavior.
    """

    # IPython shell setup code - ensure cwd is in sys.path for importing from files writtten using %writefile (coverage report needs that)
    _IPYTHON_SETUP = """import sys
import os
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from IPython.terminal.embed import InteractiveShellEmbed
_ipython_shell = InteractiveShellEmbed()
"""

    def __init__(self, sandbox=None, port=None):
        # Support IPyBox-style usage: ExecutionClient(port=container.executor_port)
        if sandbox is None and port is not None and hasattr(port, "code_interpreter"):
            sandbox = port
        self.sandbox = sandbox
        self._context = None
        self._ipython_ready = False

    async def __aenter__(self):
        if self.sandbox is not None:
            self._context = await self.sandbox.code_interpreter.create_context()
            
            # Set up IPython shell
            logger.info("Setting up IPython shell")
            result = await self.sandbox.code_interpreter.run_code(
                code=self._IPYTHON_SETUP,
                context=self._context,
                timeout=60,
            )
            
            if result.error is not None:
                raise ExecutionError(
                    trace=f"Failed to set up IPython: {result.error.traceback or result.error.value}"
                )
            
            self._ipython_ready = True
            logger.info("IPython shell ready")
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context is not None:
            try:
                await self.sandbox.code_interpreter.delete_context(self._context)
            except Exception as e:
                logger.warning(f"Failed to delete interpreter context: {e}")
            self._context = None
            self._ipython_ready = False

    async def execute(self, code: str, timeout: int = 120) -> ExecutionResult:
        if self.sandbox is None:
            raise RuntimeError("ExecutionClient requires a sandbox.")

        if not self._ipython_ready:
            raise RuntimeError("IPython shell not initialized. Use 'async with' context manager.")

        logger.info(f"Code is:\n{code}")

        # Just call shell.run_cell() with the code
        run_code = f"_ipython_shell.run_cell({repr(code)})"

        try:
            result = await self.sandbox.code_interpreter.run_code(
                code=run_code,
                context=self._context,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise ExecutionError(trace=str(e))

        if result.error is not None:
            trace = result.error.traceback or f"{result.error.name}: {result.error.value}"
            raise ExecutionError(trace=trace)

        # Combine stdout and stderr
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(result.stderr)
        
        text = "\n".join(output_parts).strip() if output_parts else None
        
        # Ensure empty/whitespace-only output becomes None
        if not text:
            text = None
        
        # Strip IPython's Out[n]: prefix from output
        if text:
            text = re.sub(r"^Out\[\d+\]: ", "", text)
        
        return ExecutionResult(text=text)


class ExecutionContainer:
    """Container class that manages a Daytona sandbox lifecycle."""
    
    # Declarative image with ipykernel and solver dependencies
    _IMAGE = (
        Image.debian_slim("3.12")
        .pip_install([
            "ipykernel",
            "numpy",
            "scipy",
            "shapely",
            "networkx",
            "scikit-image",
            "more-itertools",
            "pillow",
            "matplotlib",
            "python-constraint",
            "ortools",
            "z3-solver",
            "coverage",
        ])
    ) if Image is not None else None
    
    def __init__(self, tag=None, api_key=None, target=None, **kwargs):
        self.tag = tag
        self.api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        self.target = target or os.environ.get("DAYTONA_TARGET")
        self._daytona = None
        self._sandbox = None
        self.executor_port = None
    
    @property
    def sandbox(self):
        return self._sandbox
    
    async def __aenter__(self):
        if AsyncDaytona is None or DaytonaConfig is None:
            raise ImportError("Daytona SDK not installed. Install with: pip install daytona")
        
        if not self.api_key:
            raise ValueError("Daytona API key required. Set DAYTONA_API_KEY environment variable.")
        
        config = DaytonaConfig(api_key=self.api_key)
        if self.target:
            config.target = self.target
        
        self._daytona = AsyncDaytona(config)
        
        # Create sandbox with custom image that has ipykernel and solver deps
        # 5 min timeout for package installation
        self._sandbox = await self._daytona.create(
            CreateSandboxFromImageParams(image=self._IMAGE),
            timeout=300
        )
        self.executor_port = self._sandbox
        
        logger.info(f"Created Daytona sandbox: {self._sandbox.id}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._sandbox is not None:
            try:
                await self._sandbox.delete()
                logger.info("Deleted Daytona sandbox")
            except Exception as e:
                logger.warning(f"Failed to delete sandbox: {e}")
            self._sandbox = None
        self._daytona = None
