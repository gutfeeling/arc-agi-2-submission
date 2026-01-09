"""
Daytona implementation of the Sandbox interface.

Wraps the Daytona SDK to provide cloud-based code execution sandboxes
using IPython for rich REPL behavior.
"""

import logging
import os
import re
from typing import Optional, Self

from daytona import AsyncDaytona, DaytonaConfig, DaytonaError, DaytonaTimeoutError, Image, CreateSandboxFromImageParams, Resources

from arcagi2.sandbox.base import ExecutionError, ExecutionResult, REPL, Sandbox
from arcagi2.sandbox.exceptions import SandboxInfrastructureError, ExecutionTimeoutError
from arcagi2.utils.logging_utils import infra_logger


logger = logging.getLogger(__name__)


class DaytonaREPL(REPL):
    """
    REPL implementation using Daytona's AsyncCodeInterpreter with IPython.
    
    Uses IPython's InteractiveShellEmbed for proper REPL behavior including
    magic commands and rich error formatting.
    """
    
    # Setup code: ensure cwd is in sys.path (needed for %writefile imports)
    # and initialize IPython shell
    _IPYTHON_SETUP = """import sys
import os
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from IPython.terminal.embed import InteractiveShellEmbed
_ipython_shell = InteractiveShellEmbed()
"""
    
    def __init__(self, sandbox: "DaytonaSandbox"):
        """
        Args:
            sandbox: The Daytona sandbox object with code_interpreter attribute.
        """
        self._sandbox = sandbox
        self._context = None
        self._ipython_ready = False
    
    async def __aenter__(self) -> Self:
        try:
            self._context = await self._sandbox.code_interpreter.create_context()
            
            # Set up IPython shell
            logger.info("Setting up IPython shell")
            result = await self._sandbox.code_interpreter.run_code(
                code=self._IPYTHON_SETUP,
                context=self._context,
                timeout=60,    # Setting a reasonable fixed timeout value for this
            )
            
            if result.error is not None:
                raise SandboxInfrastructureError(
                    f"Failed to set up IPython: {result.error.traceback or result.error.value}"
                )
            
            self._ipython_ready = True
            logger.info("IPython shell ready")
        except SandboxInfrastructureError:
            raise
        except DaytonaError as e:
            raise SandboxInfrastructureError(f"Failed to create REPL context: {e}") from e
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context is not None:
            try:
                await self._sandbox.code_interpreter.delete_context(self._context)
            except DaytonaError as e:
                logger.warning(f"Failed to delete interpreter context: {e}")
            finally:
                self._context = None
                self._ipython_ready = False
    
    async def execute(self, code: str, timeout: float) -> ExecutionResult:
        if not self._ipython_ready:
            raise RuntimeError("IPython shell not initialized. Use 'async with' context manager.")
        
        logger.info(f"Executing code:\n{code}")
        
        # Wrap code in IPython's run_cell with error detection.
        # If the cell fails, we re-raise the exception so that:
        # 1. result.error gets populated (for detection)
        # 2. The nice IPython traceback appears in stdout
        wrapper_code = f"""_cell_result = _ipython_shell.run_cell({repr(code)})
if not _cell_result.success:
    _err = _cell_result.error_in_exec or _cell_result.error_before_exec
    if _err is not None:
        raise _err
"""
        
        try:
            result = await self._sandbox.code_interpreter.run_code(
                code=wrapper_code,
                context=self._context,
                timeout=timeout,
            )
        except DaytonaTimeoutError:
            # Convert to our ExecutionTimeoutError for consistent handling
            raise ExecutionTimeoutError(f"Code execution timed out after {timeout} seconds")
        except DaytonaError as e:
            # Infrastructure error during code execution
            raise SandboxInfrastructureError(f"Code execution failed: {e}") from e
        
        # Check for user code errors
        if result.error is not None:
            # Use stdout for the rich IPython-formatted traceback
            trace = result.stdout or result.error.traceback or f"{result.error.name}: {result.error.value}"
            raise ExecutionError(trace=trace)
        
        # Combine stdout and stderr for successful execution
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(result.stderr)
        
        text = "\n".join(output_parts).strip() if len(output_parts) > 0 else None
        
        # Ensure empty/whitespace-only output becomes None
        if not text:
            text = None
        
        # Strip IPython's Out[n]: prefix from output
        if text:
            text = re.sub(r"^Out\[\d+\]: ", "", text)
        
        return ExecutionResult(text=text)


class DaytonaSandbox(Sandbox):
    """
    Sandbox implementation using Daytona's cloud-based sandboxes.
    
    Usage:
        from daytona import Image
        
        image = Image.debian_slim("3.12").pip_install(["numpy"])
        resources = Resources(cpu=1, memory=1, disk=3)
        
        async with DaytonaSandbox(image=image, resources=resources, creation_timeout=300) as sandbox:
            async with sandbox.create_repl() as repl:
                result = await repl.execute("print('hello')")
    """
    
    def __init__(self, image: Image, resources: Resources, creation_timeout: int, auto_stop_interval: int, api_key: Optional[str] = None, target: Optional[str] = None):
        """
        Args:
            image: Daytona Image object (declarative image specification).
            resources: Resources for the sandbox.
            creation_timeout: Timeout in seconds for sandbox creation.
            auto_stop_interval: Interval in minutes after which the sandbox will be stopped.
            api_key: Daytona API key (defaults to DAYTONA_API_KEY env var).
            target: Daytona target (defaults to DAYTONA_TARGET env var).
        """
        self.image = image
        self.resources = resources
        self.creation_timeout = creation_timeout
        self.auto_stop_interval = auto_stop_interval
        self.api_key = api_key or os.environ.get("DAYTONA_API_KEY")
        if self.api_key is None:
            raise ValueError("Daytona API key is required. Set DAYTONA_API_KEY environment variable.")
        self.target = target or os.environ.get("DAYTONA_TARGET")
        if isinstance(target, str) and target not in ["eu", "us"]:
            raise ValueError(f"Invalid Daytona target. Must be 'eu' or 'us'. Got {target}")
        
        # Set up config and client (but don't connect yet)
        config = DaytonaConfig(api_key=self.api_key)
        if self.target is not None:
            config.target = self.target
        self._daytona = AsyncDaytona(config)
        
        # Workaround for Daytona SDK bug: _api_clients is a class variable shared across
        # all AsyncDaytona instances. When one instance closes, it closes all clients.
        # Fix: give this instance its own isolated client list, and remove from class list
        # to prevent memory leak (class list never gets cleaned up).
        # Already fixed in Daytona main branch, but they haven't released yet.
        # See: https://github.com/daytonaio/daytona/commit/13542e5677668ed03279ac27596db3e01e22be7d
        # TODO: Remove after upgrading to daytona SDK version with the fix
        self._daytona._api_clients = [self._daytona._api_client]
        try:
            AsyncDaytona._api_clients.remove(self._daytona._api_client)
        except ValueError:
            pass  # Already removed or not present
        
        self._sandbox = None
    
    async def __aenter__(self) -> Self:
        try:
            await self._daytona.__aenter__()
            
            logger.info(f"Creating Daytona sandbox (timeout={self.creation_timeout}s)")
            self._sandbox = await self._daytona.create(
                CreateSandboxFromImageParams(
                    image=self.image,
                    resources=self.resources,
                    auto_stop_interval=self.auto_stop_interval,
                    auto_delete_interval=0    # Sandbox will be deleted immediately after stopping
                ),
                timeout=self.creation_timeout
            )
            logger.info(f"Daytona sandbox ready: {self._sandbox.id}")
        except DaytonaError as e:
            # All Daytona errors derive from DaytonaError
            raise SandboxInfrastructureError(f"Failed to create Daytona sandbox: {e}") from e
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._sandbox is not None:
            try:
                await self._sandbox.delete()
                logger.info("Daytona sandbox deleted")
            except DaytonaError as e:
                infra_logger.warning(f"Failed to delete Daytona sandbox: {e}")
            finally:
                self._sandbox = None
        
        # Always exit the Daytona client context
        try:
            await self._daytona.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            logger.warning(f"Failed to close Daytona client: {e}")
    
    def create_repl(self) -> DaytonaREPL:
        if self._sandbox is None:
            raise RuntimeError("Sandbox not initialized. Use 'async with' context manager.")
        return DaytonaREPL(self._sandbox)
