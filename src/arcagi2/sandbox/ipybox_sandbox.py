"""
IPyBox implementation of the Sandbox interface.

Wraps the ipybox library to provide Docker-based code execution sandboxes.
"""

import asyncio
import logging
from typing import Self

from ipybox import ExecutionClient, ExecutionContainer, ExecutionError as IPyBoxExecutionError

from arcagi2.sandbox.base import ExecutionError, ExecutionResult, REPL, Sandbox
from arcagi2.sandbox.exceptions import SandboxInfrastructureError, ExecutionTimeoutError
from arcagi2.utils.logging_utils import infra_logger


logger = logging.getLogger(__name__)


class IPyBoxREPL(REPL):
    """
    REPL implementation using ipybox's ExecutionClient.
    """
    
    def __init__(self, executor_port):
        self._executor_port = executor_port
        self._client = None
    
    async def __aenter__(self) -> Self:
        try:
            self._client = ExecutionClient(port=self._executor_port)
            await self._client.__aenter__()
        except Exception as e:
            raise SandboxInfrastructureError(f"Failed to create REPL client: {e}") from e
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client is not None:
            try:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.warning(f"Failed to cleanup REPL client: {e}")
            finally:
                self._client = None
    
    async def execute(self, code: str, timeout: float) -> ExecutionResult:
        logger.info(f"Executing code:\n{code}")
        
        try:
            result = await self._client.execute(code=code, timeout=timeout)
        except IPyBoxExecutionError as e:
            # User code failure - convert to our ExecutionError
            raise ExecutionError(trace=e.trace)
        except asyncio.TimeoutError:
            # Convert to our ExecutionTimeoutError for consistent handling
            raise ExecutionTimeoutError(f"Code execution timed out after {timeout} seconds")
        except Exception as e:
            # Other errors are infrastructure failures
            raise SandboxInfrastructureError(f"Code execution failed: {e}") from e
        
        # Convert ipybox's result to our ExecutionResult
        return ExecutionResult(text=result.text)


class IPyBoxSandbox(Sandbox):
    """
    Sandbox implementation using ipybox's ExecutionContainer (Docker-based).
    
    Usage:
        async with IPyBoxSandbox(tag="ipybox:solver") as sandbox:
            async with sandbox.create_repl() as repl:
                result = await repl.execute("print('hello')")
    """
    
    def __init__(self, tag: str):
        self.tag = tag
        self._container = None
    
    async def __aenter__(self) -> Self:
        try:
            logger.info(f"Creating IPyBox container with tag: {self.tag}")
            self._container = ExecutionContainer(tag=self.tag)
            await self._container.__aenter__()
            logger.info("IPyBox container ready")
        except Exception as e:
            raise SandboxInfrastructureError(f"Failed to create sandbox: {e}") from e
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._container is not None:
            try:
                await self._container.__aexit__(exc_type, exc_val, exc_tb)
                logger.info("IPyBox container stopped")
            except Exception as e:
                infra_logger.warning(f"Failed to delete IPyBox container: {e}")
            finally:
                self._container = None
    
    def create_repl(self) -> IPyBoxREPL:
        if self._container is None:
            raise RuntimeError("Sandbox not initialized. Use 'async with' context manager.")
        return IPyBoxREPL(self._container.executor_port)
