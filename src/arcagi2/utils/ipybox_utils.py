# Helpers using IPyBox
import contextlib
import logging
from typing import Union

import os

# USE_DAYTONA=True: Daytona cloud sandboxes, False (default): ipybox Docker
if os.getenv("USE_DAYTONA", "False") == "True":
    from arcagi2.tools.execution_client_daytona import ExecutionContainer, ExecutionClient, ExecutionResult
else:
    from ipybox import ExecutionContainer, ExecutionClient
    from ipybox.executor import ExecutionResult


logger = logging.getLogger(__name__)

async def ensure_container(stack: contextlib.AsyncExitStack, container_or_tag: Union[ExecutionContainer, str]) -> ExecutionContainer:
    if isinstance(container_or_tag, str):
        container = await stack.enter_async_context(ExecutionContainer(tag=container_or_tag))
    else:
        container = container_or_tag
    return container

async def run_cells_ipybox(cells: list[str], container_or_tag: Union[ExecutionContainer, str]) -> list[ExecutionResult]:
    async with contextlib.AsyncExitStack() as stack:
        container = await ensure_container(stack, container_or_tag)
        async with ExecutionClient(port=container.executor_port) as ipybox_client:
            results = []
            for cell in cells:
                logger.info(f"Executing cell:\n{cell}")
                result = await ipybox_client.execute(code=cell)
                result_text = result.text if result.text is not None else "Code executed successfully. No output was returned."
                logger.info(f"Result was:\n{result_text}")
                results.append(result)
    return results
