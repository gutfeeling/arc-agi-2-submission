from asyncio import TimeoutError
import logging
import re

# hack to be able to import this file even if ipybox is not installed
# this is a valid use case when we need the tool spec but not run it
try:
    import os
    # USE_DAYTONA=True: Daytona cloud sandboxes, False (default): ipybox Docker
    if os.getenv("USE_DAYTONA", "False") == "True":
        from arcagi2.tools.execution_client_daytona import ExecutionError
    else:
        from ipybox import ExecutionError
except ImportError:
    pass

from arcagi2.tools.base import Tool

logger = logging.getLogger(__name__)


class IPyBox(Tool):
    NAME = "execute"
    DESCRIPTION = "Executes code using an IPython kernel and returns the output."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The code to execute"}
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    async def run(self, client, code, timeout=120):
        logger.info(f"Code is:\n{code}")
        try:
            result = await client.execute(code=code, timeout=timeout)
            # When no output is returned, `result.text` is `None`. We need to return a string. Best to return a model-friendly natural language explanation.
            result = (
                result.text
                if result.text is not None
                else "Code executed successfully. No output was returned."
            )
        except ExecutionError as e:
            result = (
                e.trace if isinstance(e.trace, str) else "Code execution failed."
            )    # Make sure we return a string.
        except TimeoutError as e:
            result = f"Code execution terminated forcefully after timeout {timeout} seconds was exhausted."
        logger.info(f"Code execution result:\n{result}") 
        return result

class IPyBoxWithProtection(IPyBox):
    def __init__(self, protected_variables=["puzzle"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protected_variables = protected_variables

    async def run(self, client, code, *args, **kwargs):
        for protected_variable in self.protected_variables:
            if re.search(rf"{protected_variable}\s*=", code):
                logger.info(f"Detected code trying to redefine the existing {protected_variable} variable")
                logger.info(f"Code:\n{code}")
                return f"Code is trying to redefine the existing {protected_variable} variable, which is not allowed. Use the existing {protected_variable} variable. Please remove the offending line and try again."
            if re.search(rf"(?m)^{protected_variable}(?:\s*\n|$)", code) or re.search(rf"print\({protected_variable}", code):
                logger.info(f"Detected code trying to print the {protected_variable} variable")
                logger.info(f"Code:\n{code}")
                return f"Code is trying to print the {protected_variable} variable, which is not allowed because it will lead to a very large and wasteful print. Please remove the offending line and try again."
        return await super().run(client, code, *args, **kwargs)
