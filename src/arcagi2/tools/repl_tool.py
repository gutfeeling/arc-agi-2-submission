import logging
import re

from arcagi2.sandbox.base import ExecutionError, REPL
from arcagi2.sandbox.exceptions import ExecutionTimeoutError

from arcagi2.tools.base import Tool


logger = logging.getLogger(__name__)


class REPLTool(Tool):
    NAME = "python"
    DESCRIPTION = "Executes code using an IPython kernel and returns the output."
    PARAMETERS = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The code to execute"}
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(self, timeout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"timeout={self.timeout!r}, "
            f"strict={self.strict!r}, "
            f"include_strict={self.include_strict!r})"
        )

    async def run(self, repl: REPL, code: str) -> str:
        logger.info(f"Code is:\n{code}")
        try:
            result = await repl.execute(code=code, timeout=self.timeout)
            # When no output is returned, `result.text` is `None`. We need to return a string. Best to return a model-friendly natural language explanation.
            result = (
                result.text
                if result.text is not None
                else "Code executed successfully. No output was returned."
            )
        except ExecutionError as e:
            # trace returned by IPyBox might sometimes not be a string.
            result = (
                e.trace if isinstance(e.trace, str) else "Code execution failed."
            )    # Make sure we return a string.
        except ExecutionTimeoutError:
            result = f"Code execution terminated forcefully after timeout {self.timeout} seconds was exhausted."
        logger.info(f"Code execution result:\n{result}") 
        return result

class REPLToolWithProtection(REPLTool):
    def __init__(self, protected_variables=["puzzle"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protected_variables = protected_variables  

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"timeout={self.timeout!r}, "
            f"protected_variables={self.protected_variables!r}, "
            f"strict={self.strict!r}, "
            f"include_strict={self.include_strict!r})"
        )

    async def run(self, repl: REPL, code: str) -> str:
        for protected_variable in self.protected_variables:
            if re.search(rf"{protected_variable}\s*=", code):
                logger.info(f"Detected code trying to redefine the existing {protected_variable} variable")
                logger.info(f"Code:\n{code}")
                return f"Code is trying to redefine the existing {protected_variable} variable, which is not allowed. Use the existing {protected_variable} variable. Please remove the offending line and try again."
            if re.search(rf"(?m)^{protected_variable}(?:\s*\n|$)", code) or re.search(rf"print\({protected_variable}", code):
                logger.info(f"Detected code trying to print the {protected_variable} variable")
                logger.info(f"Code:\n{code}")
                return f"Code is trying to print the {protected_variable} variable, which is not allowed because it will lead to a very large and wasteful print. Please remove the offending line and try again."
        return await super().run(repl, code)
