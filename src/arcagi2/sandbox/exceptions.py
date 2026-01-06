"""
Sandbox exception classes.

Exception hierarchy:
- SandboxInfrastructureError: Infrastructure failures (caller should retry by recreating another sandbox and trying from scratch)
- ExecutionError: User code failure (model is simply informed)
- ExecutionTimeoutError: Execution timeouts (model is simply informed)

Resource deletion errors are simply logged and ignored.
This is because the caller already has the results before this happens.
So they may actually choose to ignore the error and handle cleanup manually.

It's unclear what to do with this error in any case. It could be transient or permanent.
Wasting time retrying the deletion doesn't seem to be a good idea. It's probably just a one-off that can be ignored.
Creating a new sandbox and trying from scratch is not necessary since the caller already has the results.
"""


class SandboxInfrastructureError(Exception):
    """
    Raised when sandbox infrastructure fails.
    
    This includes:
    - Sandbox creation failures (Docker issues, cloud API errors)
    - REPL creation failures
    - Communication errors during code execution
    
    The caller may want to retry by recreating another sandbox and trying from scratch.
    
    Not included (separate exceptions):
    - ExecutionError: User code failures (model is simply informed)
    - ExecutionTimeoutError: Execution timeouts (model is simply informed)
    """
    pass


class ExecutionTimeoutError(Exception):
    """
    Raised when code execution times out.
    
    This is distinct from infrastructure errors - the sandbox is healthy,
    but the user's code took too long to execute.
    
    The model should be informed so it can adjust its approach.
    """
    pass

