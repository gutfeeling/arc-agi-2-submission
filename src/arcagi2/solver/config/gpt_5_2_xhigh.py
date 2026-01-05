from httpx import Timeout

from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import OPENAI_API_PROVIDER
from arcagi2.solver.config.base import SystemConfig, PROMPTS_FOLDER
from arcagi2.tools.code_interpreter_ipybox import IPyBoxWithProtection


INTERLEAVED_THINKING_SOLVER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    model="gpt-5.2",
    api_provider=OPENAI_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "interleaved_thinking_solver.txt",
    client_kwargs = {
        "timeout": Timeout(timeout=60.0, connect=5.0),    # In background mode, retrieve should return fast
    },
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["puzzle"], name="python")],
    raw_request_kwargs = {
        "reasoning": {
            "effort": "xhigh",
        },
        "background": True,
        # Background mode requires store=True
        "store": True   
    },
    max_retries=2
)

SOFT_VERIFIER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    model="gpt-5.2",
    api_provider=OPENAI_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "soft_verifier.txt",
    client_kwargs = {
        "timeout": Timeout(timeout=60.0, connect=5.0),    # In background mode, retrieve should return fast
    },
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["train"], name="python")],
    raw_request_kwargs = {
        "reasoning": {
            "effort": "xhigh",
        },
        "background": True,
        # Background mode requires store=True
        "store": True   
    },
    max_retries=2
)

GENERALIZER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    model="gpt-5.2",
    api_provider=OPENAI_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "generalizer.txt",
    client_kwargs = {
        "timeout": Timeout(timeout=60.0, connect=5.0),    # In background mode, retrieve should return fast
    },
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["puzzle"], name="python")],
    raw_request_kwargs = {
        "reasoning": {
            "effort": "xhigh",
        },
        "background": True,
        # Background mode requires store=True
        "store": True   
    },
    max_retries=2
)

GPT_5_2_XHIGH_SYSTEM_CONFIG = SystemConfig(
    code_sandbox_container_tag="ipybox:solver",
    interleaved_thinking_solver=INTERLEAVED_THINKING_SOLVER,
    soft_verifier=SOFT_VERIFIER,
    generalizer=GENERALIZER,
)