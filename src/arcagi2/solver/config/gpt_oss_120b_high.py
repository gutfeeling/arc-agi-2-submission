from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import VLLM_API_PROVIDER
from arcagi2.solver.config.base import (
    AgenticCodingConfig, 
    PROMPTS_FOLDER, 
    IPYBOX_SANDBOX_CLS, 
    IPYBOX_SANDBOX_KWARGS,
    DAYTONA_SANDBOX_CLS,
    DAYTONA_SANDBOX_KWARGS,
)
from arcagi2.tools.repl_tool import REPLToolWithProtection


_COMMON_KWARGS = dict(
    model="openai/gpt-oss-120b",
    api_provider=VLLM_API_PROVIDER,
    system_prompt_path=PROMPTS_FOLDER / "system_prompt.txt",
    client_kwargs={},
    raw_request_kwargs={
        "reasoning": {
            "effort": "high"
        }
    },
    tools=[REPLToolWithProtection(name="python", timeout=120, protected_variables=["puzzle"])],
    max_retries=2,
    sleep=0,
    initial_code_timeout=120,
    stateful=False,
)

_IPYBOX_COMMON_KWARGS = dict(
    **_COMMON_KWARGS,
    sandbox_cls=IPYBOX_SANDBOX_CLS,
    sandbox_kwargs=IPYBOX_SANDBOX_KWARGS,
)

_DAYTONA_COMMON_KWARGS = dict(
    **_COMMON_KWARGS,
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
)

GPT_OSS_120B_HIGH_AGENTIC_CODING_CONFIG = AgenticCodingConfig(
    sandbox_cls=IPYBOX_SANDBOX_CLS,
    sandbox_kwargs=IPYBOX_SANDBOX_KWARGS,
    call_config=AsyncResponsesAPIClient.ResponsesAPICallConfig(
        **_IPYBOX_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "agentic_coding_solver.txt"
    )
)

GPT_OSS_120B_HIGH_DAYTONA_AGENTIC_CODING_CONFIG = AgenticCodingConfig(
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    call_config=AsyncResponsesAPIClient.ResponsesAPICallConfig(
        **_DAYTONA_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "agentic_coding_solver.txt"
    )
)