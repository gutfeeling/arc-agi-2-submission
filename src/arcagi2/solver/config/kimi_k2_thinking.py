from arcagi2.api.clients import AsyncChatCompletionsAPIClient
from arcagi2.api.providers import MOONSHOT_AI_API_PROVIDER
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
    model="kimi-k2-thinking",
    api_provider=MOONSHOT_AI_API_PROVIDER,
    system_prompt_path=PROMPTS_FOLDER / "system_prompt.txt",
    client_kwargs={
        "max_retries": 0
    },
    raw_request_kwargs={
        "temperature": 1.0,
        "max_tokens": 256_000,
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    },
    tools=[REPLToolWithProtection(name="python", timeout=120, protected_variables=["puzzle"])],
    max_retries=2,
    sleep=0,
    initial_code_timeout=120,
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

KIMI_K2_THINKING_AGENTIC_CODING_CONFIG = AgenticCodingConfig(
    sandbox_cls=IPYBOX_SANDBOX_CLS,
    sandbox_kwargs=IPYBOX_SANDBOX_KWARGS,
    call_config=AsyncChatCompletionsAPIClient.ChatCompletionsAPICallConfig(
        **_IPYBOX_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "agentic_coding_solver.txt"
    ),
)

KIMI_K2_THINKING_DAYTONA_AGENTIC_CODING_CONFIG = AgenticCodingConfig(
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    call_config=AsyncChatCompletionsAPIClient.ChatCompletionsAPICallConfig(
        **_DAYTONA_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "agentic_coding_solver.txt"
    ),
)