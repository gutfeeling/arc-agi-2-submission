from arcagi2.api.clients import AsyncMessagesAPIClient
from arcagi2.api.providers import MINIMAX_API_PROVIDER
from arcagi2.solver.config.base import (
    InterleavedThinkingConfig, 
    PROMPTS_FOLDER, 
    DAYTONA_SANDBOX_CLS,
    DAYTONA_SANDBOX_KWARGS,
    IPYBOX_SANDBOX_CLS,
    IPYBOX_SANDBOX_KWARGS,
)
from arcagi2.tools.repl_tool import REPLToolWithProtection


# Our Anthropic client streams by default. For streaming requests, we handle all retries explicitly in our code.
# So turning off automatic retries here.
_COMMON_KWARGS = dict(
    model="MiniMax-M2.1",
    api_provider=MINIMAX_API_PROVIDER,
    client_kwargs={
        "max_retries": 0
    },
    system_prompt_path=PROMPTS_FOLDER / "system_prompt.txt",
    raw_request_kwargs={
        "thinking": {
            "type": "enabled",
        },
        "max_tokens": 196_608
    },
    tools=[REPLToolWithProtection(name="python", timeout=120, protected_variables=["puzzle"])],
    max_retries=2,
    sleep=0,
    initial_code_timeout=120,
    cache_ttl="5m"    # Doesn't use the actual value. Always 5m. But we need to pass a non-None value for caching to happen.
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

MINIMAX_M2_1_SYSTEM_CONFIG = InterleavedThinkingConfig(
    sandbox_cls=IPYBOX_SANDBOX_CLS,
    sandbox_kwargs=IPYBOX_SANDBOX_KWARGS,
    interleaved_thinking_solver=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_IPYBOX_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "interleaved_thinking_solver.txt"
    ),
    soft_verifier=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_IPYBOX_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "soft_verifier.txt"
    ),
    generalizer=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_IPYBOX_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "generalizer.txt"
    ),
)

MINIMAX_M2_1_DAYTONA_SYSTEM_CONFIG = InterleavedThinkingConfig(
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    interleaved_thinking_solver=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_DAYTONA_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "interleaved_thinking_solver.txt"
    ),
    soft_verifier=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_DAYTONA_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "soft_verifier.txt"
    ),
    generalizer=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_DAYTONA_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "generalizer.txt"
    ),
)