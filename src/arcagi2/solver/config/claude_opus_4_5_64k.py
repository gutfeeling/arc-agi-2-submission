from arcagi2.api.clients import AsyncMessagesAPIClient
from arcagi2.api.providers import ANTHROPIC_API_PROVIDER
from arcagi2.solver.config.base import (
    AgenticCodingConfig, 
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
    model="claude-opus-4-5-20251101",
    api_provider=ANTHROPIC_API_PROVIDER,
    client_kwargs={
        "max_retries": 0
    },
    system_prompt_path=PROMPTS_FOLDER / "system_prompt.txt",
    raw_request_kwargs={
        "thinking": {
            "type": "enabled", 
            "budget_tokens": 200_000
        },
        "betas": ["interleaved-thinking-2025-05-14"],
        "max_tokens": 64000
    },
    tools=[REPLToolWithProtection(name="python", timeout=120, protected_variables=["puzzle"])],
    max_retries=2,
    sleep=0,
    initial_code_timeout=120,
    cache_ttl="5m"
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

CLAUDE_OPUS_4_5_64K_AGENTIC_CODING_CONFIG = AgenticCodingConfig(
    sandbox_cls=IPYBOX_SANDBOX_CLS,
    sandbox_kwargs=IPYBOX_SANDBOX_KWARGS,
    call_config=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_IPYBOX_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "agentic_coding_solver.txt"
    ),
)

CLAUDE_OPUS_4_5_64K_DAYTONA_AGENTIC_CODING_CONFIG = AgenticCodingConfig(
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    call_config=AsyncMessagesAPIClient.MessagesAPICallConfig(
        **_DAYTONA_COMMON_KWARGS,
        prompt_path=PROMPTS_FOLDER / "agentic_coding_solver.txt"
    ),
)