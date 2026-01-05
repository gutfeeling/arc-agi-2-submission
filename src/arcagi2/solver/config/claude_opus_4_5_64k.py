from arcagi2.api.clients import AsyncMessagesAPIClient
from arcagi2.api.providers import ANTHROPIC_API_PROVIDER
from arcagi2.solver.config.base import SystemConfig, PROMPTS_FOLDER
from arcagi2.tools.code_interpreter_ipybox import IPyBoxWithProtection


INTERLEAVED_THINKING_SOLVER = AsyncMessagesAPIClient.MessagesAPICallConfig(
    model="claude-opus-4-5-20251101",
    api_provider=ANTHROPIC_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "interleaved_thinking_solver.txt",
    # Our Anthropic client streams by default. For streaming requests, we handle all retries explicitly in our code.
    # So turning off automatic retries here.
    client_kwargs = {
        "max_retries": 0,
    },
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["puzzle"], name="python")],
    raw_request_kwargs = {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 200_000,
        },
        "betas": ["interleaved-thinking-2025-05-14"],
        "max_tokens": 64000
    },
    max_retries=2
)

SOFT_VERIFIER = AsyncMessagesAPIClient.MessagesAPICallConfig(
    model="claude-opus-4-5-20251101",
    api_provider=ANTHROPIC_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "soft_verifier.txt",
    client_kwargs = {
        "max_retries": 0,
    },
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["train"], name="python")],
    raw_request_kwargs = {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 200_000,
        },
        "betas": ["interleaved-thinking-2025-05-14"],
        "max_tokens": 64000,
    },
    max_retries=2
)

GENERALIZER = AsyncMessagesAPIClient.MessagesAPICallConfig(
    model="claude-opus-4-5-20251101",
    api_provider=ANTHROPIC_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "generalizer.txt",
    client_kwargs = {
        "max_retries": 0,
    },
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["puzzle"], name="python")],
    raw_request_kwargs = {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 200_000,
        },
        "betas": ["interleaved-thinking-2025-05-14"],
        "max_tokens": 64000,
    },
    max_retries=2
)

CLAUDE_OPUS_4_5_64K_SYSTEM_CONFIG = SystemConfig(
    code_sandbox_container_tag="ipybox:solver",
    interleaved_thinking_solver=INTERLEAVED_THINKING_SOLVER,
    soft_verifier=SOFT_VERIFIER,
    generalizer=GENERALIZER,
)

