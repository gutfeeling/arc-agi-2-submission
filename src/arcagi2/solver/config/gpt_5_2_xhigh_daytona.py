from httpx import Timeout

from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import OPENAI_API_PROVIDER
from arcagi2.solver.config.base import InterleavedThinkingConfig, PROMPTS_FOLDER, DAYTONA_SANDBOX_CLS, DAYTONA_SANDBOX_KWARGS
from arcagi2.tools.repl_tool import REPLToolWithProtection


_COMMON_KWARGS = dict(
    model="gpt-5.2",
    api_provider=OPENAI_API_PROVIDER,
    system_prompt_path=PROMPTS_FOLDER / "system_prompt.txt",
    client_kwargs={"timeout": Timeout(timeout=60.0, connect=5.0)},  # In background mode, retrieve should return fast
    raw_request_kwargs={
        "reasoning": {
            "effort": "xhigh"
        },
        "background": True,
        "store": True  # Background mode requires store=True
    },
    tools=[REPLToolWithProtection(name="python", timeout=120, protected_variables=["puzzle"])],
    max_retries=2,
    sleep=10,
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    initial_code_timeout=120,
    background_mode_polling_interval=2,
    stateful=True
)

INTERLEAVED_THINKING_SOLVER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    **_COMMON_KWARGS,
    prompt_path=PROMPTS_FOLDER / "interleaved_thinking_solver.txt"
)

SOFT_VERIFIER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    **_COMMON_KWARGS,
    prompt_path=PROMPTS_FOLDER / "soft_verifier.txt"
)

GENERALIZER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    **_COMMON_KWARGS,
    prompt_path=PROMPTS_FOLDER / "generalizer.txt"
)

GPT_5_2_XHIGH_DAYTONA_SYSTEM_CONFIG = InterleavedThinkingConfig(
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    interleaved_thinking_solver=INTERLEAVED_THINKING_SOLVER,
    soft_verifier=SOFT_VERIFIER,
    generalizer=GENERALIZER,
)