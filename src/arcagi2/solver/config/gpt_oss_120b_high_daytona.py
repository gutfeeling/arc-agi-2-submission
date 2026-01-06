from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import VLLM_API_PROVIDER
from arcagi2.sandbox.daytona_sandbox import DaytonaSandbox
from arcagi2.solver.config.base import InterleavedThinkingConfig, PROMPTS_FOLDER, DAYTONA_SANDBOX_CLS, DAYTONA_SANDBOX_KWARGS
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
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    initial_code_timeout=120,
    stateful=False,
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

GPT_OSS_120B_HIGH_DAYTONA_SYSTEM_CONFIG = InterleavedThinkingConfig(
    sandbox_cls=DAYTONA_SANDBOX_CLS,
    sandbox_kwargs=DAYTONA_SANDBOX_KWARGS,
    interleaved_thinking_solver=INTERLEAVED_THINKING_SOLVER,
    soft_verifier=SOFT_VERIFIER,
    generalizer=GENERALIZER,
)