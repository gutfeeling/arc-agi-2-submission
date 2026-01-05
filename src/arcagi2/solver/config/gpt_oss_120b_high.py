from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import VLLM_API_PROVIDER
from arcagi2.solver.config.base import SystemConfig, PROMPTS_FOLDER
from arcagi2.tools.code_interpreter_ipybox import IPyBoxWithProtection


INTERLEAVED_THINKING_SOLVER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    model="openai/gpt-oss-120b",
    api_provider=VLLM_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "interleaved_thinking_solver.txt",
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["puzzle"], name="python")],
    raw_request_kwargs = {
        "reasoning": {
            "effort": "high",
        }
    },
    max_retries=2,
    stateful=False
)

SOFT_VERIFIER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    model="openai/gpt-oss-120b",
    api_provider=VLLM_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "soft_verifier.txt",
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["train"], name="python")],
    raw_request_kwargs = {
        "reasoning": {
            "effort": "high",
        },
    },
    max_retries=2,
    stateful=False
)

GENERALIZER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    model="openai/gpt-oss-120b",
    api_provider=VLLM_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "generalizer.txt",
    system_prompt_path = PROMPTS_FOLDER / "system_prompt.txt",
    tools=[IPyBoxWithProtection(protected_variables=["puzzle"], name="python")],
    raw_request_kwargs = {
        "reasoning": {
            "effort": "high",
        }
    },
    max_retries=2,
    stateful=False
)

GPT_OSS_120B_HIGH_SYSTEM_CONFIG = SystemConfig(
    code_sandbox_container_tag="ipybox:solver",
    interleaved_thinking_solver=INTERLEAVED_THINKING_SOLVER,
    soft_verifier=SOFT_VERIFIER,
    generalizer=GENERALIZER,
)