from httpx import Timeout

from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import VLLM_API_PROVIDER
from arcagi2.solver.config.base import BaselineConfig, PROMPTS_FOLDER


GPT_OSS_120B_HIGH_BASELINE_CONFIG = BaselineConfig(
    call_config=AsyncResponsesAPIClient.ResponsesAPICallConfig(
        model="openai/gpt-oss-120b",
        api_provider=VLLM_API_PROVIDER,
        prompt_path = PROMPTS_FOLDER / "plain_cot_solver.txt",
        system_prompt_path=None,
        client_kwargs = {
            "timeout": Timeout(timeout=60.0, connect=5.0),    # In background mode, retrieve should return fast
        },
        raw_request_kwargs = {
            "reasoning": {
                "effort": "high",
            },
            "background": True,
            # Background mode requires store=True, but on VLLM this currently causes a memory leak in RAM. We have to live with that.
            # Must start VLLM with env var VLLM_ENABLE_RESPONSES_API_STORE=1
            "store": True   
        },
        background_mode_polling_interval=2,
    )
)