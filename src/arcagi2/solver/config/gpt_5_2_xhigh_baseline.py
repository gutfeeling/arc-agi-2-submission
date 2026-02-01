from httpx import Timeout

from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import OPENAI_API_PROVIDER
from arcagi2.solver.config.base import PlainCOTConfig, PROMPTS_FOLDER


GPT_5_2_XHIGH_PLAIN_COT_CONFIG = PlainCOTConfig(
    call_config=AsyncResponsesAPIClient.ResponsesAPICallConfig(
        model="gpt-5.2-2025-12-11",
        api_provider=OPENAI_API_PROVIDER,
        prompt_path = PROMPTS_FOLDER / "plain_cot_solver.txt",
        system_prompt_path=None,
        client_kwargs = {
            "timeout": Timeout(timeout=60.0, connect=5.0),    # In background mode, retrieve should return fast
        },
        raw_request_kwargs = {
            "reasoning": {
                "effort": "xhigh",
            },
            "background": True,
            # Background mode requires store=True
            "store": True   
        },
        background_mode_polling_interval=2,
    )
)