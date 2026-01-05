from httpx import Timeout

from arcagi2.api.clients import AsyncResponsesAPIClient
from arcagi2.api.providers import OPENAI_API_PROVIDER
from arcagi2.solver.config.base import BaselineConfig, PROMPTS_FOLDER


PLAIN_COT_SOLVER = AsyncResponsesAPIClient.ResponsesAPICallConfig(
    model="gpt-5.2",
    api_provider=OPENAI_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "plain_cot_solver.txt",
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
    max_retries=2
)

GPT_5_2_XHIGH_BASELINE_CONFIG = BaselineConfig(
    plain_cot_solver=PLAIN_COT_SOLVER,
)