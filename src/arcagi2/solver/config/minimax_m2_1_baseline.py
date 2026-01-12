from arcagi2.api.clients import AsyncMessagesAPIClient
from arcagi2.api.providers import MINIMAX_API_PROVIDER
from arcagi2.solver.config.base import BaselineConfig, PROMPTS_FOLDER


PLAIN_COT_SOLVER = AsyncMessagesAPIClient.MessagesAPICallConfig(
    model="MiniMax-M2.1",
    api_provider=MINIMAX_API_PROVIDER,
    prompt_path = PROMPTS_FOLDER / "plain_cot_solver.txt",
    system_prompt_path=None,
    client_kwargs = {
        "max_retries": 0,
    },
    raw_request_kwargs = {
        "thinking": {
            "type": "enabled",
        },
        "max_tokens": 196_608
    },
    cache_ttl=None
)

MINIMAX_M2_1_BASELINE_CONFIG = BaselineConfig(
    plain_cot_solver=PLAIN_COT_SOLVER,
)