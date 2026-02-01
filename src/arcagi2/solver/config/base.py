from arcagi2.api.clients import AsyncMessagesAPIClient
from arcagi2.api.providers import ANTHROPIC_API_PROVIDER
from arcagi2.solver.config.base import PlainCOTConfig, PROMPTS_FOLDER


CLAUDE_OPUS_4_5_64K_PLAIN_COT_CONFIG = PlainCOTConfig(
    call_config=AsyncMessagesAPIClient.MessagesAPICallConfig(
        model="claude-opus-4-5-20251101",
        api_provider=ANTHROPIC_API_PROVIDER,
        prompt_path = PROMPTS_FOLDER / "plain_cot_solver.txt",
        system_prompt_path=None,
        client_kwargs = {
            "max_retries": 0,
        },
        raw_request_kwargs = {
            "thinking": {
                "type": "enabled",
                "budget_tokens": 63999,
            },
            "max_tokens": 64000,
        },
        cache_ttl=None
    )
)