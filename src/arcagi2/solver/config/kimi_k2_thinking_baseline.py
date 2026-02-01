from arcagi2.api.clients import AsyncChatCompletionsAPIClient
from arcagi2.api.providers import MOONSHOT_AI_API_PROVIDER
from arcagi2.solver.config.base import PlainCOTConfig, PROMPTS_FOLDER


KIMI_K2_THINKING_PLAIN_COT_CONFIG = PlainCOTConfig(
    call_config=AsyncChatCompletionsAPIClient.ChatCompletionsAPICallConfig(
        model="kimi-k2-thinking",
        api_provider=MOONSHOT_AI_API_PROVIDER,
        prompt_path = PROMPTS_FOLDER / "plain_cot_solver.txt",
        system_prompt_path=None,
        client_kwargs = {
            "max_retries": 0,
        },
        raw_request_kwargs = {
            "temperature": 1.0,
            "max_tokens": 256_000,
            "stream": True,
            "stream_options": {
                "include_usage": True
            }
        },
    )
)