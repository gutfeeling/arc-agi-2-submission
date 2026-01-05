from dataclasses import dataclass

@dataclass
class APIProvider:
    name: str
    base_url: str
    api_key_env_var: str

OPENAI_API_PROVIDER = APIProvider(
    name="openai",
    base_url="https://api.openai.com/v1",
    api_key_env_var="OPENAI_API_KEY"
)

MOONSHOT_AI_API_PROVIDER = APIProvider(
    name="moonshot",
    base_url="https://api.moonshot.ai/v1",
    api_key_env_var="MOONSHOT_AI_API_KEY"
)

VLLM_API_PROVIDER = APIProvider(
    name="vllm",
    base_url="http://localhost:8000/v1",
    api_key_env_var="VLLM_API_KEY"
)

ANTHROPIC_API_PROVIDER = APIProvider(
    name="anthropic",
    base_url="https://api.anthropic.com/v1",
    api_key_env_var="ANTHROPIC_API_KEY"
)
