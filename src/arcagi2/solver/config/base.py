from dataclasses import dataclass
from pathlib import Path
from typing import Union

from arcagi2.api.clients import AbstractAPIClient


PROMPTS_FOLDER = Path(__file__).absolute().parents[1] / "prompts"
CODE_TEMPLATES_FOLDER = Path(__file__).absolute().parents[1] / "code_templates"

@dataclass
class SystemConfig:
    code_sandbox_container_tag: Union[str, None]
    interleaved_thinking_solver: AbstractAPIClient.CallConfig
    soft_verifier: AbstractAPIClient.CallConfig
    generalizer: AbstractAPIClient.CallConfig

@dataclass
class BaselineConfig:
    plain_cot_solver: AbstractAPIClient.CallConfig