from dataclasses import dataclass, fields
from pathlib import Path
from typing import Type

from daytona import Image, Resources

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.sandbox.base import Sandbox
from arcagi2.sandbox.daytona_sandbox import DaytonaSandbox
from arcagi2.sandbox.ipybox_sandbox import IPyBoxSandbox
from arcagi2.utils.utils import SerializableDataclassMixin


PROMPTS_FOLDER = Path(__file__).absolute().parents[1] / "prompts"

IPYBOX_SANDBOX_CLS = IPyBoxSandbox
IPYBOX_SANDBOX_KWARGS = {"tag": "ipybox:solver"} # "localhost/ipybox:solver" if using podman

DAYTONA_SANDBOX_CLS = DaytonaSandbox
DAYTONA_SANDBOX_KWARGS = {
    "image": Image.debian_slim("3.12").pip_install([
        "numpy", 
        "scipy", 
        "shapely", 
        "networkx", 
        "scikit-image", 
        "more-itertools", 
        "pillow",
        "python-constraint", 
        "ortools", 
        "z3-solver",
        "ipykernel" 
    ]),
    "resources": Resources(
        cpu=1,
        memory=2,
        disk=3
    ),
    "creation_timeout": 600,
    "auto_stop_interval": 12 * 60    # 12 hours
}

@dataclass
class SolverConfig(SerializableDataclassMixin):
    call_config: AbstractAPIClient.CallConfig

    # For avoidable model failures and avoidable sandbox bugs
    max_retries: int = 2

    # For infrastructure errors
    base_delay: int = 2
    delay_multiplier: float = 2
    max_delay: int = 600
    max_backoff_retries: int = 20    # Around 2 hours of backoff before giving up

    # For code execution timeout (for non-tool code executions)
    code_timeout: float = 120
    use_tools: bool = True

    def set_vllm_base_url(self, vllm_base_url: str) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, AbstractAPIClient.CallConfig):
                call_config = value        
                if call_config.api_provider.name == "vllm":
                    call_config.api_provider.base_url = vllm_base_url

    @property
    def uses_daytona(self) -> bool:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, AbstractAPIClient.CallConfig):
                call_config = value        
                if call_config.sandbox_cls is DaytonaSandbox:
                    return True
        return False

@dataclass(kw_only=True)
class AgenticCodingConfig(SolverConfig):
    sandbox_cls: Type[Sandbox]
    sandbox_kwargs: dict

@dataclass(kw_only=True)
class PlainCOTConfig(SolverConfig):
    use_tools: bool = False