from dataclasses import dataclass
from pathlib import Path
from typing import Type

from daytona import Image, Resources

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.sandbox.base import Sandbox
from arcagi2.sandbox.daytona_sandbox import DaytonaSandbox
from arcagi2.sandbox.ipybox_sandbox import IPyBoxSandbox
from arcagi2.utils.utils import SerializableDataclassMixin


PROMPTS_FOLDER = Path(__file__).absolute().parents[1] / "prompts"
CODE_TEMPLATES_FOLDER = Path(__file__).absolute().parents[1] / "code_templates"

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
        "coverage",
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
    num_samples: int = 5

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

@dataclass(kw_only=True)
class InterleavedThinkingConfig(SolverConfig):
    sandbox_cls: Type[Sandbox]
    sandbox_kwargs: dict
    interleaved_thinking_solver: AbstractAPIClient.CallConfig
    soft_verifier: AbstractAPIClient.CallConfig
    generalizer: AbstractAPIClient.CallConfig

@dataclass(kw_only=True)
class BaselineConfig(SolverConfig):
    use_tools: bool = False
    plain_cot_solver: AbstractAPIClient.CallConfig