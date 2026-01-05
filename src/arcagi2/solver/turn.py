import asyncio
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import re
from typing import Callable, Union

from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic import APIConnectionError as AnthropicAPIConnectionError
from anthropic import APITimeoutError as AnthropicAPITimeoutError
from anthropic import RateLimitError as AnthropicRateLimitError
import httpx
from jsonschema import validate, ValidationError
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APITimeoutError as OpenAIAPITimeoutError
from openai import BadRequestError as OpenAIBadRequestError
from openai import RateLimitError as OpenAIRateLimitError

from arcagi2.api.clients import AbstractAPIClient, TurnResult
from arcagi2.api.exceptions import (
    InvalidChatCompletionException, 
    InvalidMessageException,
    InvalidResponseException
)
from arcagi2.solver.exceptions import InvalidTurnResult, MaxRetriesExceeded
from arcagi2.tools.code_interpreter_ipybox import IPyBox
from arcagi2.utils.utils import (
    get_code_matches, 
    save_jsonl, 
    save_text, 
    save_json, 
    code_block_defines_function,
    get_json_matches,
)

logger = logging.getLogger(__name__)


class AbstractTurn:
    @dataclass
    class AbstractParsedTurnResult:
        trace: list[str]
        
        @classmethod
        def from_turn_result(cls, result: TurnResult) -> "AbstractTurn.AbstractParsedTurnResult":
            raise NotImplementedError("Subclasses must implement this method")

        def save_to_dir(self, dir: Path, config: AbstractAPIClient.CallConfig):
            dir.mkdir(parents=True, exist_ok=True)
            if self.trace is not None:
                try:
                    trace_file_path = dir / "trace.jsonl"
                    logger.info(f"Saving trace to {trace_file_path}")
                    save_jsonl(self.trace, trace_file_path)

                    cot_file_path = dir / "cot.txt"
                    logger.info(f"Saving COT to {cot_file_path}")
                    code_interpreter_tool_name = None
                    for tool in config.tools:
                        if isinstance(tool, IPyBox):
                            code_interpreter_tool_name = tool.name
                    save_text(
                        config.client_class.get_readable_trace(
                            self.trace, 
                            include_prompt=True,
                            include_interpreter_output=True,
                            code_interpreter_tool_name=code_interpreter_tool_name,
                        ), 
                        cot_file_path
                    )

                    token_consumption_file_path = dir / "token_consumption.json"
                    logger.info(f"Saving token consumption to {token_consumption_file_path}")
                    token_consumption = config.client_class.get_token_consumption_from_trace(self.trace)
                    save_json(asdict(token_consumption), token_consumption_file_path)
                    
                    max_context_file_path = dir / "max_context.json"
                    logger.info(f"Saving max context to {max_context_file_path}")
                    max_context = config.client_class.get_max_context_from_trace(self.trace)
                    save_json({"max_context": max_context}, max_context_file_path)
                except Exception as e:
                    logger.exception(f"Error saving additional information from trace")
    
    PARSED_TURN_RESULT_CLS = None

    def __init__(
            self, 
            config: AbstractAPIClient.CallConfig, 
            max_retries: int, 
            base_delay: int=2, 
            max_delay: int=600
            ):
        self.config = config
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def _with_retry(self, func: Callable, **kwargs) -> "AbstractTurn.AbstractParsedTurnResult":
        """
        Robust retry logic
        
        - Limited retries (up to max_retries) for incomplete responses  bad request errors
        - Limited retries (up to max_retries) for invalid turn results
        - Unlimited retries with exponential backoff for: connection errors, rate limits, timeouts
        
        Returns ParsedTurnResult.
        Raises MaxRetriesExceeded if all retries fail.
        """
        attempt = 0
        delay = self.base_delay
        while True:
            if attempt > 0:
                logger.info(f"Retry (limited retry error) {attempt} of {self.max_retries}")
            try:
                result = await func(config=self.config, **kwargs)
            except (
                InvalidChatCompletionException,
                InvalidResponseException,
                InvalidMessageException,
                AnthropicBadRequestError,
                OpenAIBadRequestError,
            ):
                if attempt < self.max_retries:
                    attempt += 1
                    delay = self.base_delay
                    logger.exception("Limited retry error, retrying...")
                    continue
                else:
                    logger.exception(f"Failed to get response after {self.max_retries} retries (for limited retry error)")
                    raise MaxRetriesExceeded(f"Failed to get response after {self.max_retries} retries (for limited retry error)")
            except (
                OpenAIAPIConnectionError,
                AnthropicAPIConnectionError,
                OpenAIRateLimitError,
                AnthropicRateLimitError,
                OpenAIAPITimeoutError,
                AnthropicAPITimeoutError,
                httpx.TimeoutException,  # Catches ReadTimeout, WriteTimeout, ConnectTimeout, PoolTimeout
            ):
                # AnthropicAPITimeoutError: We use streaming for anthropic, so timeout errors
                # are not due to long thinking. We can safely retry with exponential backoff.
                # httpx.TimeoutException: Can leak through during streaming when OpenAI SDK doesn't wrap it.
                logger.exception("Unlimited retry error, retrying with exponential backoff...")
                delay = min(delay * 2, self.max_delay)
                await asyncio.sleep(delay)
                continue
            try:
                return self.PARSED_TURN_RESULT_CLS.from_turn_result(result)
            except InvalidTurnResult as e:
                logger.exception(f"Error converting text response to turn result")
                if attempt < self.max_retries:
                    attempt += 1
                    delay = self.base_delay
                    logger.exception("Limited retry error, retrying...")
                    continue
                else:
                    logger.exception(f"Failed to get response after {self.max_retries} retries (for limited retry error)")
                    raise MaxRetriesExceeded(f"Failed to get response after {self.max_retries} retries (for limited retry error)")

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        # If prompt contains pattern like {{variable}}, raise a ValueError
        # Use regex to find all matches
        matches = re.findall(r"{{.*?}}", prompt)
        if matches:
            raise ValueError(f"Prompt contains unfilled variables: {matches}")
            
    def _render_prompt(self, prompt_template_vars: dict[str, str]) -> str:
        prompt = self.config.prompt
        for key, value in prompt_template_vars.items():
            prompt = prompt.replace("{{" + key + "}}", value if value is not None else "Not provided")
        self._validate_prompt(prompt)
        return prompt

    async def run(
            self, 
            prompt_template_vars: dict[str, str], 
            use_tools: bool, 
            save_to_dir: Union[Path, None]=None, 
            **kwargs
            ) -> "AbstractTurn.AbstractParsedTurnResult":
        prompt = self._render_prompt(prompt_template_vars)
        logger.info(f"Prompt:\n{prompt}")
        if use_tools:
            func = self.config.client_class.call_with_tools
        else:
            func = self.config.client_class.call_without_tools
        parsed_turn_result = await self._with_retry(
            func, 
            messages=[{"role": "user", "content": prompt}], 
            **kwargs
        )
        if save_to_dir is not None:
            parsed_turn_result.save_to_dir(save_to_dir, self.config)
        return parsed_turn_result

    async def run_cells(self, cells):
        # TODO: Implement this
        pass

class CodeSolutionTurn(AbstractTurn):
    """
    Expects model to output a code block defining a function called 'solution'

    Used by the minimal inductive solver (interleaved thinking) and generalizer.
    """
    @dataclass
    class CodeSolution(AbstractTurn.AbstractParsedTurnResult):
        solution: str

        def save_to_dir(self, dir: Path, config: AbstractAPIClient.CallConfig) -> None:
            super().save_to_dir(dir, config)
            solution_file_path = dir / "solution.py"
            logger.info(f"Saving solution to {solution_file_path}")
            save_text(self.solution, solution_file_path)
            
        @classmethod
        def from_turn_result(cls, result: TurnResult) -> "CodeSolutionTurn.CodeSolution":
            code_matches = get_code_matches(result.content)
            if len(code_matches) > 0:
                try:
                    # Use last code block since generalizer may have code blocks in the preceding change descriptions
                    code_block = code_matches[-1].group(1)
                    if code_block_defines_function(code_block, "solution"):
                        solution = code_block
                        logger.info(f"Solution:\n{solution}")
                        return cls(
                            solution=solution,
                            trace=result.trace
                        )
                    else:
                        raise InvalidTurnResult(f"Solution doesn't define a function called 'solution'")
                except Exception as e:
                    raise InvalidTurnResult(f"Error parsing code block returned by model")
            else:
                raise InvalidTurnResult(f"No code block returned by model")

    PARSED_TURN_RESULT_CLS = CodeSolution

class GridSolutionTurn(AbstractTurn):
    """Expects model to output a grid of numbers. Used for plain COT baseline."""
    @dataclass
    class GridSolution(AbstractTurn.AbstractParsedTurnResult):
        grid: list[list[int]]

        def save_to_dir(self, dir: Path, config: AbstractAPIClient.CallConfig) -> None:
            super().save_to_dir(dir, config)
            solution_file_path = dir / "solution.json"
            logger.info(f"Saving solution to {solution_file_path}")
            save_json(self.grid, solution_file_path)
            
        @classmethod
        def from_turn_result(cls, result: TurnResult) -> "GridSolutionTurn.GridSolution":
            try:
                # Find all grids (consecutive sequences of grid rows)
                grids = []
                current = []
                for line in result.content.split("\n"):
                    stripped = line.strip()
                    if re.fullmatch(r"[0-9]( [0-9])*", stripped):
                        current.append([int(part) for part in stripped.split()])
                    elif len(current) > 0:
                        grids.append(current)
                        current = []
                if len(current) > 0:
                    grids.append(current)
                
                # Take the last grid
                if len(grids) > 0:
                    grid = grids[-1]
                else:
                    raise InvalidTurnResult("No grid found in model's response")

                if not all(len(row) == len(grid[0]) for row in grid):
                    raise InvalidTurnResult("Grid is not rectangular")

                return cls(
                    grid=grid,
                    trace=result.trace
                )
            except Exception:
                raise InvalidTurnResult(f"Error getting output grid from model's response")

    PARSED_TURN_RESULT_CLS = GridSolution

class SoftVerificationTurn(AbstractTurn):
    """Expects model to output a JSON array containing objects fenced by ```json and ```."""

    @dataclass
    class SoftVerificationResult(AbstractTurn.AbstractParsedTurnResult):
        DECISION_SCHEMA = {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "property": {"type": "string", "enum": ["special_casing", "color_arithmetic"]},
                    "decision": {"type": "boolean"},
                    "rationale": {"type": "string"}
                },
                "required": ["property", "decision", "rationale"]
            }
        }

        decision: list[dict]

        @property
        def passed(self) -> bool:
            if self.decision is not None:
                if not any(property["decision"] for property in self.decision):
                    return True
            return False

        def save_to_dir(self, dir: Path, config: AbstractAPIClient.CallConfig) -> None:
            super().save_to_dir(dir, config)
            decision_file_path = dir / "soft_verification.json"
            logger.info(f"Saving decision to {decision_file_path}")
            save_json(self.decision, decision_file_path)
            
        @classmethod
        def from_turn_result(cls, result: TurnResult) -> "SoftVerificationTurn.SoftVerificationResult":
            json_matches = get_json_matches(result.content)
            if len(json_matches) == 0:
                raise InvalidTurnResult("No JSON block returned by soft verifier")
            try:
                decision = json.loads(json_matches[-1].group(1))
                validate(decision, cls.DECISION_SCHEMA)
                logger.info(f"Decision:\n{decision}")
                return cls(
                    decision=decision,
                    trace=result.trace
                )
            except json.JSONDecodeError as e:
                raise InvalidTurnResult(f"Error parsing JSON")
            except ValidationError as e:
                raise InvalidTurnResult(f"Invalid decision format")

    PARSED_TURN_RESULT_CLS = SoftVerificationResult