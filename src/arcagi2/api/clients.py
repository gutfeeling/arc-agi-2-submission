from abc import ABC, abstractmethod
import asyncio
import copy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Optional, Self, Union, Type
from arcagi2.sandbox.base import Sandbox

import anthropic
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic.types.beta import BetaMessage
from jsonschema.exceptions import ValidationError
from openai import AsyncOpenAI
from openai import BadRequestError as OpenAIBadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.responses import Response
from pydantic import BaseModel, ValidationError as PydanticValidationError

from arcagi2.api.exceptions import (
    InvalidChatCompletionException, 
    InvalidMessageException,
    InvalidResponseException
)
from arcagi2.api.providers import APIProvider
from arcagi2.api.utils import replace_developer_message, messages_contains_system_prompt
from arcagi2.sandbox.base import REPL, Sandbox
from arcagi2.tools.base import Tool
from arcagi2.tools.repl_tool import REPLTool
from arcagi2.utils.utils import read_file, SerializableDataclassMixin


logger = logging.getLogger(__name__)
    
@dataclass
class RawTurnResult:
    """Result of a single turn of the conversation. When using tool calling, one turn might have multiple steps."""
    response: Union[ChatCompletion, Response, BetaMessage]    # this is the final response obtained in the last step of the turn.
    trace: list[str]    # trace should contain details of all steps in the turn, so that we can analyze them later.

@dataclass
class TurnResult:
    role: str
    content: str
    trace: list[str]

@dataclass
class TokenConsumption:
    input: int
    output: int
    cached_input: int
    total: int
    
    def __add__(self, other: "TokenConsumption") -> "TokenConsumption":
        if isinstance(other, TokenConsumption):
            return TokenConsumption(
                input=self.input + other.input,
                output=self.output + other.output,
                cached_input=self.cached_input + other.cached_input,
                total=self.total + other.total
            )
        return NotImplemented
    
    def __iadd__(self, other: "TokenConsumption") -> Self:
        if isinstance(other, TokenConsumption):
            self.input += other.input
            self.output += other.output
            self.cached_input += other.cached_input
            self.total += other.total
            return self
        return NotImplemented
    
    def __mul__(self, other: Union[int, float]) -> "TokenConsumption":
        if isinstance(other, (int, float)):
            return TokenConsumption(
                input=int(self.input * other),
                output=int(self.output * other),
                cached_input=int(self.cached_input * other),
                total=int(self.total * other)
            )
        return NotImplemented
    
    def __rmul__(self, other: Union[int, float]) -> "TokenConsumption":
        return self.__mul__(other)

class AbstractAPIClient(ABC):

    @dataclass
    class CallConfig(SerializableDataclassMixin):
        api_provider: APIProvider
        model: str
        prompt_path: Path    # path to the prompt template
        system_prompt_path: Optional[Path]    # path to the system prompt template

        # For non-streaming requests, the client is responsible for retrying timeouts that happens when model thinks too long. 
        # So don't reset `max_retries` in the client_kwargs to 0, unless you really want to disable retrying those.
        client_kwargs: dict
        
        # kwargs is passed to the API request. Don't pass system prompts here (even though it is allowed in Responses and Messages APIs). 
        # Use the system_prompt_path property instead.
        raw_request_kwargs: dict

        # Tool loop related configs; all values should be non-None if tools are provided
        tools: Optional[list[Tool]] = None
        # Maximum number of retries that preserve state (i.e. doesn't try to reset the state).
        max_retries: Optional[int] = None
        # Sleep time between tool loop steps
        sleep: Optional[float] = None
        sandbox_cls: Optional[Type[Sandbox]] = None
        sandbox_kwargs: Optional[dict] = None
        initial_code_timeout: Optional[float] = None

        @property
        def prompt(self) -> str:
            if hasattr(self, "_prompt"):
                return self._prompt
            self._prompt = read_file(self.prompt_path)
            return self._prompt

        @property
        def system_prompt(self) -> Union[str, None]:
            if hasattr(self, "_system_prompt"):
                return self._system_prompt
            if isinstance(self.system_prompt_path, Path):
                self._system_prompt = read_file(self.system_prompt_path)
            else:
                self._system_prompt = None
            return self._system_prompt

        @property
        def request_kwargs(self) -> dict:
            return self.raw_request_kwargs

        def __post_init__(self) -> None:
            if self.tools is not None:
                if self.max_retries is None:
                    raise ValueError("max_retries must be provided if tools are provided")
                if self.sleep is None:
                    raise ValueError("sleep must be provided if tools are provided")
                if any(isinstance(tool, REPLTool) for tool in self.tools): 
                    if self.sandbox_cls is None:
                        raise ValueError("sandbox_cls must be provided if tools are provided")
                    if self.sandbox_kwargs is None:
                        raise ValueError("sandbox_kwargs must be provided if tools are provided")
                    if self.initial_code_timeout is None:
                        raise ValueError("initial_code_timeout must be provided if tools are provided")

    def __init__(self, api_provider: APIProvider, **kwargs):
        self.api_provider = api_provider
        api_key = os.getenv(api_provider.api_key_env_var)
        if api_key is None and kwargs.get("api_key") is None:
            raise ValueError(f"{api_provider.api_key_env_var} is not set")
        self.client = AsyncOpenAI(
            base_url=api_provider.base_url,
            api_key=api_key,
            **kwargs
        )

    async def __aenter__(self) -> Self:
        # Supports context manager mode, just like the original AsyncOpenAI client
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return await self.client.__aexit__(exc_type, exc_val, exc_tb)

    @classmethod
    @abstractmethod
    async def call(cls, config: CallConfig, messages: list[dict], initial_code: Optional[list[str]] = None) -> TurnResult:
        pass

    async def ensure_repl_and_run_tool_loop(
            self,
            tools: list[Tool],
            sandbox_cls: Optional[Type[Sandbox]] = None,
            sandbox_kwargs: Optional[dict] = None,
            initial_code: Optional[list[str]] = None,
            initial_code_timeout: Optional[float] = None,
            **kwargs
        ) -> RawTurnResult:
        if any(isinstance(tool, REPLTool) for tool in tools):
            async with sandbox_cls(**sandbox_kwargs) as sandbox:
                async with sandbox.create_repl() as repl:
                    if initial_code is not None and len(initial_code) > 0:
                        logger.info(f"Running initial code in REPL")
                        for cell in initial_code:
                            await repl.execute(
                                code=cell,
                                timeout=initial_code_timeout
                            )
                        logger.info(f"Initial code in REPL executed successfully")

                    return await self.run_tool_loop(
                        tools=tools, 
                        repl=repl,
                        **kwargs
                    )
        else:
            return await self.run_tool_loop(
                tools=tools, 
                repl=None, 
                **kwargs
            )

    async def run_tool_loop(self, tools: list[Tool], repl: Optional[REPL], **kwargs) -> RawTurnResult:
        self._validate_tools(tools)
        return await self._run_tool_loop(
            tools=tools,
            repl=repl,
            **kwargs
        )

    def _validate_tools(self, tools: list[Tool]) -> None:
        if not isinstance(tools, list):
            raise ValueError(f"Tools must be a list. Got {type(tools)}")
        if len(tools) == 0:
            raise ValueError(f"Tools list must not be empty")
        for tool in tools:
            if not isinstance(tool, Tool):
                raise ValueError(f"Tool {tool} must be an instance of Tool. Got {type(tool)}")
        num_repl_tools = sum(1 for tool in tools if isinstance(tool, REPLTool))
        if num_repl_tools > 1:
            raise ValueError(f"Only one REPL tool is supported. Got {num_repl_tools} REPL tools")

    @abstractmethod
    def _run_tool_loop(self, tools: list[Tool], repl: Optional[REPL], **kwargs) -> RawTurnResult:
        pass

    @staticmethod
    @abstractmethod
    def token_consumption_from_response(response: Union[ChatCompletion, Response, BetaMessage]) -> TokenConsumption:
        pass

    @staticmethod
    @abstractmethod
    def get_max_context_from_trace(trace: list[str]) -> int:
        pass

    @staticmethod
    @abstractmethod
    def get_readable_trace(trace: list[str]) -> str:
        pass       

    @staticmethod
    def _find_tool(name: str, tools: list[Tool]) -> Tool:
        for tool in tools:
            if tool.name == name:
                return tool
        raise ValueError(f"Tool name {name} not found in tools list {[tool.name for tool in tools]}")

class AsyncChatCompletionsAPIClient(AbstractAPIClient):
    """Client for the OpenAI Chat Completions API."""

    @dataclass
    class ChatCompletionsAPICallConfig(AbstractAPIClient.CallConfig):
        pass

    @classmethod 
    async def call(cls, config: ChatCompletionsAPICallConfig, messages: list[dict], initial_code: Optional[list[str]] = None) -> TurnResult:
        if not isinstance(config, cls.ChatCompletionsAPICallConfig):
            raise ValueError(f"Config must be an instance of {cls.ChatCompletionsAPICallConfig}")
        messages = replace_developer_message(
            messages=messages, 
            system_prompt=config.system_prompt
        )
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            if config.tools is None:
                result = await client.create_chat_completion(
                    model=config.model,
                    messages=messages,
                    **config.request_kwargs
                )
            else:
                result = await client.ensure_repl_and_run_tool_loop(
                    tools=config.tools,
                    sandbox_cls=config.sandbox_cls,
                    sandbox_kwargs=config.sandbox_kwargs,
                    initial_code=initial_code,
                    initial_code_timeout=config.initial_code_timeout,
                    model=config.model,
                    messages=messages,
                    max_retries=config.max_retries,
                    sleep=config.sleep,
                    **config.request_kwargs
                )
            client.validate_response(result.response)
            return TurnResult(
                role=result.response.choices[0].message.role,
                content=result.response.choices[0].message.content,
                trace=result.trace
            )

    @staticmethod
    def get_token_consumption_from_trace(trace: list[str]) -> TokenConsumption:
        token_consumption = TokenConsumption(
            input=0,
            output=0,
            cached_input=0,
            total=0
        )
        for item in trace:
            try:
                response = ChatCompletion.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption += AsyncChatCompletionsAPIClient.token_consumption_from_response(response)
        return token_consumption

    @staticmethod
    def get_readable_trace(
            trace: list[str], 
            include_interpreter_output: bool = True, 
            separator: str = "\n\n",
            include_prompt: bool = False,
            code_interpreter_tool_name: Optional[str] = None
            ) -> str:
        # https://huggingface.co/datasets/JoeYing/ReTool-SFT
        text = ""
        if include_prompt:
            first_request = json.loads(trace[0])
            messages_in_first_request = first_request["messages"]
            num_user_prompts = len([message for message in messages_in_first_request if message["role"] == "user"])
            assert num_user_prompts <= 1, f"Expected less than or equal to 1 user prompt in the first request, but got {num_user_prompts}"
            for message in messages_in_first_request:
                if message["role"] == "user":
                    text += f"<prompt>\n{message['content'].strip()}\n</prompt>{separator}"
                    break
        for line_num in range(1, len(trace), 2):
            response = ChatCompletion.model_validate_json(trace[line_num])
            content = response.choices[0].message.content
            reasoning = getattr(
                response.choices[0].message, "reasoning", None
            )  # OpenRouter
            if reasoning is not None and len(reasoning) > 0:
                text += f"<think>\n{reasoning.strip()}\n</think>{separator}"
            reasoning_content = getattr(
                response.choices[0].message, "reasoning_content", None
            )  # DeepSeek
            if reasoning_content is not None and len(reasoning_content) > 0:
                text += f"<think>\n{reasoning_content.strip()}\n</think>{separator}"
            if content is not None and len(content) > 0:
                text += f"{content.strip()}{separator}"
            if code_interpreter_tool_name is None:
                continue
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls is not None and len(tool_calls) > 0:    # GPT OSS on VLLM seems to return empty tool calls if no tool calls are needed.
                next_line = json.loads(trace[line_num + 1])
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except Exception as e:
                        text += f"<malformed_tool_call>\n{tool_call.function.arguments}\n</malformed_tool_call>{separator}"
                        continue
                    if tool_name == code_interpreter_tool_name:
                        text += f"<code>\n{tool_args['code'].strip()}\n</code>{separator}"
                        if include_interpreter_output:
                            for message in next_line["messages"]:
                                if (
                                    message["role"] == "tool"
                                    and message["tool_call_id"] == tool_call.id
                                ):
                                    text += f"<interpreter>\n{message['content'].strip()}\n</interpreter>{separator}"
        return text

    @staticmethod
    def get_max_context_from_trace(trace: list[str]) -> int:
        for item in reversed(trace):
            try:
                response = ChatCompletion.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption = AsyncChatCompletionsAPIClient.token_consumption_from_response(response)
            return token_consumption.total
        return 0
    
    @staticmethod
    def token_consumption_from_response(response: ChatCompletion) -> TokenConsumption:
        try:
            cached_input = response.usage.prompt_tokens_details.cached_tokens
        # prompt_token_details might not always be present when using OpenRouter
        # https://openrouter.ai/docs/use-cases/usage-accounting
        except AttributeError:    
            cached_input = 0
        else:
            if cached_input is None:
                cached_input = 0
        return TokenConsumption(
            input=response.usage.prompt_tokens,
            output=response.usage.completion_tokens,
            cached_input=cached_input,
            total=response.usage.total_tokens
        )
    
    @staticmethod
    def validate_response(response: ChatCompletion) -> None:
        finish_reason = response.choices[0].finish_reason
        if response.choices[0].finish_reason not in ["stop", "tool_calls", "function_call"]:
            raise InvalidChatCompletionException(finish_reason)
        
    def _make_messages_compatible(self, messages: list[dict]) -> list[dict]:
        """Deepseek, Gemini and Moonshot AI doesn't seem to support the new developer messages that replaced system messages in the OpenAI spec"""
        if self.api_provider.name in ["deepseek", "gemini", "moonshot"]:
            for message in messages:
                # Sometimes the messages list contains ChatCompletionMessage objects, because we use that to add the model's function calls to the history
                if isinstance(message, dict) and message["role"] == "developer":
                    message["role"] = "system"
        return messages
    
    @staticmethod
    def _make_json_serializable(messages: list[Union[dict, ChatCompletionMessage]]) -> list[dict]:
        json_serializable_messages = []
        for message in messages:
            if isinstance(message, ChatCompletionMessage):    # Used for sending back function calls back to the model
                message = json.loads(message.model_dump_json())
            json_serializable_messages.append(message)
        return json_serializable_messages

    def _response_contains_malformed_tool_calls(self, response: ChatCompletion, tools: list[Tool]) -> bool:
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is None:
            return False
        for tool_call in tool_calls:
            try:
                tool = self._find_tool(tool_call.function.name, tools)
                args = json.loads(tool_call.function.arguments)
                tool.validate_arguments(args)
            except Exception as e:
                logger.exception(f"Malformed tool call")
                return True
        return False

    async def create_chat_completion(self, model: str, messages: list[Union[dict, ChatCompletionMessage]], **kwargs) -> RawTurnResult:
        messages = self._make_messages_compatible(messages)
        logger.info(f"Sending request to model: {model}")
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        # Defensive check for malformed response that happened once using a DeepSeek model in Openrouter
        if response.choices is None or len(response.choices) == 0:
            logger.error(f"Response has no choices: {response}")
            raise InvalidChatCompletionException("Response has no choices")
        logger.info(f"Response message content:\n{response.choices[0].message.content}")
        if self.api_provider.name == "openrouter" and hasattr(response.choices[0].message, "reasoning"):
            logger.info(f"Reasoning content:\n{response.choices[0].message.reasoning}")
        elif self.api_provider.name in ["deepseek", "vllm"] and hasattr(response.choices[0].message, "reasoning_content"):
            logger.info(f"Reasoning content:\n{response.choices[0].message.reasoning_content}")
        # OpenRouter providers often don't report token consumption in the response.
        try:
            logger.info(f"Tokens consumed by this call: {self.token_consumption_from_response(response)}")
        except Exception:
            logger.exception(f"Failed to get token consumption from response")
        return RawTurnResult(
            response=response,
            trace=[
                json.dumps(
                    {"model": model, "messages": self._make_json_serializable(messages), **kwargs}
                ),    # TODO: Will fail if kwargs contain non-serializable objects.
                response.model_dump_json()
            ],
        )
    
    async def _run_tool_loop(
            self,
            tools: list[Tool],
            repl: Optional[REPL],
            model: str,
            messages: list[dict],
            max_retries: int,
            sleep: float,
            **kwargs
            ) -> RawTurnResult:
        trace = []
        messages = copy.deepcopy(messages)    # prevent side effects of modifying the messages list in place
        api_tools = [tool.chat_completions_schema for tool in tools]
        while True:
            retry_count = max_retries
            while True:
                try:
                    result = await self.create_chat_completion(
                        model=model,
                        messages=messages,
                        tools=api_tools,
                        **kwargs
                    )
                except OpenAIBadRequestError as e:
                    logger.exception("Bad request error during tool loop")
                    if retry_count > 0:
                        logger.info(f"Retrying")
                        retry_count -= 1
                        logger.info(f"Retry {max_retries - retry_count} of {max_retries} to avoid errors")
                        continue
                    else:
                        logger.info(f"Failed to avoid errors after {max_retries} retries")
                        raise e
                if self._response_contains_malformed_tool_calls(result.response, tools):
                    if retry_count > 0:
                        logger.info(f"Malformed tool calls found in response, retrying")
                        retry_count -= 1
                        logger.info(f"Retry {max_retries - retry_count} of {max_retries} to avoid malformed tool calls")
                        continue
                    else:
                        logger.info(f"Failed to avoid malformed tool calls after {max_retries} retries")
                        break
                break
            response = result.response
            trace.extend(result.trace)
            
            self.validate_response(response)
            message = response.choices[0].message
            tool_calls = message.tool_calls
            if tool_calls is None or len(tool_calls) == 0:
                logger.info("No tool calls in response")
                break
            logger.info(f"Found {len(tool_calls)} tool call(s)")
            
            # Add message to conversation, including any reasoning content
            # Providers that support interleaved thinking will know how to handle it.
            messages.append(message)    
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.function.name
                logger.info(f"Model called tool named {tool_name}")
                try:
                    tool = self._find_tool(tool_name, tools)
                except ValueError as e:
                    error_message = str(e)
                    logger.exception(error_message)
                    logger.info(f"Passing error message back to model")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_message,
                    })
                    break
                
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception as e:
                    error_message = f"Error parsing tool call arguments {tool_call.function.arguments}. Error message: {str(e)}"
                    logger.exception(error_message)
                    logger.exception(f"Passing error message back to model")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_message,
                    })
                    break
                logger.info(f"Function call with arguments: {args.keys()}")
                try:
                    tool.validate_arguments(args)
                except ValidationError as e:
                    error_message = str(e)
                    logger.exception(error_message)
                    logger.info(f"Passing error message back to model")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_message,
                    })
                    break
                if isinstance(tool, REPLTool):
                    result = await tool.run(repl=repl, **args)
                else:
                    result = await tool.run(**args)
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id, 
                    "content": result
                })
                logger.info("Added result to conversation")
            await asyncio.sleep(sleep)
        return RawTurnResult(
            response=response,
            trace=trace
        )

# This seems to be the cleanest way to set up the reverse relationship since the client class is not defined yet when the nested class is defined.
AsyncChatCompletionsAPIClient.ChatCompletionsAPICallConfig.client_class = AsyncChatCompletionsAPIClient
                   
class AsyncResponsesAPIClient(AbstractAPIClient):
    """Client for the OpenAI Responses API."""

    @dataclass
    class ResponsesAPICallConfig(AbstractAPIClient.CallConfig):
        background_mode_polling_interval: Optional[float] = None
        stateful: Optional[bool] = None   # Set to True when using tools with VLLM as the inference engine

        def __post_init__(self) -> None:
            super().__post_init__()
            if self.tools is not None:
                if self.stateful is None:
                    raise ValueError("stateful must be provided if tools are provided")
            if self.request_kwargs.get("background", False):
                if self.background_mode_polling_interval is None:
                    raise ValueError("background_mode_polling_interval must be provided if background mode is enabled")

        @property
        def request_kwargs(self) -> dict:
            if self.system_prompt is not None:
                if hasattr(self.raw_request_kwargs, "instructions"):
                    raise ValueError("Config contains both system_prompt and request keyword argument 'instructions'")
                return {**self.raw_request_kwargs, "instructions": self.system_prompt}
            return self.raw_request_kwargs

    def __init__(self, api_provider: APIProvider, *args, **kwargs):
        # We use Responses API for accessing GPT OSS using VLLM
        if api_provider.name not in ["openai", "vllm"]:
            raise ValueError(f"Responses API is only supported by OpenAI and VLLM. {api_provider.name} is not supported.")
        super().__init__(api_provider, *args, **kwargs)

    @classmethod
    async def call(cls, config: ResponsesAPICallConfig, messages: list[dict], initial_code: Optional[list[str]] = None) -> TurnResult:
        if messages_contains_system_prompt(messages):
            raise ValueError("We don't allow system messages in input when using Responses API. Use request param 'instructions' instead.")
        if not isinstance(config, cls.ResponsesAPICallConfig):
            raise ValueError(f"Config must be an instance of {cls.ResponsesAPICallConfig}")
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            if config.tools is None:
                result = await client.create_response(
                    model=config.model,
                    input=messages,
                    background_mode_polling_interval=config.background_mode_polling_interval,
                    **config.request_kwargs
                )
            else:
                result = await client.ensure_repl_and_run_tool_loop(
                    tools=config.tools,
                    sandbox_cls=config.sandbox_cls,
                    sandbox_kwargs=config.sandbox_kwargs,
                    initial_code=initial_code,
                    initial_code_timeout=config.initial_code_timeout,
                    model=config.model,
                    input=messages,
                    background_mode_polling_interval=config.background_mode_polling_interval,
                    max_retries=config.max_retries,
                    sleep=config.sleep,
                    stateful=config.stateful,
                    **config.request_kwargs
                )
            client.validate_response(result.response)
            return TurnResult(
                role="assistant",
                content=result.response.output_text,
                trace=result.trace
            )

    @staticmethod
    def get_readable_trace(
            trace: list[str],
            include_interpreter_output: bool = True,
            separator: str = "\n\n",
            include_prompt: bool = False,
            code_interpreter_tool_name: Optional[str] = None
            ) -> str:
        # https://huggingface.co/datasets/JoeYing/ReTool-SFT
        text = ""
        if include_prompt:
            first_request = json.loads(trace[0])
            input_items = first_request.get("input", [])
            num_user_prompts = len([item for item in input_items if item.get("role") == "user"])
            assert num_user_prompts <= 1, f"Expected less than or equal to 1 user prompt in the first request, but got {num_user_prompts}"
            for item in input_items:
                if item.get("role") == "user":
                    text += f"<prompt>\n{item['content'].strip()}\n</prompt>{separator}"
                    break
        for line_num in range(1, len(trace), 2):
            response = Response.model_validate_json(trace[line_num])
            # Add reasoning content: each reasoning text gets its own <think></think>
            if response.output is not None:
                reasoning_items = [el for el in response.output if getattr(el, "type", None) == "reasoning"]
                for reasoning_item in reasoning_items:
                    if getattr(reasoning_item, "content", None):
                        for content in reasoning_item.content:
                            if getattr(content, "text", None):
                                text += f"<think>\n{content.text.strip()}\n</think>{separator}"
            # Add assistant text
            content_text = response.output_text
            if content_text is not None and len(content_text) > 0:
                text += f"{content_text.strip()}{separator}"
            if code_interpreter_tool_name is None:
                continue
            # Handle function calls (tools)
            function_calls = [el for el in (response.output or []) if getattr(el, "type", None) == "function_call"]
            if len(function_calls) > 0 and line_num + 1 < len(trace):
                next_request = json.loads(trace[line_num + 1])
                next_input_items = next_request.get("input", [])
                for function_call in function_calls:
                    tool_name = getattr(function_call, "name", None)
                    try:
                        args_str = getattr(function_call, "arguments", None)
                        tool_args = json.loads(args_str) if args_str is not None else {}
                    except Exception:
                        text += f"<malformed_tool_call>\n{args_str}\n</malformed_tool_call>{separator}"
                        continue
                    if tool_name == code_interpreter_tool_name:
                        if "code" in tool_args and tool_args["code"] is not None:
                            text += f"<code>\n{tool_args['code'].strip()}\n</code>{separator}"
                        if include_interpreter_output:
                            call_id = getattr(function_call, "call_id", None)
                            for item in next_input_items:
                                if item.get("type") == "function_call_output" and item.get("call_id") == call_id:
                                    output = item.get("output")
                                    if output is not None and len(output) > 0:
                                        text += f"<interpreter>\n{output.strip()}\n</interpreter>{separator}"
                                    break
        return text

    @staticmethod
    def get_token_consumption_from_trace(trace: list[str]) -> TokenConsumption:
        token_consumption = TokenConsumption(
            input=0,
            output=0,
            cached_input=0,
            total=0
        )
        for item in trace:
            try:
                response = Response.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption += AsyncResponsesAPIClient.token_consumption_from_response(response)
        return token_consumption

    @staticmethod
    def get_max_context_from_trace(trace: list[str]) -> int:
        for item in reversed(trace):
            try:
                response = Response.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption = AsyncResponsesAPIClient.token_consumption_from_response(response)
            return token_consumption.total
        return 0

    @staticmethod
    def token_consumption_from_response(response: Response) -> TokenConsumption:
        # vllm doesn't return cached input tokens
        try:
            cached_input = response.usage.prompt_tokens_details.cached_tokens
        except AttributeError:    
            cached_input = 0
        else:
            if cached_input is None:
                cached_input = 0
        return TokenConsumption(
            input=response.usage.input_tokens,
            output=response.usage.output_tokens,
            cached_input=cached_input,
            total=response.usage.total_tokens
        )

    @staticmethod
    def _make_json_serializable(input_items: list) -> list:
        json_serializable_input_items = []
        for item in input_items:
            # Reasoning content sent back to model in stateless mode may contain objects of type ResponseOutputItem, which is a subscripted generic.
            # Instance check against ResponseOutputItem not possible as it is a subscripted generic
            if hasattr(item, "model_dump_json"):
                item = json.loads(item.model_dump_json())
            json_serializable_input_items.append(item)
        return json_serializable_input_items

    def _tool_loop_response_is_invalid(self, response: Response) -> bool:
        function_calls = [el for el in response.output if getattr(el, "type", None) == "function_call"]
        if len(function_calls) == 0:
            try:
                self.validate_response(response)
            except InvalidResponseException:
                return True
        return False

    def _response_contains_malformed_function_calls(self, response: Response, tools: list[Tool]) -> bool:
        function_calls = [el for el in response.output if getattr(el, "type", None) == "function_call"]
        for function_call in function_calls:
            try:
                tool = self._find_tool(function_call.name, tools)
                args = json.loads(function_call.arguments)
                tool.validate_arguments(args)
            except Exception:
                logger.exception(f"Malformed function call")
                return True
        return False

    def _response_contains_malformed_reasoning_content(self, response: Response, model: str) -> bool:
        # In the following cases, we are dealing with an open reasoning model which potentially has this problem
        # GPT-OSS definitely has the problem
        # Fine tuned versions of GPT-OSS models may have the problem too, so we check for vllm provider as well
        # VLLM means open reasoning model too, so that should return reasoning content as well and it's harmless to do the check
        if "gpt-oss" in model or self.api_provider.name == "vllm":
            reasoning_content = [el for el in response.output if getattr(el, "type", None) == "reasoning"]
            if len(reasoning_content) > 1:
                logger.exception(f"Response contains multiple reasoning items")
                return True
            if len(reasoning_content) > 0 and reasoning_content[0].content is not None:
                if len(reasoning_content[0].content) > 1:
                    logger.exception(f"Response contains multiple reasoning content items")
                    return True
        return False
    
    @staticmethod
    def validate_response(response: Response) -> None:
        status = response.status    
        if status != "completed":
            raise InvalidResponseException(status)
    
    def _fix_reasoning_content(self, output: list, model: str) -> list:
        """Keep only the first reasoning block with non-whitespace content, and only its first content item."""
        if "gpt-oss" in model or self.api_provider.name == "vllm":
            filtered_output = []
            reasoning_added = False
            
            for el in output:
                if el.type == "reasoning":
                    if reasoning_added or el.content is None:
                        continue
                    
                    # Filter out whitespace-only content items and keep only the first
                    non_whitespace_content = [
                        content for content in el.content
                        if content.text and content.text.strip()
                    ]
                    
                    if len(non_whitespace_content) > 0:
                        content = non_whitespace_content[0]
                        content.text = content.text.rstrip()    # Strip ending whitespaces
                        el.content = [content]
                        filtered_output.append(el)
                        reasoning_added = True
                else:
                    filtered_output.append(el)
            
            return filtered_output
        else:
            return output

    async def _handle_background_response(self, response: Response, background_mode_polling_interval: Optional[float]) -> Response:
        while response.status in {"queued", "in_progress"}:
            logger.debug(f"Current status: {response.status}")

            await asyncio.sleep(background_mode_polling_interval)
            response = await self.client.responses.retrieve(response.id)
        logger.info(f"Final status: {response.status} (ID: {response.id})")
        return response
    
    async def create_response(self, model: str, input: list[dict], background_mode_polling_interval: Optional[float], **kwargs) -> RawTurnResult:
        logger.info(f"Sending request to model: {model}")
        response = await self.client.responses.create(
            model=model,
            input=input,
            **kwargs
        )
        
        if kwargs.get("background", False):
            logger.info(f"Background mode enabled, polling for response (ID: {response.id})")
            response = await self._handle_background_response(response, background_mode_polling_interval)
        logger.info(f"Response output text:\n{response.output_text}")
        logger.info(f"Tokens consumed by this call: {self.token_consumption_from_response(response)}")
        return RawTurnResult(
            response=response,
            trace=[
                json.dumps(
                    {"model": model, "input": self._make_json_serializable(input), **kwargs}
                ),
                response.model_dump_json()
            ]
        )
    
    async def _run_tool_loop(
            self,
            tools: list[Tool],
            repl: Optional[REPL],
            model: str,
            input: list[dict],
            background_mode_polling_interval: Optional[float],
            max_retries: int,
            sleep: float,
            stateful: bool,
            **kwargs
            ) -> RawTurnResult:
        trace = []
        input = copy.deepcopy(input)    # prevent side effects of modifying the input list in place
        if stateful:
            previous_response_id = None
        api_tools = [tool.responses_api_schema for tool in tools]
        while True:
            retry_count = max_retries
            while True:
                request_args = {
                    "model": model,
                    "input": input,
                    "tools": api_tools,
                    "background_mode_polling_interval": background_mode_polling_interval,
                }
                if stateful:
                    request_args["previous_response_id"] = previous_response_id
                try:
                    result = await self.create_response(
                        **request_args,
                        **kwargs
                    )
                # At least for GPT OSS on VLLM, BadRequestErrors are recoverable. We assume same for other providers.
                # Errors have messages like: Unknown channel, tokens remaining etc.
                except OpenAIBadRequestError as e:
                    logger.exception("Bad request error during tool loop")
                    if retry_count > 0:
                        logger.info(f"Retrying")
                        retry_count -= 1
                        logger.info(f"Retry {max_retries - retry_count} of {max_retries} to avoid errors")
                        continue
                    else:
                        logger.info(f"Failed to avoid errors after {max_retries} retries")
                        raise e
                if (
                    self._response_contains_malformed_function_calls(result.response, tools) or 
                    self._tool_loop_response_is_invalid(result.response) or 
                    self._response_contains_malformed_reasoning_content(result.response, model)
                ):
                    if retry_count > 0:
                        logger.info(f"Errored (malformed function calls or incomplete response), retrying")
                        retry_count -= 1
                        logger.info(f"Retry {max_retries - retry_count} of {max_retries} to avoid errors")
                        continue
                    else:
                        logger.info(f"Failed to avoid errors after {max_retries} retries")
                        break
                break
            response = result.response
            trace.extend(result.trace)
            
            self.validate_response(response)
            if stateful:
                previous_response_id = response.id
            output_text = response.output_text
            if output_text is not None:
                logger.info(f"Model response:\n{output_text}")
            reasoning_items = [el for el in response.output if el.type == "reasoning"]
            for reasoning_item in reasoning_items:
                if reasoning_item.content is not None:
                    for content in reasoning_item.content:
                        logger.info(f"Reasoning text:\n{content.text}")
            if stateful:
                input = []
            else:
                fixed_output = self._fix_reasoning_content(response.output, model)
                input.extend(fixed_output)

            function_calls = [el for el in response.output if el.type == "function_call"]
            if len(function_calls) == 0:
                logger.info("No function calls in response")
                break
            logger.info(f"Found {len(function_calls)} function call(s)")
            for function_call in function_calls:
                logger.info(f"Model called function named {function_call.name}")
                tool_name = function_call.name
                try:
                    tool = self._find_tool(tool_name, tools)
                except ValueError as e:
                    error_message = str(e)
                    logger.exception(error_message)
                    logger.info(f"Passing error message back to model")
                    input.append({
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": error_message,
                    })
                    break
                try:
                    args = json.loads(function_call.arguments)
                except Exception as e:
                    error_message = f"Error parsing function call arguments {function_call.arguments}. Error message: {str(e)}"
                    logger.exception(error_message)
                    logger.exception(f"Passing error message back to model")
                    input.append({
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": error_message,
                    })
                    break
                logger.info(f"Function call with arguments: {args.keys()}")
                try:
                    tool.validate_arguments(args)
                except ValidationError as e:
                    error_message = str(e)
                    logger.exception(error_message)
                    logger.info(f"Passing error message back to model")
                    input.append({
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": error_message,
                    })
                    break
                if isinstance(tool, REPLTool):
                    result = await tool.run(repl=repl, **args)
                else:
                    result = await tool.run(**args)
                input.append({
                    "type": "function_call_output", 
                    "call_id": function_call.call_id, 
                    "output": result
                })
                logger.info("Added result to conversation")
            await asyncio.sleep(sleep)
        return RawTurnResult(
            response=response,
            trace=trace,
        )

# This seems to be the cleanest way to set up the reverse relationship since the client class is not defined yet when the nested class is defined.
AsyncResponsesAPIClient.ResponsesAPICallConfig.client_class = AsyncResponsesAPIClient

class AsyncMessagesAPIClient(AbstractAPIClient):
    """Client for the Anthropic Messages API."""

    @dataclass(kw_only=True)
    class MessagesAPICallConfig(AbstractAPIClient.CallConfig):
        cache_ttl: Union[str, None]

        @property
        def request_kwargs(self) -> dict:
            if self.system_prompt is not None:
                if hasattr(self.raw_request_kwargs, "system"):
                    raise ValueError("Config contains both system_prompt and request keyword argument 'system'")
                return {**self.raw_request_kwargs, "system": self.system_prompt}
            return self.raw_request_kwargs

    def __init__(self, api_provider: APIProvider, **kwargs):
        self.api_provider = api_provider
        api_key = os.getenv(api_provider.api_key_env_var) or kwargs.get("api_key")
        if api_key is None:
            raise ValueError(f"{api_provider.api_key_env_var} is not set")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    @classmethod
    async def call(cls, config: MessagesAPICallConfig, messages: list[dict], initial_code: Optional[list[str]] = None, **kwargs):
        if messages_contains_system_prompt(messages):
            raise ValueError("We don't allow system messages in messages when using Messages API. Use request param 'system' instead.")
        if not isinstance(config, cls.MessagesAPICallConfig):
            raise ValueError(f"Config must be an instance of {cls.MessagesAPICallConfig}")
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            if config.tools is None:
                result = await client.create_message(
                    model=config.model,
                    messages=messages,
                    **config.request_kwargs
                )
            else:
                result = await client.ensure_repl_and_run_tool_loop(
                    tools=config.tools,
                    sandbox_cls=config.sandbox_cls,
                    sandbox_kwargs=config.sandbox_kwargs,
                    initial_code=initial_code,
                    initial_code_timeout=config.initial_code_timeout,
                    model=config.model,
                    messages=messages,
                    max_retries=config.max_retries,
                    sleep=config.sleep,
                    cache_ttl=config.cache_ttl,
                    **config.request_kwargs
                )
            client.validate_response(result.response)
            text_blocks = client.extract_text_blocks(result.response)
            return TurnResult(
                role="assistant",
                content="\n".join(text_blocks),
                trace=result.trace
            )

    @staticmethod
    def get_readable_trace(
            trace: list[str],
            include_interpreter_output: bool = True,
            separator: str = "\n\n",
            include_prompt: bool = False,
            code_interpreter_tool_name: Optional[str] = None
            ) -> str:
        # https://huggingface.co/datasets/JoeYing/ReTool-SFT
        text = ""
        if include_prompt:
            first_request = json.loads(trace[0])
            messages_in_first_request = first_request.get("messages", [])
            num_user_prompts = len([msg for msg in messages_in_first_request if msg.get("role") == "user"])
            assert num_user_prompts <= 1, f"Expected less than or equal to 1 user prompt in the first request, but got {num_user_prompts}"
            for msg in messages_in_first_request:
                if msg.get("role") == "user":
                    content = msg["content"]
                    if isinstance(content, str):
                        content_text = content.strip()
                    elif isinstance(content, list):
                        # Content blocks format (used when caching is enabled)
                        text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
                        content_text = "\n".join(text_parts).strip()
                    text += f"<prompt>\n{content_text}\n</prompt>{separator}"
                    break
        
        for line_num in range(1, len(trace), 2):
            response = BetaMessage.model_validate_json(trace[line_num])
            
            # Process content blocks
            for block in response.content:
                block_type = block.type
                
                # Add thinking content
                if block_type == "thinking":
                    text += f"<think>\n{block.thinking.strip()}\n</think>{separator}"

                elif block_type == "redacted_thinking":
                    text += f"<think>\n[REDACTED THINKING]\n</think>{separator}"
                
                # Add text content
                elif block_type == "text":
                    text += f"{block.text.strip()}{separator}"
                
                # Handle tool use
                if code_interpreter_tool_name is None:
                    continue
                elif block_type == "tool_use":
                    tool_name = block.name
                    tool_id = block.id
                    tool_input = block.input
                    
                    if tool_name == code_interpreter_tool_name:
                        code = tool_input.get("code")
                        if code is not None:
                            text += f"<code>\n{code.strip()}\n</code>{separator}"
                        
                        if include_interpreter_output and line_num + 1 < len(trace):
                            next_request = json.loads(trace[line_num + 1])
                            next_messages = next_request.get("messages", [])
                            # Find the tool result for this tool call
                            for msg in next_messages:
                                content = msg.get("content", [])
                                if isinstance(content, list):
                                    for result_block in content:
                                        if (isinstance(result_block, dict) and 
                                            result_block.get("type") == "tool_result" and 
                                            result_block.get("tool_use_id") == tool_id):
                                            output = result_block.get("content", "")
                                            if output and len(output.strip()) > 0:
                                                text += f"<interpreter>\n{output.strip()}\n</interpreter>{separator}"
                                            break
        return text

    @staticmethod
    def get_token_consumption_from_trace(trace: list[str]) -> TokenConsumption:
        token_consumption = TokenConsumption(
            input=0,
            output=0,
            cached_input=0,
            total=0
        )
        for item in trace:
            try:
                response = BetaMessage.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption += AsyncMessagesAPIClient.token_consumption_from_response(response)
        return token_consumption

    @staticmethod
    def get_max_context_from_trace(trace: list[str]) -> int:
        for item in reversed(trace):
            try:
                response = BetaMessage.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption = AsyncMessagesAPIClient.token_consumption_from_response(response)
            return token_consumption.total
        return 0

    @staticmethod
    def token_consumption_from_response(response: BetaMessage) -> TokenConsumption:
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_creation = usage.cache_creation_input_tokens
        if cache_creation is None:
            cache_creation = 0
        cache_read = usage.cache_read_input_tokens
        if cache_read is None:
            cache_read = 0
        # Total input = input_tokens + cache_creation + cache_read
        # These fields are mutually exclusive per Anthropic's docs:
        # https://platform.claude.com/docs/en/build-with-claude/prompt-caching
        total_input = input_tokens + cache_creation + cache_read
        total = total_input + output_tokens
        return TokenConsumption(
            input=input_tokens,
            output=output_tokens,
            cached_input=cache_read,
            total=total,
        )

    @staticmethod
    def _make_json_serializable(messages: list[dict]) -> list[dict]:
        json_serializable_messages = []
        for message in messages:
            message_copy = copy.deepcopy(message)
            if "content" not in message_copy:
                json_serializable_messages.append(message_copy)
                continue
            content = message_copy["content"]
            if isinstance(content, list):
                converted_content = []
                for block in content:
                    # For interleaved thinking, we pass response.content as-is in the tool loop
                    if isinstance(block, BaseModel):
                        converted_content.append(json.loads(block.model_dump_json()))
                    else:
                        converted_content.append(block)
                message_copy["content"] = converted_content
            elif isinstance(content, BaseModel):    # Single PyDantic mode, shouldn't happen but keeping for safety
                message_copy["content"] = json.loads(content.model_dump_json())
            else:    # Could be a string
                message_copy["content"] = content
            json_serializable_messages.append(message_copy)
        return json_serializable_messages

    @staticmethod
    def validate_response(response: BetaMessage) -> None:
        stop_reason = response.stop_reason
        if stop_reason not in ["end_turn", "stop_sequence", "tool_use"]:
            raise InvalidMessageException(stop_reason)
    
    @staticmethod
    def _update_cache_breakpoint(messages: list[dict], cache_ttl: Union[str, None]) -> None:
        """Move cache_control to the last content block of the last message.
        
        This implements a "rolling cache" strategy where we keep one cache breakpoint
        that moves forward with each iteration of the tool loop. This ensures that
        the growing conversation prefix is cached, reducing rate limit impact.
        """
        if not messages:
            return
        
        # Remove cache_control from all previous user messages
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        del block["cache_control"]
        
        if cache_ttl is not None:
            # Add cache_control to the last block of the last message
            last_msg = messages[-1]
            content = last_msg.get("content", [])
            if isinstance(content, str):
                last_msg["content"] = [{"type": "text", "text": content}]
                content = last_msg["content"]
            if isinstance(content, list) and len(content) > 0:
                last_block = content[-1]
                if isinstance(last_block, dict):
                    last_block["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
    
    @staticmethod
    def extract_tool_use_blocks(response: BetaMessage):
        return [block for block in response.content if block.type == "tool_use"]
    
    @staticmethod
    def extract_thinking_blocks(response: BetaMessage) -> list[str]:
        thinking_texts = []
        try:
            for block in response.content:
                block_type = block.type
                if block_type == "thinking":
                    thinking_texts.append(block.thinking)
                elif block_type == "redacted_thinking":
                    thinking_texts.append("[REDACTED THINKING]")
        except Exception:
            logger.exception("Failed to extract thinking blocks")
        return thinking_texts
    
    @staticmethod
    def extract_text_blocks(response: BetaMessage) -> list[str]:
        texts = []
        try:
            for block in response.content:
                if block.type == "text":
                    texts.append(block.text)
        except Exception:
            logger.exception("Failed to extract text blocks")
        return texts
    
    def _response_contains_malformed_tool_calls(self, response: BetaMessage, tools: list[Tool]) -> bool:
        tool_use_blocks = self.extract_tool_use_blocks(response)
        for block in tool_use_blocks:
            try:
                tool = self.find_tool(block.name, tools)
                tool.validate_arguments(block.input)
            except Exception:
                logger.exception("Malformed tool_use found in Anthropic response")
                return True
        return False
    
    async def create_message(
        self,
        model: str,
        messages: list[dict],
        cache_ttl: Union[str, None],
        **kwargs: dict
    ) -> RawTurnResult:
        self._update_cache_breakpoint(messages, cache_ttl)
        if cache_ttl is not None:
            logger.info(f"Prompt caching enabled with TTL {cache_ttl}")
        else:
            logger.info(f"Prompt caching disabled")
            
        logger.info(f"Sending request to Anthropic model: {model}")
        
        # Use streaming API to keep connection alive during long-running requests (> 10 min)
        async with self.client.beta.messages.stream(model=model, messages=messages, **kwargs) as stream:
            response = await stream.get_final_message()
        
        # Logging
        thinking_texts = self.extract_thinking_blocks(response)
        text_contents = self.extract_text_blocks(response)
        for thinking in thinking_texts:
            logger.info(f"Reasoning content:\n{thinking}")
        for text_content in text_contents:
            logger.info(f"Response message content:\n{text_content}")
        try:
            logger.info(f"Tokens consumed: {self.token_consumption_from_response(response)}")
        except Exception:
            logger.exception("Failed to get token consumption from response")
        return RawTurnResult(
            response=response,
            trace=[
                json.dumps(
                    {"model": model, "messages": self._make_json_serializable(messages), **kwargs}
                ),    # TODO: Will fail if kwargs contain non-serializable objects.
                response.model_dump_json()
            ],
        )
    
    async def _run_tool_loop(
        self,
        tools: list[Tool],
        repl: Optional[REPL],
        model: str,
        messages: list[dict],
        max_retries: int,
        sleep: float,
        cache_ttl: Union[str, None],
        **kwargs: dict
    ) -> RawTurnResult:
        trace = []
        messages = copy.deepcopy(messages)
        api_tools = [tool.anthropic_schema for tool in tools]
        
        while True:
            retry_count = max_retries
            while True:         
                try:
                    result = await self.create_message(
                        model=model, 
                        messages=messages, 
                        tools=api_tools,
                        cache_ttl=cache_ttl,
                        **kwargs
                    )
                except AnthropicBadRequestError as e:
                    logger.exception("Bad request error during tool loop")
                    if retry_count > 0:
                        logger.info(f"Retrying")
                        retry_count -= 1
                        logger.info(f"Retry {max_retries - retry_count} of {max_retries} to avoid errors")
                        continue
                    else:
                        logger.info(f"Failed to avoid errors after {max_retries} retries")
                        raise e

                response = result.response
                # Check for malformed tool calls
                if self._response_contains_malformed_tool_calls(response, tools):
                    if retry_count > 0:
                        logger.info("Malformed tool calls found, retrying")
                        retry_count -= 1
                        logger.info(f"Retry {max_retries - retry_count} of {max_retries}")
                        continue
                    else:
                        logger.warning(f"Failed to avoid malformed tool calls after {max_retries} retries")
                        break
                break
            
            trace.extend(result.trace)
            self.validate_response(response)
            
            # Extract tool calls
            tool_use_blocks = self.extract_tool_use_blocks(response)
            if len(tool_use_blocks) == 0:
                logger.info("No tool calls in response")
                break
            logger.info(f"Found {len(tool_use_blocks)} tool call(s)")
            
            # Add assistant message with full content (including thinking blocks) to conversation
            # We need to preserve the content blocks as-is for multi-turn
            messages.append({
                "role": response.role, 
                "content": response.content
            })
            
            tool_results = []
            for tool_use_block in tool_use_blocks:
                tool_id = tool_use_block.id
                tool_name = tool_use_block.name
                args = tool_use_block.input
                
                logger.info(f"Model called tool: {tool_name}")
                
                try:
                    tool = self._find_tool(tool_name, tools)
                except ValueError as e:
                    error_message = str(e)
                    logger.exception(error_message)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": error_message,
                        "is_error": True
                    })
                    continue
                
                try:
                    tool.validate_arguments(args)
                except ValidationError as e:
                    error_message = str(e)
                    logger.exception(error_message)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": error_message,
                        "is_error": True
                    })
                    continue
                
                # Execute tool
                logger.info(f"Executing tool {tool_name} with args: {list(args.keys())}")
                if isinstance(tool, REPLTool):
                    result = await tool.run(repl=repl, **args)
                else:
                    result = await tool.run(**args)
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result
                })
                logger.info(f"Tool {tool_name} executed successfully")
            
            # Add tool results as user message
            messages.append({
                "role": "user",
                "content": tool_results
            })
            
            await asyncio.sleep(sleep)
        
        return RawTurnResult(
            response=response,
            trace=trace,
        )

# This seems to be the cleanest way to set up the reverse relationship since the client class is not defined yet when the nested class is defined.
AsyncMessagesAPIClient.MessagesAPICallConfig.client_class = AsyncMessagesAPIClient
