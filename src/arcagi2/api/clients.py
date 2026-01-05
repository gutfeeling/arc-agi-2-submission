import asyncio
import contextlib
import copy
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Union

import anthropic
from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic.types.beta import BetaMessage
# USE_DAYTONA=True: Daytona cloud sandboxes, False (default): ipybox Docker
if os.getenv("USE_DAYTONA", "False") == "True":
    from arcagi2.tools.execution_client_daytona import ExecutionClient, ExecutionContainer, ExecutionError
else:
    from ipybox import ExecutionClient, ExecutionContainer, ExecutionError
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
from arcagi2.tools.base import Tool
from arcagi2.tools.code_interpreter_ipybox import IPyBox
from arcagi2.utils.ipybox_utils import ensure_container
from arcagi2.utils.utils import read_file


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
    input: int = 0
    output: int = 0
    cached_input: int = 0    # default values for backward compatibility, when we didn't have cached_input
    total: int = 0
    
    def __add__(self, other):
        if isinstance(other, TokenConsumption):
            return TokenConsumption(
                input=self.input + other.input,
                output=self.output + other.output,
                cached_input=self.cached_input + other.cached_input,
                total=self.total + other.total
            )
        return NotImplemented
    
    def __iadd__(self, other):
        if isinstance(other, TokenConsumption):
            self.input += other.input
            self.output += other.output
            self.cached_input += other.cached_input
            self.total += other.total
            return self
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return TokenConsumption(
                input=int(self.input * other),
                output=int(self.output * other),
                cached_input=int(self.cached_input * other),
                total=int(self.total * other)
            )
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)

class AbstractAPIClient:

    @dataclass
    class CallConfig:
        api_provider: APIProvider
        model: str
        prompt_path: Path    # path to the prompt template
        # For non-streaming requests, the client is responsible for retrying timeouts that happens when model thinks too long. 
        # So don't reset `max_retries` in the client_kwargs to 0, unless you really want to disable retrying those.
        client_kwargs: dict = field(default_factory=dict)   # passed to the API client
        system_prompt_path: Union[Path, None] = None    # path to the system prompt template
        # kwargs is passed to the API request. Don't pass system prompts here (even though it is allowed in Responses and Messages APIs). 
        # Use the system_prompt_path property instead.
        raw_request_kwargs: dict = field(default_factory=dict)

        # Tool loop related configs
        tools: list[Tool] = field(default_factory=list)
        code_timeout: Union[int, None] = 120
        max_retries: int = 0

        @property
        def prompt(self):
            return read_file(self.prompt_path)

        @property
        def system_prompt(self):
            if isinstance(self.system_prompt_path, Path):
                return read_file(self.system_prompt_path)
            return self.system_prompt_path

        @property
        def request_kwargs(self):
            return self.raw_request_kwargs

    def __init__(self, api_provider, **kwargs):
        self.api_provider = api_provider
        api_key = os.getenv(api_provider.api_key_env_var)
        if api_key is None and kwargs.get("api_key") is None:
            raise ValueError(f"{api_provider.api_key_env_var} is not set")
        self.client = AsyncOpenAI(
            base_url=api_provider.base_url,
            api_key=api_key,
            **kwargs
        )

    async def __aenter__(self):
        # Supports context manager mode, just like the original AsyncOpenAI client
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.client.__aexit__(exc_type, exc_val, exc_tb)

    @classmethod
    async def call_without_tools(cls, messages, config: CallConfig) -> TurnResult:
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    async def call_with_tools(cls, messages, container_or_tag, initial_code, config: CallConfig) -> TurnResult:
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def token_consumption_from_response(response):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def get_token_consumption_from_trace(trace):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def get_max_context_from_trace(trace):
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def get_readable_trace(trace):
        raise NotImplementedError("Subclasses must implement this method")
    
    @staticmethod
    def validate_tools(tools):
        if not isinstance(tools, list):
            raise ValueError(f"Tools must be a list. Got {type(tools)}")
        if len(tools) == 0:
            raise ValueError(f"Tools list must not be empty")
        for tool in tools:
            if not isinstance(tool, Tool):
                raise ValueError(f"Tool {tool} must be an instance of Tool. Got {type(tool)}")    

    @staticmethod
    def find_tool(name, tools):
        for tool in tools:
            if tool.name == name:
                return tool
        raise ValueError(f"Tool name {name} not found in tools list {[tool.name for tool in tools]}")
            
    @staticmethod
    async def run_initial_code_ipybox(ipybox_client, initial_code):
        if len(initial_code) > 0:
            logger.info(f"Running initial code")
            for cell in initial_code:
                try:
                    logger.info(f"Running initial code cell:\n {cell}")
                    await ipybox_client.execute(code=cell)
                    logger.info(f"Cell ran successfully")
                except ExecutionError as e:
                    logger.exception(f"An error occurred during initial code execution: {e.trace}")
                    raise
            logger.info("Initial code ran successfully")
        else:
            logger.info("No initial code to run")

    def tools_contains_ipybox(self, tools):
        return any(isinstance(tool, IPyBox) for tool in tools)

class AsyncChatCompletionsAPIClient(AbstractAPIClient):

    @dataclass
    class ChatCompletionsAPICallConfig(AbstractAPIClient.CallConfig):
        pass

    @classmethod 
    async def call_without_tools(cls, messages, config):
        if not isinstance(config, cls.ChatCompletionsAPICallConfig):
            raise ValueError(f"Config must be an instance of {cls.ChatCompletionsAPICallConfig}")
        messages = replace_developer_message(
            messages=messages, 
            system_prompt=config.system_prompt
        )
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            result = await client.create_chat_completion(
                model=config.model,
                messages=messages,
                **config.request_kwargs
            )
            client.validate_response(result.response)
            return TurnResult(
                role=result.response.choices[0].message.role,
                content=result.response.choices[0].message.content,
                trace=result.trace
            )

    @classmethod
    async def call_with_tools(cls, messages, container_or_tag, initial_code, config):
        if not isinstance(config, cls.ChatCompletionsAPICallConfig):
            raise ValueError(f"Config must be an instance of {cls.ChatCompletionsAPICallConfig}")
        messages = replace_developer_message(
            messages=messages, 
            system_prompt=config.system_prompt
        )
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            async with contextlib.AsyncExitStack() as stack:
                container = await ensure_container(stack, container_or_tag)
                result = await client.create_chat_completion_with_tools(
                    model=config.model,
                    messages=messages,
                    tools=config.tools,
                    container=container,
                    initial_code=initial_code,
                    code_timeout=config.code_timeout,
                    max_retries=config.max_retries,
                    **config.request_kwargs
                )
            
            client.validate_response(result.response)
            return TurnResult(
                role=result.response.choices[0].message.role,
                content=result.response.choices[0].message.content,
                trace=result.trace
            )

    @staticmethod
    def get_token_consumption_from_trace(trace):
        token_consumption = TokenConsumption()
        for item in trace:
            try:
                response = ChatCompletion.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption += AsyncChatCompletionsAPIClient.token_consumption_from_response(response)
        return token_consumption

    @staticmethod
    def get_readable_trace(
            trace, 
            include_interpreter_output=True, 
            separator="\n\n",
            include_prompt=False,
            code_interpreter_tool_name=None
            ):
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
    def get_max_context_from_trace(trace):
        for item in reversed(trace):
            try:
                response = ChatCompletion.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption = AsyncChatCompletionsAPIClient.token_consumption_from_response(response)
            return token_consumption.total
        return 0
        
    
    @staticmethod
    def token_consumption_from_response(response):
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
    def validate_response(response):
        finish_reason = response.choices[0].finish_reason
        if response.choices[0].finish_reason not in ["stop", "tool_calls", "function_call"]:
            raise InvalidChatCompletionException(finish_reason)
        
    def make_messages_compatible(self, messages):
        """Deepseek, Gemini and Moonshot AI doesn't seem to support the new developer messages that replaced system messages in the OpenAI spec"""
        if self.api_provider.name in ["deepseek", "gemini", "moonshot"]:
            for message in messages:
                # Sometimes the messages list contains ChatCompletionMessage objects, because we use that to add the model's function calls to the history
                if isinstance(message, dict) and message["role"] == "developer":
                    message["role"] = "system"
        return messages
    
    @staticmethod
    def make_json_serializable(messages):
        json_serializable_messages = []
        for message in messages:
            if isinstance(message, ChatCompletionMessage):    # Used for sending back function calls back to the model
                message = json.loads(message.model_dump_json())
            json_serializable_messages.append(message)
        return json_serializable_messages

    def response_contains_malformed_tool_calls(self, response, tools):
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is None:
            return False
        for tool_call in tool_calls:
            try:
                tool = self.find_tool(tool_call.function.name, tools)
                args = json.loads(tool_call.function.arguments)
                tool.validate_arguments(args)
            except Exception as e:
                logger.exception(f"Malformed tool call")
                return True
        return False

    async def create_chat_completion(self, model, messages, **kwargs):
        messages = self.make_messages_compatible(messages)
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
                    {"model": model, "messages": self.make_json_serializable(messages), **kwargs}
                ),    # TODO: Will fail if kwargs contain non-serializable objects.
                response.model_dump_json()
            ],
        )
    
    async def run_tool_loop(
            self, 
            model, 
            messages, 
            tools, 
            client=None, 
            sleep=0, 
            code_timeout=120,
            max_retries=0,
            **kwargs
            ):
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
                if self.response_contains_malformed_tool_calls(result.response, tools):
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
                    tool = self.find_tool(tool_name, tools)
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
                if isinstance(tool, IPyBox):
                    result = await tool.run(client, timeout=code_timeout, **args)
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
            
    async def create_chat_completion_with_tools(
            self, 
            model, 
            messages, 
            tools=None,
            container=None,
            initial_code=[],
            code_timeout=120,
            max_retries=0,
            **kwargs
            ):
        self.validate_tools(tools)
        if self.tools_contains_ipybox(tools):
            # Use the same ipython kernel for all function calls to allow for stateful execution.
            async with ExecutionClient(port=container.executor_port) as ipybox_client: 
                await self.run_initial_code_ipybox(ipybox_client, initial_code)
                sleep=0
                if self.api_provider.name == "openai":
                    # Sometimes openai flags us for violating usage policies. Will a delay help?
                    sleep=10
                result = await self.run_tool_loop(
                    model, 
                    messages, 
                    tools, 
                    client=ipybox_client, 
                    sleep=sleep, 
                    code_timeout=code_timeout,
                    max_retries=max_retries,
                    **kwargs
                )
        else:
            result = await self.run_tool_loop(
                model, 
                messages, 
                tools,
                max_retries=max_retries,
                **kwargs
            )
        return result

# This seems to be the cleanest way to set up the reverse relationship since the client class is not defined yet when the nested class is defined.
AsyncChatCompletionsAPIClient.ChatCompletionsAPICallConfig.client_class = AsyncChatCompletionsAPIClient
                   
class AsyncResponsesAPIClient(AbstractAPIClient):

    @dataclass
    class ResponsesAPICallConfig(AbstractAPIClient.CallConfig):
        stateful: bool = False    # Set to True when using VLLM as the inference engine

        @property
        def request_kwargs(self):
            if self.system_prompt is not None:
                if hasattr(self.raw_request_kwargs, "instructions"):
                    raise ValueError("Config contains both system_prompt and request keyword argument 'instructions'")
                return {**self.raw_request_kwargs, "instructions": self.system_prompt}
            return self.raw_request_kwargs

    def __init__(self, api_provider, *args, **kwargs):
        # We use Responses API for accessing GPT OSS using VLLM
        if api_provider.name not in ["openai", "vllm"]:
            raise ValueError(f"Responses API is only supported by OpenAI and VLLM. {api_provider.name} is not supported.")
        super().__init__(api_provider, *args, **kwargs)

    @classmethod
    def validate_call_params(cls, messages, config):
        if messages_contains_system_prompt(messages):
            raise ValueError("We don't allow system messages in input when using Responses API. Use request param 'instructions' instead.")
        if not isinstance(config, cls.ResponsesAPICallConfig):
            raise ValueError(f"Config must be an instance of {cls.ResponsesAPICallConfig}")

    @classmethod
    async def call_without_tools(cls, messages, config):
        cls.validate_call_params(messages, config)
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            result = await client.create_response(
                model=config.model,
                input=messages,
                **config.request_kwargs
            )
            client.validate_response(result.response)
            return TurnResult(
                role="assistant",    # In Responses API, it's not safe to assume that response.output[0] is the text response from the model.
                content=result.response.output_text,
                trace=result.trace
            )

    @classmethod
    async def call_with_tools(cls, messages, container_or_tag, initial_code, config):
        cls.validate_call_params(messages, config)
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            async with contextlib.AsyncExitStack() as stack:
                container = await ensure_container(stack, container_or_tag)
                result = await client.create_response_with_tools(
                    model=config.model,
                    input=messages,
                    tools=config.tools,
                    container=container,
                    initial_code=initial_code,
                    code_timeout=config.code_timeout,
                    max_retries=config.max_retries,
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
            trace,
            include_interpreter_output=True,
            separator="\n\n",
            include_prompt=False,
            code_interpreter_tool_name=None
            ):
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
    def get_token_consumption_from_trace(trace):
        token_consumption = TokenConsumption()
        for item in trace:
            try:
                response = Response.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption += AsyncResponsesAPIClient.token_consumption_from_response(response)
        return token_consumption

    @staticmethod
    def get_max_context_from_trace(trace):
        for item in reversed(trace):
            try:
                response = Response.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption = AsyncResponsesAPIClient.token_consumption_from_response(response)
            return token_consumption.total
        return 0

    @staticmethod
    def token_consumption_from_response(response):
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
    def make_json_serializable(input_items):
        json_serializable_input_items = []
        for item in input_items:
            # Reasoning content sent back to model in stateless mode may contain objects of type ResponseOutputItem, which is a subscripted generic.
            # Instance check against ResponseOutputItem not possible as it is a subscripted generic
            if hasattr(item, "model_dump_json"):
                item = json.loads(item.model_dump_json())
            json_serializable_input_items.append(item)
        return json_serializable_input_items

    def tool_loop_response_is_invalid(self, response):
        function_calls = [el for el in response.output if getattr(el, "type", None) == "function_call"]
        if len(function_calls) == 0:
            try:
                self.validate_response(response)
            except InvalidResponseException:
                return True
        return False

    def response_contains_malformed_function_calls(self, response, tools):
        function_calls = [el for el in response.output if getattr(el, "type", None) == "function_call"]
        for function_call in function_calls:
            try:
                tool = self.find_tool(function_call.name, tools)
                args = json.loads(function_call.arguments)
                tool.validate_arguments(args)
            except Exception:
                logger.exception(f"Malformed function call")
                return True
        return False

    def response_contains_malformed_reasoning_content(self, response, model):
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
    def validate_response(response):
        status = response.status    
        if status != "completed":
            raise InvalidResponseException(status)
    
    def fix_reasoning_content(self, output, model):
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

    async def handle_background_response(self, response):
        while response.status in {"queued", "in_progress"}:
            logger.debug(f"Current status: {response.status}")

            await asyncio.sleep(2)  # Wait 2 seconds before polling again
            response = await self.client.responses.retrieve(response.id)
        logger.info(f"Final status: {response.status} (ID: {response.id})")
        return response
    
    async def create_response(self, model, input, **kwargs):
        logger.info(f"Sending request to model: {model}")
        response = await self.client.responses.create(
            model=model,
            input=input,
            **kwargs
        )
        
        if kwargs.get("background", False):
            logger.info(f"Background mode enabled, polling for response (ID: {response.id})")
            response = await self.handle_background_response(response)
        logger.info(f"Response output text:\n{response.output_text}")
        logger.info(f"Tokens consumed by this call: {self.token_consumption_from_response(response)}")
        return RawTurnResult(
            response=response,
            trace=[
                json.dumps(
                    {"model": model, "input": self.make_json_serializable(input), **kwargs}
                ),
                response.model_dump_json()
            ]
        )
    
    async def run_tool_loop(
            self, 
            model, 
            input, 
            tools, 
            client=None, 
            sleep=0, 
            code_timeout=120, 
            max_retries=0, 
            stateful=True, 
            **kwargs
            ):
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
                    self.response_contains_malformed_function_calls(result.response, tools) or 
                    self.tool_loop_response_is_invalid(result.response) or 
                    self.response_contains_malformed_reasoning_content(result.response, model)
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
                fixed_output = self.fix_reasoning_content(response.output, model)
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
                    tool = self.find_tool(tool_name, tools)
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
                if isinstance(tool, IPyBox):
                    result = await tool.run(client, timeout=code_timeout, **args)
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
    
    async def create_response_with_tools(
            self,
            model,
            input,
            tools,
            container=None,
            initial_code=[],
            code_timeout=120,
            max_retries=0,
            stateful=True,
            **kwargs
            ):
        self.validate_tools(tools)
        if self.tools_contains_ipybox(tools):
            # Use the same ipython kernel for all function calls to allow for stateful execution.
            async with ExecutionClient(port=container.executor_port) as ipybox_client: 
                await self.run_initial_code_ipybox(ipybox_client, initial_code)
                sleep=0
                if self.api_provider.name == "openai":
                    # Sometimes openai flags us for violating usage policies. Will a delay help?
                    sleep=10
                result = await self.run_tool_loop(
                    model, 
                    input, 
                    tools, 
                    client=ipybox_client, 
                    sleep=sleep, 
                    code_timeout=code_timeout, 
                    max_retries=max_retries, 
                    stateful=stateful, 
                    **kwargs
                )
        else:
            result = await self.run_tool_loop(
                model, 
                input, 
                tools, 
                max_retries=max_retries, 
                stateful=stateful, 
                **kwargs
            )
        return result

# This seems to be the cleanest way to set up the reverse relationship since the client class is not defined yet when the nested class is defined.
AsyncResponsesAPIClient.ResponsesAPICallConfig.client_class = AsyncResponsesAPIClient

class AsyncMessagesAPIClient(AbstractAPIClient):
    """Client for the Anthropic Messages API."""

    @dataclass
    class MessagesAPICallConfig(AbstractAPIClient.CallConfig):
        cache_ttl: Union[str, None] = "5m"    # 5m cache is suitable for interleaved thinking as model usually doesn't think for too long for each interleaved thinking block.

        @property
        def request_kwargs(self):
            if self.system_prompt is not None:
                if hasattr(self.raw_request_kwargs, "system"):
                    raise ValueError("Config contains both system_prompt and request keyword argument 'system'")
                return {**self.raw_request_kwargs, "system": self.system_prompt}
            return self.raw_request_kwargs

    def __init__(self, api_provider, **kwargs):
        self.api_provider = api_provider
        api_key = os.getenv(api_provider.api_key_env_var) or kwargs.get("api_key")
        if api_key is None:
            raise ValueError(f"{api_provider.api_key_env_var} is not set")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    @classmethod
    def validate_call_params(cls, messages, config):
        if messages_contains_system_prompt(messages):
            raise ValueError("We don't allow system messages in messages when using Messages API. Use request param 'system' instead.")
        if not isinstance(config, cls.MessagesAPICallConfig):
            raise ValueError(f"Config must be an instance of {cls.MessagesAPICallConfig}")

    @classmethod
    async def call_without_tools(cls, messages, config):
        cls.validate_call_params(messages, config)
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            result = await client.create_message(
                model=config.model,
                messages=messages,
                **config.request_kwargs
            )
            client.validate_response(result.response)
            
            text_blocks = client.extract_text_blocks(result.response)
            return TurnResult(
                role="assistant",
                content="\n".join(text_blocks),
                trace=result.trace
            )

    @classmethod
    async def call_with_tools(cls, messages, container_or_tag, initial_code, config):
        cls.validate_call_params(messages, config)
        async with cls(api_provider=config.api_provider, **config.client_kwargs) as client:
            async with contextlib.AsyncExitStack() as stack:
                container = await ensure_container(stack, container_or_tag)
                result = await client.create_message_with_tools(
                    model=config.model,
                    messages=messages,
                    tools=config.tools,
                    container=container,
                    initial_code=initial_code,
                    code_timeout=config.code_timeout,
                    max_retries=config.max_retries,
                    cache_ttl=config.cache_ttl,
                    **config.request_kwargs
                )
            
            client.validate_response(result.response)
            
            # Extract text content from response
            text_blocks = client.extract_text_blocks(result.response)
            return TurnResult(
                role="assistant",
                content="\n".join(text_blocks),
                trace=result.trace
            )

    @staticmethod
    def get_readable_trace(
            trace,
            include_interpreter_output=True,
            separator="\n\n",
            include_prompt=False,
            code_interpreter_tool_name=None
            ):
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
    def get_token_consumption_from_trace(trace):
        token_consumption = TokenConsumption()
        for item in trace:
            try:
                response = BetaMessage.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption += AsyncMessagesAPIClient.token_consumption_from_response(response)
        return token_consumption

    @staticmethod
    def get_max_context_from_trace(trace):
        for item in reversed(trace):
            try:
                response = BetaMessage.model_validate_json(item)
            except PydanticValidationError:
                continue
            token_consumption = AsyncMessagesAPIClient.token_consumption_from_response(response)
            return token_consumption.total
        return 0

    @staticmethod
    def token_consumption_from_response(response):
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
    def make_json_serializable(messages):
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
    def validate_response(response):
        stop_reason = response.stop_reason
        if stop_reason not in ["end_turn", "stop_sequence", "tool_use"]:
            raise InvalidMessageException(stop_reason)
    
    @staticmethod
    def _update_cache_breakpoint(messages, cache_ttl):
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
    def extract_tool_use_blocks(response):
        return [block for block in response.content if block.type == "tool_use"]
    
    @staticmethod
    def extract_thinking_blocks(response):
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
    def extract_text_blocks(response):
        texts = []
        try:
            for block in response.content:
                if block.type == "text":
                    texts.append(block.text)
        except Exception:
            logger.exception("Failed to extract text blocks")
        return texts
    
    def response_contains_malformed_tool_calls(self, response, tools):
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
        model,
        messages,
        **kwargs
    ):
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
                    {"model": model, "messages": self.make_json_serializable(messages), **kwargs}
                ),    # TODO: Will fail if kwargs contain non-serializable objects.
                response.model_dump_json()
            ],
        )
    
    async def run_tool_loop(
        self,
        model,
        messages,
        tools,
        client=None,
        sleep=0,
        code_timeout=120,
        max_retries=0,
        cache_ttl=None,
        **kwargs
    ):
        trace = []
        messages = copy.deepcopy(messages)
        api_tools = [tool.anthropic_schema for tool in tools]

        if cache_ttl is not None:
            logger.info(f"Prompt caching enabled with TTL {cache_ttl}")
        else:
            logger.info(f"Prompt caching disabled")
        
        while True:
            retry_count = max_retries
            while True:         
                if cache_ttl is not None:
                    self._update_cache_breakpoint(messages, cache_ttl)
                try:
                    result = await self.create_message(
                        model=model, 
                        messages=messages, 
                        tools=api_tools,
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
                if self.response_contains_malformed_tool_calls(response, tools):
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
                    tool = self.find_tool(tool_name, tools)
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
                if isinstance(tool, IPyBox):
                    result = await tool.run(client, timeout=code_timeout, **args)
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
    
    async def create_message_with_tools(
        self,
        model,
        messages,
        tools,
        container=None,
        initial_code=[],
        code_timeout=120,
        sleep=0,
        max_retries=0,
        cache_ttl=None,
        **kwargs
    ):
        self.validate_tools(tools)
        
        if self.tools_contains_ipybox(tools):
            async with ExecutionClient(port=container.executor_port) as ipybox_client:
                await self.run_initial_code_ipybox(ipybox_client, initial_code)
                result = await self.run_tool_loop(
                    model=model,
                    messages=messages,
                    tools=tools,
                    client=ipybox_client,
                    sleep=sleep,
                    code_timeout=code_timeout,
                    max_retries=max_retries,
                    cache_ttl=cache_ttl,
                    **kwargs,
                )
        else:
            result = await self.run_tool_loop(
                model=model,
                messages=messages,
                tools=tools,
                max_retries=max_retries,
                cache_ttl=cache_ttl,
                **kwargs,
            )
        
        return result

# This seems to be the cleanest way to set up the reverse relationship since the client class is not defined yet when the nested class is defined.
AsyncMessagesAPIClient.MessagesAPICallConfig.client_class = AsyncMessagesAPIClient
