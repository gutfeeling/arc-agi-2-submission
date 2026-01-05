from jsonschema import validate
from jsonschema.exceptions import ValidationError

class Tool:
    NAME = NotImplemented  # Use for setting default values for a tool, can be overridden in __init__
    DESCRIPTION = NotImplemented
    PARAMETERS = NotImplemented

    def __init__(
        self,
        name=None,
        description=None,
        parameters=None,
        strict=True,
        include_strict=True
    ):
        self.name = self.NAME
        self.description = self.DESCRIPTION
        self.parameters = self.PARAMETERS
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        self.strict = strict
        self.include_strict = include_strict

    @property
    def chat_completions_schema(self):
        function_dict = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.include_strict:    # TRL format or VLLM doesn't support strict, so our finetuned models don't need this key
            function_dict["strict"] = self.strict
        return {
            "type": "function",
            "function": function_dict,
        }

    @property
    def responses_api_schema(self):
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
        }

    @property
    def anthropic_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def run(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def validate_arguments(self, args: dict):
        try:
            validate(instance=args, schema=self.parameters)
        except ValidationError as e:
            e.message = f"Invalid tool parameters: {e.message}"
            raise e