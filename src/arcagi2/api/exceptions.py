class InvalidChatCompletionException(Exception):
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason
        message = f"Invalid Chat Completions response with finish reason: {finish_reason}"
        super().__init__(message)

class InvalidResponseException(Exception):
    def __init__(self, status):
        self.status = status
        message = f"Invalid Responses API response with status: {status}"
        super().__init__(message)

class InvalidMessageException(Exception):
    def __init__(self, stop_reason):
        self.stop_reason = stop_reason
        message = f"Invalid Anthropic Message API response with stop reason: {stop_reason}"
        super().__init__(message)
