# Common utilities for the chat completions and responses APIs

from typing import Optional


def replace_developer_message(messages: list[dict], system_prompt: Optional[str]) -> list[dict]:
    new_messages = []
    for message in messages:
        if message["role"] != "developer":
            new_messages.append(message)
    if system_prompt is not None:
        new_messages = [
            {"role": "developer", "content": system_prompt},
        ] + new_messages
    return new_messages

def messages_contains_system_prompt(messages: list[dict]) -> bool:
    for message in messages:
        if message["role"] in ["developer", "system"]:
            return True
    return False