# Common utilities for the chat completions and responses APIs

def replace_developer_message(messages, system_prompt):
    new_messages = []
    for message in messages:
        if message["role"] != "developer":
            new_messages.append(message)
    if system_prompt is not None:
        new_messages = [
            {"role": "developer", "content": system_prompt},
        ] + new_messages
    return new_messages

def messages_contains_system_prompt(messages):
    for message in messages:
        if message["role"] in ["developer", "system"]:
            return True
    return False