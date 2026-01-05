# General purpose utilities

import json
from pathlib import Path
import re


def read_file(file_path: Path, encoding: str = "utf-8") -> str:
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


def save_text(text: str, file_path: Path, encoding: str = "utf-8") -> None:
    with open(file_path, "w", encoding=encoding) as f:
        f.write(text)


def save_json(json_object: dict, file_path: Path, encoding: str = "utf-8") -> None:
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(json_object, f, indent=4)


def save_jsonl(json_strings: list[str], file_path: Path, encoding: str = "utf-8") -> None:
    with open(file_path, "w", encoding=encoding) as f:
        for json_string in json_strings:
            f.write(json_string + "\n")

def read_jsonl(file_path: Path, encoding: str = "utf-8") -> list[str]:
    with open(file_path, "r", encoding=encoding) as f:
        return [line.strip() for line in f if line.strip()]

def code_block_defines_function(code_block: str, function_name: str) -> bool:
    return re.search(r"def\s+{}\s*\(".format(function_name), code_block)

def get_code_block_defining_function_from_text(text: str, function_name: str) -> str:
    # 1) Look for fenced python blocks first (```python ...``` or ``` ...```)
    code_block_pattern = r"```(?:python)?(.+?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)

    for block in code_blocks:
        if code_block_defines_function(block, function_name):
            code = block.strip()
            return code

def get_markdown_heading(heading_level: int) -> str:
    return "#" * heading_level

def get_json_matches(text: str) -> list[re.Match[str]]:
    json_pattern = r"(?m)^```json\s*\n([\s\S]*?)^```"
    return list(re.finditer(json_pattern, text))

def get_code_matches(text: str) -> list[re.Match[str]]:
    code_pattern = r"(?m)^```python\s*\n([\s\S]*?)^```"
    return list(re.finditer(code_pattern, text))