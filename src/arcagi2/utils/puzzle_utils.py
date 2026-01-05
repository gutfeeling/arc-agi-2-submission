from typing import Union

import numpy as np

from arcagi2.utils.utils import get_markdown_heading


def grid_to_text(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(map(str, row)) for row in grid)

def text_to_grid(text: str) -> list[list[int]]:
    lines = text.strip().split("\n")
    return np.array([list(map(int, line.split())) for line in lines])

def puzzle_to_text(
    puzzle: dict,
    markdown_level: int = 1,
    separator: str = "\n\n",
    include_test: bool = True,
    include_all_tests: bool = True,
    include_solutions: bool = False,
) -> Union[str, list[str]]:
    training_text = get_markdown_heading(markdown_level) + f" Training examples"
    for idx, train_example in enumerate(puzzle["train"]):
        train_example_num = idx
        training_text += f"{separator}{get_markdown_heading(markdown_level + 1)} Training example {train_example_num}"
        training_text += (
            f"{separator}Input:{separator}{grid_to_text(train_example['input'])}"
        )
        training_text += (
            f"{separator}Output:{separator}{grid_to_text(train_example['output'])}"
        )

    if not include_test:
        return training_text

    testing_heading_singular = f"{get_markdown_heading(markdown_level)} Test"
    testing_heading_plural = testing_heading_singular + "s"

    num_tests = len(puzzle["test"])
    testing_text = []
    for idx, test_example in enumerate(puzzle["test"]):
        test_example_num = idx
        text = ""
        if include_all_tests and num_tests > 1:
            text += (
                f"{get_markdown_heading(markdown_level + 1)} Test {test_example_num}{separator}"
            )
        text += f"Input:{separator}{grid_to_text(test_example['input'])}"
        if include_solutions:
            text += (
                f"{separator}Output:{separator}{grid_to_text(test_example['output'])}"
            )
        testing_text.append(text)
    if include_all_tests:
        return f"{training_text}{separator}{testing_heading_plural if num_tests > 1 else testing_heading_singular}{separator}{separator.join(testing_text)}"
    return [
        f"{training_text}{separator}{testing_heading_singular}{separator}{text}"
        for text in testing_text
    ]

def get_copy_without_tests(puzzle: dict) -> dict:
    return {
        "train": puzzle["train"],
        "test": [],
    }

def get_copy_without_solutions(puzzle: dict) -> dict:
    return {
        "train": puzzle["train"],
        "test": [{"input": test["input"]} for test in puzzle["test"]],
    }
