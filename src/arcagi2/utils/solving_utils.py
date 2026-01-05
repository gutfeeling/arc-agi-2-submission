import ast
import logging
import json
from typing import Union
import uuid
import os

# USE_DAYTONA=True: Daytona cloud sandboxes, False (default): ipybox Docker
if os.getenv("USE_DAYTONA", "False") == "True":
    from arcagi2.tools.execution_client_daytona import ExecutionContainer
else:
    from ipybox import ExecutionContainer

from arcagi2.utils.config.base import CODE_TEMPLATES_FOLDER
from arcagi2.utils.ipybox_utils import run_cells_ipybox
from arcagi2.utils.puzzle_utils import get_copy_without_solutions
from arcagi2.utils.utils import read_file


logger = logging.getLogger(__name__)

async def score_single_solution(
        puzzle: dict, 
        solution: str, 
        container_or_tag : Union[ExecutionContainer, str], 
        max_retries: int = 3,
        ) -> list[int]:
    check_solution_code_template = read_file(
        CODE_TEMPLATES_FOLDER / "check_solution_each_pair.py"
    )
    cells = [
        # sometimes solution relies on the presence of the variable `puzzles`. We remove the solutions to not leak answers.
        f"puzzle={repr(get_copy_without_solutions(puzzle))}",
        solution,
        f"tests={repr(puzzle['test'])}",
        check_solution_code_template,
    ]
    for attempt in range(max_retries):
        result = await run_cells_ipybox(
            cells, 
            container_or_tag
        )
        output = result[-1].text
        if output is None:
            if attempt + 1 < max_retries:
                logger.warning(f"Hit the IPyBox bug: last cell's output is unexpectedly None. Retrying...")
                continue
            else:
                raise RuntimeError("Hit the IPyBox bug: last cell's output is unexpectedly None")
        else:
            break
    
    last_line = output.splitlines()[-1]
    score = [int(score) for score in last_line.split()]
    return score

async def solution_works_on_training_examples(
        puzzle: dict, 
        solution: str, 
        container_or_tag : Union[ExecutionContainer, str], 
        max_retries: int = 3,
        ) -> tuple[str, dict, list[int]]:
    check_solution_on_training_examples_code_template = read_file(
        CODE_TEMPLATES_FOLDER / "check_solution_on_training_examples.py"
    )
    cells = [
        f"puzzle={repr(get_copy_without_solutions(puzzle))}",
        solution,
        check_solution_on_training_examples_code_template,
        "import json\n\njson.dumps(diff_info, indent=4)",
        "print(' '.join(str(score) for score in results))",
    ]
    # There's an issue in IPyBox where the last cell's output is not always returned and we get None instead.
    for attempt in range(max_retries):
        result = await run_cells_ipybox(
            cells, container_or_tag
        )
        diff_str_output = result[-3].text
        diff_info_output = result[-2].text    # Never saw a case where this is None, but still checking for safety
        score_output = result[-1].text    # This sometimes returns None
        if diff_str_output is None or diff_info_output is None or score_output is None:
            if attempt + 1 < max_retries:
                logger.warning(f"Hit the IPyBox bug: diff_str_output, diff_info_output, or score_output is unexpectedly None. Retrying...")
                continue
            else:
                raise RuntimeError("Hit the IPyBox bug: diff_str_output, diff_info_output, or score_output is unexpectedly None")
        else:
            break
    diff_str = diff_str_output.strip()
    diff_info = json.loads(ast.literal_eval(diff_info_output))
    score = [int(score) for score in score_output.split()]
    return diff_str, diff_info, score

async def get_coverage_report(
        puzzle: dict, 
        solution: str, 
        container_or_tag : Union[ExecutionContainer, str], 
        max_retries: int = 3,
        ) -> tuple[str, str, dict, dict]:
    # Container must have coverage installed
    analyze_coverage_code_template = read_file(
        CODE_TEMPLATES_FOLDER / "analyze_coverage.py"
    )
    random_id = uuid.uuid4().hex[:9]
    analyze_coverage_code_template = analyze_coverage_code_template.replace("{{random_id}}", random_id)
    cells = [
        f"puzzle={repr(get_copy_without_solutions(puzzle))}",
        f"%%writefile solution_code_{random_id}.py\n{solution}",
        analyze_coverage_code_template,
        "train_reports = generate_coverage_report(puzzle, data_type='train')",
        "test_reports = generate_coverage_report(puzzle, data_type='test')",
        "import json\n\njson.dumps(train_reports, indent=4)",
        "json.dumps(test_reports, indent=4)",
    ]
    # There's an issue in IPyBox where the last cell's output is not always returned and we get None instead.
    for attempt in range(max_retries):
        result = await run_cells_ipybox(
            cells, container_or_tag
        )
        train_reports_str_output = result[-4].text    # Never saw a case where this is None, but still checking for safety
        test_reports_str_output = result[-3].text    # Never saw a case where this is None, but still checking for safety
        train_reports_output = result[-2].text    # Never saw a case where this is None, but still checking for safety
        test_reports_output = result[-1].text    # Never saw a case where this is None, but still checking for safety
        if train_reports_str_output is None or test_reports_str_output is None or train_reports_output is None or test_reports_output is None:
            if attempt + 1 < max_retries:
                logger.warning(f"Hit the IPyBox bug: train_reports_str_output, test_reports_str_output, train_reports_output, or test_reports_output is unexpectedly None. Retrying...")
                continue
            else:
                raise RuntimeError("Hit the IPyBox bug: train_reports_str_output, test_reports_str_output, train_reports_output, or test_reports_output is unexpectedly None")
        else:
            break
    train_reports_str = train_reports_str_output.strip()
    test_reports_str = test_reports_str_output.strip()
    train_reports = json.loads(ast.literal_eval(train_reports_output))
    test_reports = json.loads(ast.literal_eval(test_reports_output))
    return train_reports_str, test_reports_str, train_reports, test_reports

def score_non_code_solution(
        puzzle: dict, 
        solution: str, 
        ) -> list[int]:
    score = [0 for _ in range(len(puzzle["test"]))]
    for idx, pair in enumerate(puzzle["test"]):
        try:
            score[idx] = 1 if pair["output"] == solution[str(idx)] else 0
        except Exception as e:
            logger.exception(f"Error scoring test index {idx}")
    return score

async def get_output_grid_from_solution(
        puzzle: dict, 
        test_idx: int, 
        solution: str, 
        container_or_tag : Union[ExecutionContainer, str], 
        max_retries: int = 3,
        ) -> list[list[int]]:
    produce_output_grid_code_template = read_file(
        CODE_TEMPLATES_FOLDER / "produce_output_grid.py"
    )
    assert test_idx >= 0 and test_idx < len(puzzle["test"]), "Test index out of range"
    cells = [
        f"puzzle={repr(get_copy_without_solutions(puzzle))}",
        solution,
        f"test_idx={test_idx}",
        produce_output_grid_code_template,
        "import json\n\njson.dumps(out, indent=4)",
    ]
    for attempt in range(max_retries):
        result = await run_cells_ipybox(
            cells, container_or_tag
        )
        output_grid = result[-1].text
        if output_grid is None:
            if attempt + 1 < max_retries:
                logger.warning(f"Hit the IPyBox bug: output_grid is unexpectedly None. Retrying...")
                continue
            else:
                raise RuntimeError("Hit the IPyBox bug: output_grid is unexpectedly None")
        else:
            break
    output_grid = json.loads(ast.literal_eval(output_grid))
    return output_grid