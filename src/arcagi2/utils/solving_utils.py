import ast
import logging
import json

from arcagi2.sandbox.base import Sandbox
from arcagi2.utils.config.base import CODE_TEMPLATES_FOLDER
from arcagi2.utils.puzzle_utils import get_copy_without_solutions
from arcagi2.utils.utils import read_file


logger = logging.getLogger(__name__)

async def solution_works_on_training_examples(
        sandbox_cls: Sandbox,
        puzzle: dict, 
        solution: str,
        max_retries: int,
        base_delay: int,
        delay_multiplier: float,
        max_delay: int,
        max_backoff_retries: int,
        timeout: float,
        **kwargs
        ) -> tuple[str, dict, list[int]]:
    check_solution_on_training_examples_code_template = read_file(
        CODE_TEMPLATES_FOLDER / "check_solution_on_training_examples.py"
    )
    cells = [
        f"puzzle={repr(get_copy_without_solutions(puzzle))}",
        solution,
        check_solution_on_training_examples_code_template,
        "print(' '.join(str(score) for score in results))",
    ]
    # There's an issue in IPyBox where the last cell's output is not always returned and we get None instead.
    attempt = 0
    while True:
        result = await sandbox_cls.run_cells(
            cells,
            timeout=timeout,
            base_delay=base_delay,
            delay_multiplier=delay_multiplier,
            max_delay=max_delay,
            max_backoff_retries=max_backoff_retries,
            **kwargs
        )
        score_output = result[-1].text    # This sometimes returns None
        if score_output is None:
            if attempt < max_retries:
                logger.warning(f"Hit the IPyBox bug: score_output is unexpectedly None. Retrying...")
                attempt += 1
                continue
            else:
                raise RuntimeError("Hit the IPyBox bug: score_output is unexpectedly None")
        else:
            break
    score = [int(score) for score in score_output.split()]
    return score

async def get_output_grid_from_solution(
        sandbox_cls: Sandbox,
        puzzle: dict, 
        solution: str,
        max_retries: int,
        base_delay: int,
        delay_multiplier: float,
        max_delay: int,
        max_backoff_retries: int,
        timeout: float,
        **kwargs
        ) -> list[list[int]]:
    produce_output_grid_code_template = read_file(
        CODE_TEMPLATES_FOLDER / "produce_output_grid.py"
    )
    cells = [
        f"puzzle={repr(get_copy_without_solutions(puzzle))}",
        solution,
        produce_output_grid_code_template,
        "import json\n\njson.dumps(out, indent=4)",
    ]
    attempt = 0
    while True:
        result = await sandbox_cls.run_cells(
            cells,
            timeout=timeout,
            base_delay=base_delay,
            delay_multiplier=delay_multiplier,
            max_delay=max_delay,
            max_backoff_retries=max_backoff_retries,
            **kwargs
        )
        output_grid = result[-1].text
        if output_grid is None:
            if attempt < max_retries:
                logger.warning(f"Hit the IPyBox bug: output_grid is unexpectedly None. Retrying...")
                attempt += 1
                continue
            else:
                raise RuntimeError("Hit the IPyBox bug: output_grid is unexpectedly None")
        else:
            break
    output_grid = json.loads(ast.literal_eval(output_grid))
    return output_grid