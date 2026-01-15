import argparse
import asyncio
from dataclasses import fields
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.exceptions import MaxRetriesExceeded
from arcagi2.solver.config import SOLVER_CONFIGS
from arcagi2.solver.config.base import SolverConfig
from arcagi2.solver.turn import CodeSolutionTurn, GridSolutionTurn
from arcagi2.utils.logging_utils import per_task_file_logger, task_id_var
from arcagi2.utils.puzzle_utils import puzzle_to_text, get_copy_without_solutions
from arcagi2.utils.solving_utils import get_output_grid_from_solution, solution_works_on_training_examples
from arcagi2.utils.utils import read_file, save_json


logger = logging.getLogger(__name__)
    
async def solver(
        config: SolverConfig,
        puzzle_id: str,
        puzzle_json: dict,
        output_folder: Path,
        ):
    """
    Solves a single puzzle and saves submission.json and artifacts/logs to the output folder.
    See the "Submission file" section in https://www.kaggle.com/competitions/arc-prize-2025 for submission.json format specification.
    The solver only produces one attempt. The "attempt_2" key in submission.json holds None.
    
    If config.use_tools is False:
    - Runs the baseline solver (plain COT).
    
    If config.use_tools is True:
    - Runs the interleaved thinking solver.
    
    Output folder structure (use_tools=False):
    
        output_folder/
        ├── config.json                    # Solver configuration
        ├── run.log                        # Main log file
        ├── submission.json                # Final submission (baseline solver output)
        ├── 0/                             # Test 0
        │   ├── solution.json              # Output grid for test 0
        │   ├── cot.txt                    # Readable chain-of-thought output
        │   ├── trace.jsonl                # Full trace including requests and responses
        │   ├── token_consumption.json     # Token consumption information
        │   └── max_context.json           # Maximum context length used in the turn
        ├── 1/                             # Test 1
        │   └── ...
        └── ...                            # More tests

    Output folder structure (use_tools=True):
    
        output_folder/
        ├── config.json                    # Solver configuration
        ├── run.log                        # Main log file
        ├── submission.json                # Final submission in ARC Prize format
        ├── solution.py                    # Generated solution code
        ├── cot.txt                        # Readable interleaved thinking output including tool use
        ├── trace.jsonl                    # Full trace including requests and responses
        ├── token_consumption.json         # Token consumption information
        ├── max_context.json               # Maximum context length used in the turn
        └── score.json                     # Training example verification scores
    """
    if not output_folder.is_absolute():
        output_folder = Path.cwd() / output_folder
    # The evaluation script creates a metadata file in the output folder to track the status of the sample.
    # So output folder exists when we are here.
    output_folder.mkdir(parents=True, exist_ok=True)

    config_file_path = output_folder / "config.json"
    logger.info(f"Saving config to {config_file_path}")
    try:
        save_json(config.to_dict(), config_file_path)
    except Exception:
        logger.exception(f"Failed to save config to {config_file_path}")

    tok = task_id_var.set(str(output_folder))
    try:
        with per_task_file_logger(output_folder / "run.log"):
            submission_data = [
                {"attempt_1": None, "attempt_2": None} 
                for _ in range(len(puzzle_json["test"]))
            ]
            submission_file_path = output_folder / "submission.json"
            if not config.use_tools:
                grid_solution_turn = GridSolutionTurn(
                    config=config.call_config,
                    max_retries=config.max_retries,
                    base_delay=config.base_delay,
                    delay_multiplier=config.delay_multiplier,
                    max_delay=config.max_delay,
                    max_backoff_retries=config.max_backoff_retries,
                )
                for test_index in range(len(puzzle_json["test"])):
                    try:
                        logger.info("Running plain COT solver")
                        await grid_solution_turn.run(
                            prompt_template_vars={
                                "puzzle": puzzle_to_text(
                                    puzzle_json,
                                    markdown_level=2,
                                    separator="\n\n",
                                    include_test=True,
                                    include_all_tests=False,
                                    include_solutions=False,                                            
                                    )[test_index],
                            },
                            save_to_dir=output_folder / str(test_index),
                        )
                    except MaxRetriesExceeded:
                        logger.exception(f"Failed to get response from solver. All {config.max_retries} retry attempts has been exhausted")
                        continue
                    else:
                        solution_file_path = output_folder / str(test_index) / "solution.json"
                        if solution_file_path.exists():
                            output_grid = json.loads(read_file(solution_file_path))
                            submission_data[test_index]["attempt_1"] = output_grid
                        save_json(
                            {puzzle_id: submission_data},
                            submission_file_path
                        )
                return
            
            code_solution_turn = CodeSolutionTurn(
                config=config.call_config,
                max_retries=config.max_retries,
                base_delay=config.base_delay,
                delay_multiplier=config.delay_multiplier,
                max_delay=config.max_delay,
                max_backoff_retries=config.max_backoff_retries,
            )
            logger.info("Running interleaved thinking solver")
            code_solution_result = await code_solution_turn.run(
                prompt_template_vars={
                    "puzzle": puzzle_to_text(
                        puzzle_json,
                        markdown_level=2,
                        separator="\n\n",
                        include_test=True,
                        include_all_tests=True,
                        include_solutions=False
                    ),
                },
                save_to_dir=output_folder,
                initial_code=[f"puzzle={repr(get_copy_without_solutions(puzzle_json))}"],
            )

            solution = code_solution_result.solution

            score_on_training_examples = await solution_works_on_training_examples(
                sandbox_cls=config.sandbox_cls,
                puzzle=puzzle_json,
                solution=solution,
                max_retries=config.max_retries,
                base_delay=config.base_delay,
                delay_multiplier=config.delay_multiplier,
                max_delay=config.max_delay,
                max_backoff_retries=config.max_backoff_retries,
                timeout=config.code_timeout,
                **config.sandbox_kwargs,
            )

            save_json(score_on_training_examples, output_folder / "score.json")

            if not all(score == 1 for score in score_on_training_examples):
                logger.info(f"Solution does not work on all training examples. Not producing output grids")
                return

            output_grids = await get_output_grid_from_solution(
                sandbox_cls=config.sandbox_cls,
                puzzle=puzzle_json,
                solution=solution,
                max_retries=config.max_retries,
                base_delay=config.base_delay,
                delay_multiplier=config.delay_multiplier,
                max_delay=config.max_delay,
                max_backoff_retries=config.max_backoff_retries,
                timeout=config.code_timeout,
                **config.sandbox_kwargs,
            )
            for idx, output_grid in enumerate(output_grids):
                submission_data[idx]["attempt_1"] = output_grid
            logger.info(f"Saving output grids to {submission_file_path}")
            save_json(
                {puzzle_id: submission_data}, 
                submission_file_path
            )
    finally:
        task_id_var.reset(tok)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solves ARC AGI puzzles"
    )
    parser.add_argument(
        "-c", "--config_name",
        type=str,
        required=True,
        help="Configuration key. The configuration dict is defined in arcagi2.solver.config.SOLVER_CONFIGS."
    ) 
    parser.add_argument(
        "-p", "--puzzle_json_path", 
        type=str, 
        required=True,
        help="Path to the puzzle json file."
    )
    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        required=True,
        help="Folder to save the output"
    )
    parser.add_argument(
        "-b", "--vllm_base_url",
        type=str,
        help="Base URL of VLLM server"
    )
    parser.add_argument(
        "-e", "--env_file",
        type=str,
        default=".env",
        help="Path to environment file containing API keys and other configuration. Useful when running multiple instances of the script in parallel with different configurations (e.g. with different OpenAI accounts). Defaults to '.env'."
    )

    args = parser.parse_args()

    return args

def main_cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    root_logger = logging.getLogger()
    
    args = parse_arguments()

    root_logger.info(f"Loading environment variables from {args.env_file}")
    load_dotenv(args.env_file)


    if args.config_name not in SOLVER_CONFIGS:
        raise ValueError(f"Config '{args.config_name}' not found in SOLVER_CONFIGS. Available configs: {list(SOLVER_CONFIGS.keys())}")
    
    config = SOLVER_CONFIGS[args.config_name]
    root_logger.info(f"Using config: {args.config_name}")

    puzzle_path = Path(args.puzzle_json_path)
    if not puzzle_path.is_absolute():
        puzzle_path = Path.cwd() / puzzle_path
    puzzle_id = puzzle_path.stem
    puzzle_json = json.loads(read_file(puzzle_path))

    output_folder = Path(args.output_folder)

    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, AbstractAPIClient.CallConfig):
            call_config = value
            if call_config.api_provider.name == "vllm":
                call_config.api_provider.base_url = args.vllm_base_url

    asyncio.run(
        solver(
            config=config,
            puzzle_id=puzzle_id,
            puzzle_json=puzzle_json,
            output_folder=output_folder,
        )
    )

if __name__ == "__main__":
    main_cli()