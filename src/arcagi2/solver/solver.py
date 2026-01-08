import argparse
import asyncio
from dataclasses import dataclass, fields
import json
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.exceptions import MaxRetriesExceeded
from arcagi2.solver.choose import choose_best_solution, get_first_working_solution, choose_best_output_grids
from arcagi2.solver.config import SOLVER_CONFIGS
from arcagi2.solver.config.base import SolverConfig
from arcagi2.solver.turn import CodeSolutionTurn, GridSolutionTurn, SoftVerificationTurn
from arcagi2.solver.verification import verify_solution
from arcagi2.utils.logging_utils import per_task_file_logger, task_id_var
from arcagi2.utils.puzzle_utils import puzzle_to_text, get_copy_without_tests, get_copy_without_solutions
from arcagi2.utils.solving_utils import get_output_grid_from_solution
from arcagi2.utils.utils import save_text, read_file, save_json


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
    If the solver doesn't produce two attempts, the "attempt_2" key in submission.json will hold None.
    
    If config.use_tools is False:
    - Runs the baseline solver (plain COT).
    - Saves submission.json in the output folder root.
    
    If config.use_tools is True:
    - Runs the interleaved thinking solver.
    - Saves two submission files, one in the output folder root and one in the "submission/core" subfolder.
    - The submission file in the "submission/core" subfolder is the solution produced by the core system.
    - The submission file in the output folder root is the solution produced by the extended system. This is what we submit to the leaderboard.
    
    Output folder structure (use_tools=False):
    
        output_folder/
        ├── config.json                    # Solver configuration
        ├── run.log                        # Main log file
        ├── submission.json                # Final submission (baseline solver output)
        ├── sample_0/                      # First sample attempt
        │   ├── run.log                    # Sample-specific log
        │   ├── 0/                         # Test 0
        │   │   └── solution.json          # Output grid for test 0
        │   │   └── cot.txt                # Readable chain-of-thought output
        │   │   └── trace.jsonl            # Full trace including requests and responses
        │   │   └── token_consumption.json # Token consumption information
        │   │   └── max_context.json       # Maximum context length used in the turn
        │   ├── 1/                         # Test 1
        │   │   └── ...
        │   └── ...                        # More tests     
        ├── sample_1/                      # Second sample attempt
        │   └── ...
        └── ...                            # More sample attempts

    Output folder structure (use_tools=True):
    
        output_folder/
        ├── config.json                    # Solver configuration
        ├── run.log                        # Main log file
        ├── submission.json                # Final submission (extended system output) in ARC Prize format
        ├── sample_0/                      # First sample attempt
        │   ├── run.log                    # Sample-specific log
        │   ├── solution.py                # Generated solution code
        │   ├── cot.txt                    # Readable interleaved thinking output including tool use
        │   ├── trace.jsonl                # Full trace including requests and responses
        │   ├── token_consumption.json     # Token consumption information
        │   ├── max_context.json           # Maximum context length used in the turn
        │   └── verification/              # Verification results
        │       ├── verification.json      # Structured training example results, diff and coverage report
        │       └── soft_verification/     # Soft verification output (contains soft verification result, readable COT, trace and token information)
        │       └── ...                    # Readable diff and coverage report
        ├── sample_1/                      # Second sample attempt (if needed)
        │   └── ...
        ├── ...                            # More sample attempts
        └── submission/
            ├── core/                      # Core system (first solution that passes all training examples)
            │   ├── solution.py
            │   ├── metadata.json
            │   └── submission.json
            └── extended/                  # Extended system (core system + soft verification + generalization)
                ├── solution.py
                ├── solution_generalized.py
                ├── metadata.json
                └── generalize/            # Generalization attempt (contains generalized solution, readable COT, trace and token information)
    """
    if not output_folder.is_absolute():
        output_folder = Path.cwd() / output_folder
    # The evaluation script has a resume function, which requires creating the puzzle's submission folder and storing a temp file before running the solver.
    # So we set exist_ok=True here.
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
            for sample_index in range(config.num_samples):
                sample_folder = output_folder / f"sample_{sample_index}"
                sample_folder.mkdir(parents=True, exist_ok=False)

                tok_for_sample = task_id_var.set(str(sample_folder))
                try:
                    with per_task_file_logger(sample_folder / "run.log"):
                        if not config.use_tools:
                            grid_solution_turn = GridSolutionTurn(
                                config=config.plain_cot_solver,
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
                                        save_to_dir=sample_folder / str(test_index),
                                    )
                                except MaxRetriesExceeded as e:
                                    logger.exception(f"Failed to get response from solver. All {config.max_retries} retry attempts has been exhausted")
                                    continue
                            continue
                        code_solution_turn = CodeSolutionTurn(
                            config=config.interleaved_thinking_solver,
                            max_retries=config.max_retries,
                            base_delay=config.base_delay,
                            delay_multiplier=config.delay_multiplier,
                            max_delay=config.max_delay,
                            max_backoff_retries=config.max_backoff_retries,
                        )
                        try:
                            logger.info("Running interleaved thinking solver")
                            code_solution_result = await code_solution_turn.run(
                                prompt_template_vars={
                                    "puzzle": puzzle_to_text(
                                        puzzle_json,
                                        markdown_level=2,
                                        separator="\n\n",
                                        include_test=False,
                                        include_all_tests=False,
                                        include_solutions=False
                                    ),
                                },
                                save_to_dir=sample_folder,
                                initial_code=[f"puzzle={repr(get_copy_without_tests(puzzle_json))}"],
                            )
                        except MaxRetriesExceeded as e:
                            logger.exception(f"Failed to get response from solver. All {config.max_retries} retry attempts has been exhausted")
                            continue

                        solution = code_solution_result.solution

                        verification_result = await verify_solution(
                            config=config,
                            puzzle_json=puzzle_json,
                            solution=solution,
                            save_to_dir=sample_folder / "verification",
                        )

                        if not verification_result.all_training_examples_passed:
                            continue

                        soft_verification_turn = SoftVerificationTurn(
                            config=config.soft_verifier,
                            max_retries=config.max_retries,
                            base_delay=config.base_delay,
                            delay_multiplier=config.delay_multiplier,
                            max_delay=config.max_delay,
                            max_backoff_retries=config.max_backoff_retries,
                        )
                        initial_code = [f"puzzle={repr(get_copy_without_tests(puzzle_json))}"]
                        solution_description = ""
                        try:
                            await config.sandbox_cls.run_cells(
                                cells=initial_code + [solution],
                                timeout=120,
                                base_delay=config.base_delay,
                                delay_multiplier=config.delay_multiplier,
                                max_delay=config.max_delay,
                                max_backoff_retries=config.max_backoff_retries,
                                **config.sandbox_kwargs,
                            )
                        except:
                            logger.exception(f"Error running solution")
                            logger.info(f"We won't include the solution in the initial code")
                        else:
                            initial_code.append(solution)
                            solution_description = "\n- The existing `solution` function (that works on all training examples) along with all its helper functions are available."
                        try:
                            logger.info("Running soft verification")
                            soft_verification_result = await soft_verification_turn.run(
                                prompt_template_vars={
                                    "puzzle": puzzle_to_text(
                                        puzzle_json,
                                        markdown_level=2,
                                        separator="\n\n",
                                        include_test=False,
                                        include_all_tests=False,
                                        include_solutions=False,
                                    ),
                                    "solution": solution,
                                    "solution_description": solution_description,
                                    "coverage_report": verification_result.train_coverage_str,
                                },
                                save_to_dir=sample_folder / "verification" / "soft_verification",
                                initial_code=initial_code,
                            )
                        except MaxRetriesExceeded:
                            logger.exception(f"Failed to get response from soft verifier. All {config.max_retries} retry attempts has been exhausted")
                            continue
                        if soft_verification_result.passed:
                            break
                finally:
                    task_id_var.reset(tok_for_sample)
            if not config.use_tools:
                save_json(
                    {puzzle_id: choose_best_output_grids(puzzle_json, output_folder)},
                    output_folder / "submission.json"
                )
                return
            best_solution_path = choose_best_solution(output_folder)
            logger.info(f"Best solution path: {best_solution_path}")
            best_solution_sample_index = int(best_solution_path.parents[0].name.split("_")[-1])
            first_working_solution_path = get_first_working_solution(output_folder)
            logger.info(f"First working solution path: {first_working_solution_path}")
            first_working_solution_sample_index = int(first_working_solution_path.parents[0].name.split("_")[-1])
            extended_submission_folder = output_folder / "submission" / "extended"
            core_submission_folder = output_folder / "submission" / "core"
            if best_solution_path is not None:
                best_solution = read_file(best_solution_path)
                extended_submission_folder.mkdir(parents=True, exist_ok=False)
                solution_file_path = extended_submission_folder / "solution.py"
                logger.info(f"Saving best solution to {solution_file_path}")
                save_text(best_solution, solution_file_path)

                metadata_file_path = extended_submission_folder / "metadata.json"
                logger.info(f"Saving best solution metadata to {metadata_file_path}")
                save_json(
                    {"sample_index": best_solution_sample_index}, 
                    metadata_file_path
                )

                submission = [
                    {"attempt_1": None, "attempt_2": None}
                    for _ in range(len(puzzle_json["test"]))
                ]
                submission_file_path = output_folder / "submission.json"
                try:
                    output_grids = await get_output_grid_from_solution(
                        sandbox_cls=config.sandbox_cls,
                        puzzle=puzzle_json,
                        solution=best_solution,
                        max_retries=config.max_retries,
                        base_delay=config.base_delay,
                        delay_multiplier=config.delay_multiplier,
                        max_delay=config.max_delay,
                        max_backoff_retries=config.max_backoff_retries,
                        timeout=config.code_timeout,
                        **config.sandbox_kwargs,
                    )
                except Exception:
                    logger.exception(f"Error producing output grids from best solution")
                else:
                    for idx, output_grid in enumerate(output_grids):
                        submission[idx]["attempt_1"] = output_grid
                    logger.info(f"Saving output grids from best solution to {submission_file_path}")
                    save_json(
                        {puzzle_id: submission}, 
                        submission_file_path
                    )
                logger.info("Running generalization")
                code_solution_turn = CodeSolutionTurn(
                    config=config.generalizer,
                    max_retries=config.max_retries,
                    base_delay=config.base_delay,
                    delay_multiplier=config.delay_multiplier,
                    max_delay=config.max_delay,
                    max_backoff_retries=config.max_backoff_retries,
                )
                initial_code = [f"puzzle={repr(get_copy_without_solutions(puzzle_json))}"]
                solution_description = ""
                try:
                    await config.sandbox_cls.run_cells(
                        cells=initial_code + [best_solution],
                        timeout=120,
                        base_delay=config.base_delay,
                        delay_multiplier=config.delay_multiplier,
                        max_delay=config.max_delay,
                        max_backoff_retries=config.max_backoff_retries,
                        **config.sandbox_kwargs,
                    )
                except:
                    logger.exception(f"Error running solution")
                    logger.info(f"We won't include the solution in the initial code")
                else:
                    initial_code.append(best_solution)
                    solution_description = "\n- The existing `solution` (that works on all training examples) along with all its helper functions are available."

                try:
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
                            "solution": best_solution,
                            "solution_description": solution_description,
                        },
                        save_to_dir=extended_submission_folder / "generalize",
                        initial_code=initial_code,
                    )
                except MaxRetriesExceeded:
                    logger.exception(f"Failed to get response from generalizer. All {config.max_retries} retry attempts has been exhausted")
                else:
                    generalized_solution = code_solution_result.solution
                    save_text(generalized_solution, extended_submission_folder / "solution_generalized.py")
                    try:
                        output_grids = await get_output_grid_from_solution(
                            sandbox_cls=config.sandbox_cls,
                            puzzle=puzzle_json,
                            solution=generalized_solution,
                            max_retries=config.max_retries,
                            base_delay=config.base_delay,
                            delay_multiplier=config.delay_multiplier,
                            max_delay=config.max_delay,
                            max_backoff_retries=config.max_backoff_retries,
                            timeout=config.code_timeout,
                            **config.sandbox_kwargs,
                        )
                    except Exception:
                        logger.exception(f"Error producing output grids from generalized solution")
                    else:
                        for idx, output_grid in enumerate(output_grids):
                            submission[idx][f"attempt_2"] = output_grid
                        logger.info(f"Saving output grids from best and generalized solution to {submission_file_path}")
                        save_json(
                            {puzzle_id: submission}, 
                            submission_file_path
                        )
            if first_working_solution_path is not None:
                first_working_solution = read_file(first_working_solution_path)
                core_submission_folder.mkdir(parents=True, exist_ok=False)
                solution_file_path = core_submission_folder / "solution.py"
                logger.info(f"Saving first working solution to {solution_file_path}")
                save_text(first_working_solution, solution_file_path)

                metadata_file_path = core_submission_folder / "metadata.json"
                logger.info(f"Saving first working solution metadata to {metadata_file_path}")
                save_json(
                    {"sample_index": first_working_solution_sample_index}, 
                    metadata_file_path
                )

                try:
                    output_grids = await get_output_grid_from_solution(
                        sandbox_cls=config.sandbox_cls,
                        puzzle=puzzle_json,
                        solution=first_working_solution,
                        max_retries=config.max_retries,
                        base_delay=config.base_delay,
                        delay_multiplier=config.delay_multiplier,
                        max_delay=config.max_delay,
                        max_backoff_retries=config.max_backoff_retries,
                        timeout=config.code_timeout,
                        **config.sandbox_kwargs,
                    )
                except Exception:
                    logger.exception(f"Error producing output grids from first working solution")
                else:
                    submission = [
                        {"attempt_1": output_grids[test_idx], "attempt_2": None}
                        for test_idx in range(len(puzzle_json["test"]))
                    ]
                    core_submission_file_path = core_submission_folder / "submission.json"
                    logger.info(f"Saving core submission.json to {core_submission_file_path}")
                    save_json(
                        {puzzle_id: submission}, 
                        core_submission_file_path
                    )
    finally:
        task_id_var.reset(tok)

@dataclass
class SolverStatus:
    """
    Current status of a solver run, derived from output folder structure.
    
    Stage detection logic (in the specific order):
    - "generalization": submission/extended/ exists (solver completed, running generalization)
    - "verification": sample_N/verification/ exists (solution found, running verification)
    - "solution": sample_N/ exists but no verification yet (still generating solution)
    - None: no sample folders yet (solver just started)
    """
    sample_num: Optional[int]  # None if no samples yet
    stage: Optional[str]  # None if no samples, else "solution", "verification", or "generalization"

    @classmethod
    def from_output_folder(cls, output_folder: Path) -> "SolverStatus":
        sample_folders = sorted(output_folder.glob("sample_[0-9]*"), key=lambda p: int(p.name.split("_")[1]))
        sample_num = int(sample_folders[-1].name.split("_")[1]) if len(sample_folders) > 0 else None

        if sample_num is None:
            stage = None
        elif (output_folder / "submission" / "extended").exists():
            stage = "generalization"
        elif (output_folder / f"sample_{sample_num}" / "verification").exists():
            stage = "verification"
        else:
            stage = "solution"

        return cls(sample_num=sample_num, stage=stage)

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

    puzzle_json = json.load(open(args.puzzle_json_path))

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
            puzzle_json=puzzle_json,
            output_folder=output_folder,
        )
    )

if __name__ == "__main__":
    main_cli()