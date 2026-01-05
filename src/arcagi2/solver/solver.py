import argparse
import asyncio
from dataclasses import fields
import json
import logging 
from pathlib import Path

from dotenv import load_dotenv

import os
# USE_DAYTONA=True: Daytona cloud sandboxes, False (default): ipybox Docker
if os.getenv("USE_DAYTONA", "False") == "True":
    from arcagi2.tools.execution_client_daytona import ExecutionContainer, ExecutionError
else:
    from ipybox import ExecutionContainer, ExecutionError

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.solver.choose import choose_best_solution, get_first_working_solution, choose_best_output_grids
from arcagi2.solver.config import SYSTEM_CONFIG
from arcagi2.solver.exceptions import MaxRetriesExceeded
from arcagi2.solver.turn import CodeSolutionTurn, GridSolutionTurn, SoftVerificationTurn
from arcagi2.solver.verification import verify_solution
from arcagi2.utils.ipybox_utils import run_cells_ipybox
from arcagi2.utils.logging_utils import per_task_file_logger, task_id_var
from arcagi2.utils.puzzle_utils import puzzle_to_text, get_copy_without_tests, get_copy_without_solutions
from arcagi2.utils.utils import save_text, read_file, save_json


logger = logging.getLogger(__name__)
    
async def solver(
        config: AbstractAPIClient.CallConfig,
        puzzle_json: dict,
        output_folder: Path,    
        num_samples: int = 5,
        max_retries: int = 2,    # Max retries for avoidable model failures
        use_tools: bool = True,    # Set to False for plain COT baseline
        ):
    if not output_folder.is_absolute():
        output_folder = Path.cwd() / output_folder
    # The submission script has a resume function, which requires creating the puzzle's submission folder and storing a temp metadata file before running the solver.
    # So we set exist_ok=True here.
    output_folder.mkdir(parents=True, exist_ok=True)

    tok = task_id_var.set(str(output_folder))
    try:
        with per_task_file_logger(output_folder / "run.log"):
            for sample_index in range(num_samples):
                sample_folder = output_folder / f"sample_{sample_index}"
                sample_folder.mkdir(parents=True, exist_ok=False)

                tok_for_sample = task_id_var.set(str(sample_folder))
                try:
                    with per_task_file_logger(sample_folder / "run.log"):
                        if not use_tools:
                            grid_solution_turn = GridSolutionTurn(
                                config=config.plain_cot_solver,
                                max_retries=max_retries,
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
                                        use_tools=False,
                                        save_to_dir=sample_folder / str(test_index),
                                    )
                                except MaxRetriesExceeded as e:
                                    logger.exception(f"Failed to get response from solver. All {max_retries} retry attempts has been exhausted")
                                    continue
                            continue
                        # We run one container per sample because we have seen that the container might get unresponsive if we open many kernels
                        async with ExecutionContainer(tag=config.code_sandbox_container_tag) as container:
                            code_solution_turn = CodeSolutionTurn(
                                config=config.interleaved_thinking_solver,
                                max_retries=max_retries,
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
                                    use_tools=True,
                                    save_to_dir=sample_folder,
                                    container_or_tag=container,
                                    initial_code=[f"puzzle={repr(get_copy_without_tests(puzzle_json))}"],
                                )
                            except MaxRetriesExceeded as e:
                                logger.exception(f"Failed to get response from solver. All {max_retries} retry attempts has been exhausted")
                                continue

                            solution = code_solution_result.solution

                            verification_result = await verify_solution(
                                container_or_tag=container,
                                puzzle_json=puzzle_json,
                                solution=solution,
                                save_to_dir=sample_folder / "verification",
                            )

                            if not verification_result.all_training_examples_passed:
                                continue

                            soft_verification_turn = SoftVerificationTurn(
                                config=config.soft_verifier,
                                max_retries=max_retries,
                            )
                            initial_code = [f"puzzle={repr(get_copy_without_tests(puzzle_json))}"]
                            solution_description = ""
                            try:
                                await run_cells_ipybox(
                                    initial_code + [solution],
                                    container_or_tag=container,
                                )
                            except (Exception, ExecutionError):
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
                                    use_tools=True,
                                    save_to_dir=sample_folder / "verification" / "soft_verification",
                                    container_or_tag=container,
                                    initial_code=initial_code,
                                )
                            except MaxRetriesExceeded:
                                logger.exception(f"Failed to get response from soft verifier. All {max_retries} retry attempts has been exhausted")
                                continue
                            if soft_verification_result.passed:
                                break
                finally:
                    task_id_var.reset(tok_for_sample)
            if not use_tools:
                majority_voting_attempts = choose_best_output_grids(puzzle_json, output_folder)
                majority_voting_folder = output_folder / "majority_voting"
                majority_voting_folder.mkdir(parents=True, exist_ok=False)
                for idx, attempt in enumerate(majority_voting_attempts):
                    save_json(attempt, majority_voting_folder / f"attempt_{idx}.json")
                return
            best_solution_path = choose_best_solution(output_folder)
            logger.info(f"Best solution path: {best_solution_path}")
            first_working_solution_path = get_first_working_solution(output_folder)
            logger.info(f"First working solution path: {first_working_solution_path}")
            best_submission_folder = output_folder / "submission" / "best"
            first_submission_folder = output_folder / "submission" / "first"
            if best_solution_path is not None:
                best_solution = read_file(best_solution_path)
                best_submission_folder.mkdir(parents=True, exist_ok=False)
                best_solution_file_path = best_submission_folder / "solution.py"
                logger.info(f"Saving best solution to {best_solution_file_path}")
                save_text(best_solution, best_solution_file_path)

                code_solution_turn = CodeSolutionTurn(
                    config=config.generalizer,
                    max_retries=max_retries,
                )
                initial_code = [f"puzzle={repr(get_copy_without_solutions(puzzle_json))}"]
                solution_description = ""
                async with ExecutionContainer(tag=config.code_sandbox_container_tag) as container:
                    try:
                        await run_cells_ipybox(
                            initial_code + [best_solution],
                            container_or_tag=container,
                        )
                    except (Exception, ExecutionError):
                        logger.exception(f"Error running solution")
                        logger.info(f"We won't include the solution in the initial code")
                    else:
                        initial_code.append(best_solution)
                        solution_description = "\n- The existing `solution` function (that works on all training examples) along with all its helper functions are available."
                    logger.info("Running generalization")
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
                            use_tools=True,
                            save_to_dir=best_submission_folder / "generalize",
                            container_or_tag=container,
                            initial_code=initial_code,
                        )
                    except MaxRetriesExceeded:
                        logger.exception(f"Failed to get response from generalizer. All {max_retries} retry attempts has been exhausted")
                    else:
                        save_text(code_solution_result.solution, best_submission_folder / "solution_generalized.py")
            if first_working_solution_path is not None:
                first_working_solution = read_file(first_working_solution_path)
                first_submission_folder.mkdir(parents=True, exist_ok=False)
                first_working_solution_file_path = first_submission_folder / "solution.py"
                logger.info(f"Saving first working solution to {first_working_solution_file_path}")
                save_text(first_working_solution, first_working_solution_file_path)       
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
        help="Configuration key. The configuration dict is defined in arcagi2.solver.config.SYSTEM_CONFIG."
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
        "-n", "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate"
    )

    parser.add_argument(
        "-r", "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries for avoidable model failures"
    )

    parser.add_argument(
        "--plain_cot",
        action="store_true",
        help="Use plain COT baseline"
    )
    
    parser.add_argument(
        "--vllm_base_url",
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


    if args.config_name not in SYSTEM_CONFIG:
        raise ValueError(f"Config '{args.config_name}' not found in SYSTEM_CONFIG. Available configs: {list(SYSTEM_CONFIG.keys())}")
    
    config = SYSTEM_CONFIG[args.config_name]
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
            num_samples=args.num_samples,
            max_retries=args.max_retries,
            use_tools=not args.plain_cot,
        )
    )

if __name__ == "__main__":
    main_cli()