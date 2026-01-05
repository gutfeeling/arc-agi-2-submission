import argparse
import asyncio
from functools import partial
import json
import logging
from pathlib import Path
import time

from dotenv import load_dotenv
import os
# USE_DAYTONA=True: Daytona cloud sandboxes, False (default): ipybox Docker
if os.getenv("USE_DAYTONA", "False") == "True":
    from arcagi2.tools.execution_client_daytona import ExecutionContainer, ExecutionError
else:
    from ipybox import ExecutionContainer, ExecutionError
import pandas as pd

from arcagi2.solver.config import SYSTEM_CONFIG
from arcagi2.utils.logging_utils import (
    setup_logger_for_parallel_processing_script,
    ProgressCounter,
)
from arcagi2.utils.solving_utils import score_single_solution, score_non_code_solution
from arcagi2.utils.utils import read_file


logger = logging.getLogger(__name__)

def compute_score(result: list[list[bool]], k: int) -> float:
    if len(result) == 0:
        return 0
    first_k = result[:k]
    aggregated = [any(values) for values in zip(*first_k)]
    return sum(aggregated) / len(aggregated)

def create_score_summary(evaluation_data, results, k):
    score_df_data = [
        {
            "evaluation_item_id": evaluation_item_id,
            "submission_id": submission_id,
            "result": result,
            "duration_minutes": duration_seconds / 60,
            "iteration": iteration
        } for evaluation_item_id, submission_id, result, duration_seconds, iteration in results 
        if evaluation_item_id in evaluation_data
    ]
    score_df = pd.DataFrame(score_df_data)
    # Add missing evaluation items
    existing_ids = set()
    if len(score_df) > 0:
        existing_ids = set(score_df["evaluation_item_id"].values)
    missing_ids = [eid for eid in evaluation_data.keys() if eid not in existing_ids]
    if missing_ids:
        missing_df = pd.DataFrame({
            "evaluation_item_id": missing_ids,
            "submission_id": [None] * len(missing_ids),
            "result": [[] for _ in range(len(missing_ids))],
            "duration_minutes": [0] * len(missing_ids),
            "iteration": [-1] * len(missing_ids)
        })
        score_df = pd.concat([score_df, missing_df], ignore_index=True)
    logger.info(f"Score dataframe:\n{score_df.to_string()}")
    # aggregate by submission_id
    # result should be list addition of the results for each evaluation_item_id
    # token_consumption should be a list containing the token_consumption for each evaluation_item_id
    score_df = score_df.groupby("evaluation_item_id").agg({
        "result": "sum",
        "duration_minutes": "sum",
        "iteration": list,
        "submission_id": list
    }).reset_index()
    logger.info(f"Score dataframe after aggregation:\n{score_df.to_string()}")
    score_df["score"] = score_df["result"].apply(partial(compute_score, k=k))
    score_df = score_df.sort_values(by=["score", "duration_minutes"], ascending=[False, False])
    logger.info(f"Score dataframe after computing score:\n{score_df.to_string()}")
    logger.info(f"Total score: {score_df['score'].sum()} / {len(score_df)}")
    score = score_df["score"].mean()
    logger.info(f"Score: {score}")

async def main(
    evaluation_data_file,
    submission_folder,
    solution_folder_relative_path,
    non_code_solution,
    k,
    config_name,
    parallel,
    env_file,
):
    logger.info(f"Loading environment file from {env_file}")
    load_dotenv(env_file)

    if config_name not in SYSTEM_CONFIG:
        raise ValueError(f"Unknown config: {config_name}. Available configs: {list(SYSTEM_CONFIG.keys())}")
    config = SYSTEM_CONFIG[config_name]

    submission_folder = Path(submission_folder)
    if not submission_folder.is_absolute():
        submission_folder = Path.cwd() / submission_folder

    log_file = submission_folder / "score.log"
    setup_logger_for_parallel_processing_script(logger, log_file)
    logger.info(f"Logging score logs to {log_file}")

    logger.info(f"Loading evaluation data from {evaluation_data_file}")
    evaluation_data_file = Path(evaluation_data_file)
    if not evaluation_data_file.is_absolute():
        evaluation_data_file = Path.cwd() / evaluation_data_file
    evaluation_data = json.loads(read_file(evaluation_data_file))
    # Re-express evaluation data as a dictionary of puzzle IDs to puzzle JSONs
    evaluation_data = {item["metadata"]["id"]: item["puzzle"] for item in evaluation_data}

    tasks = []
    for subfolder in submission_folder.iterdir():
        if not subfolder.is_dir():
            continue
        metadata_file = subfolder / "metadata.json"
        if not metadata_file.exists():
            continue
        metadata = json.loads(read_file(metadata_file))
        item_id = metadata["id"]
        if item_id not in evaluation_data:
            continue
        duration_seconds = metadata["duration_seconds"]
        # find files starting with solution in the subfolder using glob
        # sort by name
        if solution_folder_relative_path is not None:
            solution_folder = subfolder / solution_folder_relative_path
        else:
            solution_folder = subfolder
        if not non_code_solution:
            solution_files = sorted([f for f in solution_folder.glob("solution*.py")], key=lambda x: x.name)
        else:
            solution_files = sorted([f for f in solution_folder.glob("attempt*.json")], key=lambda x: x.name)
        sorted_iterations = sorted([
            int(d.name.split("_")[-1])
            for d in subfolder.rglob("iteration_*")
        ])
        if len(sorted_iterations) > 0:
            iteration = sorted_iterations[-1]
        else:
            iteration = -1    # Special value for no iterations
        tasks.append(
            {
                "evaluation_item_id": item_id, 
                "submission_id": subfolder.name, 
                "solutions": [read_file(solution_file) for solution_file in solution_files],
                "duration_seconds": duration_seconds,
                "iteration": iteration,
                "non_code_solution": non_code_solution,
            }
        )

    async with ExecutionContainer(tag=config.container_tag) as container:
        semaphore = asyncio.Semaphore(parallel)
        progress_counter = ProgressCounter(len(tasks), logger)
        async def worker(evaluation_item_id, submission_id, solutions, duration_seconds, iteration, non_code_solution):
            async with semaphore:
                result = []
                for solution in solutions:
                    try:
                        if not non_code_solution:
                            score = await score_single_solution(
                                puzzle=evaluation_data[evaluation_item_id],
                                solution=solution,
                                container_or_tag=container,
                            )
                        else:
                            score = score_non_code_solution(
                                puzzle=evaluation_data[evaluation_item_id],
                                solution=json.loads(solution),
                            )
                    except (Exception, ExecutionError) as e:
                        logger.exception(f"Error scoring solution: {e}")
                        continue
                    result.append(score)
                await progress_counter.increment()
                return evaluation_item_id, submission_id, result, duration_seconds, iteration
        tasks = [
            worker(**kwargs) for kwargs in tasks
        ]
        logger.info("Starting scoring...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Results: {results}")
        logger.info(f"Scoring complete! {len(tasks)} items processed")

    create_score_summary(evaluation_data, results, k)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Score grid descriptions in submission files using a judge model"
    )

    parser.add_argument(
        "-d",
        "--evaluation_data_file",
        type=str,
        required=True,
        help="Path to the JSON file containing evaluation data with grid and metadata.",
    )


    parser.add_argument(
        "-s",
        "--submission_folder",
        type=str,
        required=True,
        help="Path to the folder containing the submission data.",
    )

    parser.add_argument(
        "-r",
        "--solution_folder_relative_path",
        type=str,
        help="Relative path of the folder containing the solutions to score. If none provided, we will look at the submission folder root",
    )

    parser.add_argument(
        "--non_code_solution",
        action="store_true",
        help="Whether to score non-code solutions. By default, we score only code solutions.",
    )

    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=1,
        help="Compute pass@k score",
    )

    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        default="default",
        help="Configuration key used for making the submission. The configuration dict is defined in arcagi2.solver.config.SOLVE_CONFIG.",
    )

    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Number of items to process in parallel. Set to 1 for sequential processing (default). Higher values increase parallelism.",
    )

    parser.add_argument(
        "-e",
        "--env_file",
        type=str,
        required=True,
        help="Path to the environment file to use for the scoring.",
    )

    args = parser.parse_args()

    if args.parallel < 1:
        raise ValueError("--parallel must be a positive integer (1 or greater)")

    return args

def main_cli():
    args = parse_arguments()
    
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.INFO)

    asyncio.run(
        main(
            evaluation_data_file=args.evaluation_data_file,
            submission_folder=args.submission_folder,
            solution_folder_relative_path=args.solution_folder_relative_path,
            non_code_solution=args.non_code_solution,
            k=args.k,
            config_name=args.config_name,
            parallel=args.parallel,
            env_file=args.env_file,
        )
    )

if __name__ == "__main__":
    main_cli()
