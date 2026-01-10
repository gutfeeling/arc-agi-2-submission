import argparse
import asyncio
from dataclasses import fields
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Optional
import uuid

from dotenv import load_dotenv

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.evaluation.dashboard import StatusDashboard
from arcagi2.evaluation.utils import EvaluationMetadata, EvaluationStatus
from arcagi2.solver.config import SOLVER_CONFIGS
from arcagi2.solver.solver import solver
from arcagi2.utils.logging_utils import setup_infra_logger_console_handler
from arcagi2.utils.utils import read_file, save_json


logger = logging.getLogger(__name__)


async def evaluate(
    challenge_file: str,
    config_name: str,
    output_folder: str,  
    vllm_base_url: str,
    parallel: int,
    submission_folder: str,
    puzzle_timeout_minutes: int,
    resume: bool,
    env_file: Optional[str] = None,    # allowing None because it's cumbersome to use dotenv in Kaggle. It's easier to just set os.environ directly from data in User Secrets.
) -> None:

    # Set up root logger to INFO level
    logging.getLogger().setLevel(logging.INFO)
    # Silence noisy HTTP client logs (Responses API background mode polling creates a lot of noise)
    # We save all logs to files, so we would need a lot of space. This is especially bad in Kaggle I guess.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Configure console output for infrastructure issues (rate limits, connection errors, sandbox failures)
    # Setting up here since Kaggle notebooks will use the function directly instead of running this file as a script.
    setup_infra_logger_console_handler(logging.WARNING)


    if env_file is not None:
        logger.info(f"Loading environment variables from {env_file}")
        load_dotenv(env_file)

    # Load config from SOLVE_CONFIG
    if config_name not in SOLVER_CONFIGS:
        raise ValueError(f"Config '{config_name}' not found in SOLVER_CONFIGS. Available configs: {list(SOLVER_CONFIGS.keys())}")
    
    config = SOLVER_CONFIGS[config_name]
    logger.info(f"Using config: {config_name}")

    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, AbstractAPIClient.CallConfig):
            call_config = value        
            if call_config.api_provider.name == "vllm":
                    call_config.api_provider.base_url = vllm_base_url

    output_folder = Path(output_folder)
    if not output_folder.is_absolute():
        output_folder = Path.cwd() / output_folder
    submission_folder = Path(submission_folder)
    if not submission_folder.is_absolute():
        submission_folder = Path.cwd() / submission_folder
    submission_file = submission_folder / "submission.json"
    if not resume:
        output_folder.mkdir(parents=True, exist_ok=False)
        submission_folder.mkdir(parents=True, exist_ok=True)
        save_json({}, submission_file)
    elif not output_folder.exists() or not submission_file.exists():
        raise ValueError(f"Output folder {output_folder} or submission file {submission_file} does not exist. Cannot resume from where we left off.")

    # Set up file-only logging
    log_file = output_folder / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    logger.info(f"Created output directory: {output_folder}")
    logger.info(f"Logging to {log_file}")

    logger.info(f"Loading challenge from {challenge_file}")
    challenge_file = Path(challenge_file)
    if not challenge_file.is_absolute():
        challenge_file = Path.cwd() / challenge_file

    challenge = json.loads(read_file(challenge_file))
    if resume:
        done = []
        for subfolder in output_folder.iterdir():
            if not subfolder.is_dir():
                continue
            metadata_file = subfolder / "metadata.json"
            if not metadata_file.exists():
                continue
            metadata = EvaluationMetadata.from_dict(
                json.loads(read_file(metadata_file))
            )
            if metadata.status in [EvaluationStatus.SUCCESS, EvaluationStatus.TIMEOUT]:
                done.append(metadata.puzzle_id)
        challenge = {
            puzzle_id: puzzle_json 
            for puzzle_id, puzzle_json in challenge.items() 
            if puzzle_id not in done
        }
    logger.info(f"Loaded {len(challenge)} puzzles for evaluation")

    total_puzzles = len(challenge)

    logger.info(f"Processing {total_puzzles} puzzles with max concurrency of {parallel}")

    # Create semaphore to control concurrency
    semaphore = asyncio.Semaphore(parallel)
    submission_write_lock = asyncio.Lock()

    # Start the live dashboard
    eval_start_time = datetime.now()
    dashboard = StatusDashboard(
        output_folder=output_folder,
        total_puzzles=total_puzzles,
        eval_start_time=eval_start_time,
        resume=resume,
        # Following params are hardcoded to sane values. If we need to change them, they can also become an argument of `evaluate` later.
        poll_interval=10,
        max_rows=10,
    )
    logger.info("Starting dashboard")
    dashboard_task = asyncio.create_task(dashboard.run())

    async def worker(puzzle_id: str, puzzle_json: dict) -> None:
        async with semaphore:
            # Create a folder with random ID
            submission_id = uuid.uuid4().hex[:9]
            item_folder = output_folder / submission_id
            item_folder.mkdir(parents=True, exist_ok=False)
            
            start_time = datetime.now()
            metadata = EvaluationMetadata(
                submission_id=submission_id,
                puzzle_id=puzzle_id,
                start_time=start_time,
                end_time=None,
                duration_seconds=None,
                status=EvaluationStatus.RUNNING,
            )
            logger.info(f"Starting submission {submission_id} for puzzle {puzzle_id}")
            save_json(metadata.to_dict(), item_folder / "metadata.json")
            
            try:
                await asyncio.wait_for(
                    solver(
                        config=config,
                        puzzle_id=puzzle_id,
                        puzzle_json=puzzle_json,
                        output_folder=item_folder,
                    ),
                    timeout=puzzle_timeout_minutes * 60
                )    # This will create a submission.json file in the output folder if not timed out
                # Update submission file with new puzzle submission
                puzzle_submission_file = item_folder / "submission.json"
                if puzzle_submission_file.exists():
                    puzzle_submission = json.loads(read_file(puzzle_submission_file))
                    async with submission_write_lock:
                        current_submission = json.loads(read_file(submission_file))
                        save_json(
                            {**current_submission, **puzzle_submission}, 
                            submission_file
                        )
                # Success
                metadata.end_time = datetime.now()
                metadata.duration_seconds = (metadata.end_time - start_time).total_seconds()
                metadata.status = EvaluationStatus.SUCCESS
                save_json(metadata.to_dict(), item_folder / "metadata.json")
                logger.info(f"Submission {submission_id} for puzzle {puzzle_id} completed in {metadata.duration_seconds / 60:.2f} minutes")

            except asyncio.TimeoutError:
                metadata.end_time = datetime.now()
                metadata.duration_seconds = (metadata.end_time - start_time).total_seconds()
                metadata.status = EvaluationStatus.TIMEOUT
                save_json(metadata.to_dict(), item_folder / "metadata.json")
                logger.warning(f"Submission {submission_id} timed out after {puzzle_timeout_minutes} minutes")

            except Exception as e:
                metadata.end_time = datetime.now()
                metadata.duration_seconds = (metadata.end_time - start_time).total_seconds()
                metadata.status = EvaluationStatus.ERROR
                save_json(metadata.to_dict(), item_folder / "metadata.json")
                logger.exception(f"Submission {submission_id} failed due to exception: {e}")

    tasks = [
        asyncio.create_task(worker(puzzle_id, puzzle_json)) for puzzle_id, puzzle_json in challenge.items()
    ]

    logger.info("Starting processing...")
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        
        logger.info("Either all tasks completed or an exception was detected. We will try to exit.")
        for t in tasks:
            t.cancel()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if any task raised an exception
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception detected: {result}. All tasks have been cancelled.")
                raise result
        
        logger.info(
            f"Processing complete! {total_puzzles} puzzles processed"
        )
        logger.info("Evaluation completed successfully")
        dashboard.update()
    except Exception as e:
        dashboard.print_error(e)
        raise
    finally:
        dashboard_task.cancel()
        await asyncio.gather(dashboard_task, return_exceptions=True)
        
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate models on solving ARC AGI puzzles"
    )
    parser.add_argument(
        "--challenge_file",
        type=str,
        required=True,
        help="Path to the challenge JSON file in ARC Prize format",
    )
    parser.add_argument(
        "-c", "--config_name",
        type=str,
        required=True,
        help="Configuration key. The configuration dict is defined in arcagi2.solver.config.SOLVE_CONFIG."
    )
    parser.add_argument(
        "-o", "--output_folder",
        required=True,
        help="Folder to save detailed output and logs"
    ) 
    parser.add_argument(
        "-b", "--vllm_base_url",    # Only used if config uses VLLM
        type=str,
        help="Base URL of VLLM server"
    )
    parser.add_argument(
        "-s", "--submission_folder",
        type=str,
        required=True,
        help="Folder to save submission.json"
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        required=True,
        help="Number of puzzles to process in parallel. Set to 1 for sequential processing (default). Higher values increase parallelism."
    )
    parser.add_argument(
        "-t", "--puzzle_timeout_minutes",
        type=int,
        required=True,
        help="Timeout in minutes for each puzzle"
    )
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume from where we left off. If a puzzle's output folder doesn't have a submission.json, then it means that it didn't finish, so we should retry it."
    )
    parser.add_argument(
        "-e", "--env_file",
        type=str,
        required=True,
        help="Path to environment file containing API keys and other configuration."
    )
    args = parser.parse_args()

    return args

def main_cli() -> None:
    args = parse_arguments()

    asyncio.run(
        evaluate(
            challenge_file=args.challenge_file,
            config_name=args.config_name,
            output_folder=args.output_folder,
            vllm_base_url=args.vllm_base_url,
            parallel=args.parallel,
            submission_folder=args.submission_folder,
            puzzle_timeout_minutes=args.puzzle_timeout_minutes,
            resume=args.resume,
            env_file=args.env_file,
        )
    )

if __name__ == "__main__":
    main_cli()
