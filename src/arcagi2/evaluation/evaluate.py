import argparse
import asyncio
from dataclasses import dataclass, fields
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Optional
import uuid

from dotenv import load_dotenv

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.evaluation.dashboard import StatusDashboard
from arcagi2.evaluation.utils import Metadata, EvaluationStatus, sort_by_majority
from arcagi2.solver.config import SOLVER_CONFIGS
from arcagi2.solver.config.base import SolverConfig
from arcagi2.solver.solver import solver
from arcagi2.utils.logging_utils import setup_infra_logger_console_handler
from arcagi2.utils.utils import read_file, save_json


logger = logging.getLogger(__name__)

@dataclass
class MajorityVoteResult:
    submission: dict[str, list[dict]]
    stop: bool

def get_majority_voted_submission(
    results: dict[str, list[dict]],
    puzzle_id: str,
    puzzle_json: dict,
) -> MajorityVoteResult:
    """
    Get the majority voted submission from the results and make an early stopping decision.
    
    Early stopping logic: 
    - For the interleaved thinking solver, we check if two generalized solutions agree for all tests.
    - For plain COT solver, we check if any two solutions agree for all tests.
    Since both generalized solution and the plain COT solution uses "attempt_1", checking if two "attempt_1"s agree works in both cases.

    Majority voting logic:
    - For the plain COT solver, we just use usual majority voting.
    - For the interleaved thinking solver, we do the following:
        - Output the highest voted generalized solution as "attempt_1" if it got >=2 votes.
        - For any remaining attempts, we do usual majority voting between the remaining outputs, with preference given to generalized solutions in case of a tie.
    This can also be encapsulated in a common logic that works for both plain COT and interleaved thinking.
    """
    submission = [{"attempt_1": None, "attempt_2": None} for _ in range(len(puzzle_json["test"]))]
    stop = True
    for test_index in range(len(puzzle_json["test"])):
        excluded = None
        attempt_1_outputs = [
            result[puzzle_id][test_index]["attempt_1"] for result in results if result[puzzle_id][test_index]["attempt_1"] is not None
        ]
        sorted_attempt_1_outputs = sort_by_majority(attempt_1_outputs)
        if len(sorted_attempt_1_outputs) == 0:
            stop = False
        else:
            votes = sorted_attempt_1_outputs[0][1]
            if votes >= 2:
                submission[test_index]["attempt_1"] = sorted_attempt_1_outputs[0][0]
                excluded = sorted_attempt_1_outputs[0][0]
            else:
                stop = False
        attempt_2_outputs = [
            result[puzzle_id][test_index]["attempt_2"] for result in results if result[puzzle_id][test_index]["attempt_2"] is not None
        ]
        all_other_outputs = [grid for grid in attempt_1_outputs + attempt_2_outputs if grid != excluded]
        sorted_all_other_outputs = sort_by_majority(all_other_outputs)
        for attempt_idx in range(2):
            if submission[test_index][f"attempt_{attempt_idx + 1}"] is None:
                try:
                    submission[test_index][f"attempt_{attempt_idx + 1}"] = sorted_all_other_outputs.pop(0)[0]
                except IndexError:
                    pass
    return MajorityVoteResult(
        submission={puzzle_id: submission}, 
        stop=stop
    )

async def sample_worker(
        semaphore: asyncio.Semaphore, 
        lock: asyncio.Lock,
        config: SolverConfig, 
        puzzle_id: str, 
        puzzle_json: dict,
        puzzle_output_folder: Path, 
        ):
    async with semaphore:
        sample_id = uuid.uuid4().hex[:9]
        try:
            sample_output_folder = puzzle_output_folder / sample_id
            sample_output_folder.mkdir(parents=True, exist_ok=False)
            sample_metadata = Metadata(
                uid=sample_id,
                puzzle_id=puzzle_id,
                start_time=datetime.now(),
                end_time=None,
                duration_seconds=None,
                status=EvaluationStatus.RUNNING
            )
            async with lock:
                save_json(
                    sample_metadata.to_dict(), 
                    sample_output_folder / "metadata.json"
                )
            result = await solver(
                config=config,
                puzzle_id=puzzle_id,
                puzzle_json=puzzle_json,
                output_folder=sample_output_folder,
            )
        except Exception:
            logger.exception(f"Error in sample worker for puzzle {puzzle_id} and sample {sample_id}")
            try:
                sample_metadata.status = EvaluationStatus.ERROR
                sample_metadata.end_time = datetime.now()
                sample_metadata.duration_seconds = (sample_metadata.end_time - sample_metadata.start_time).total_seconds()
                async with lock:
                    save_json(
                        sample_metadata.to_dict(), 
                        sample_output_folder / "metadata.json"
                    )
            except Exception:
                logger.exception(f"Error in saving sample metadata for puzzle {puzzle_id} and sample {sample_id} after sample worker ran into an error")
            raise
        except asyncio.CancelledError:
            try:
                sample_metadata.status = EvaluationStatus.CANCELLED
                sample_metadata.end_time = datetime.now()
                sample_metadata.duration_seconds = (sample_metadata.end_time - sample_metadata.start_time).total_seconds()
                save_json(
                    sample_metadata.to_dict(), 
                    sample_output_folder / "metadata.json"
                )
            except Exception:
                pass
            finally:
                raise
        else:
            sample_metadata.status = EvaluationStatus.SUCCESS
            sample_metadata.end_time = datetime.now()
            sample_metadata.duration_seconds = (sample_metadata.end_time - sample_metadata.start_time).total_seconds()
            async with lock:
                save_json(
                    sample_metadata.to_dict(), 
                    sample_output_folder / "metadata.json"
                )
            return result

async def puzzle_worker(
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    config: SolverConfig,
    puzzle_id: str,
    puzzle_json: dict,
    evaluation_output_folder: Path,
    submission_file: Path,
    num_samples: int,
):
    uid = uuid.uuid4().hex[:9]
    pending = []
    try:
        puzzle_output_folder = evaluation_output_folder / uid
        puzzle_output_folder.mkdir(parents=True, exist_ok=False)
        metadata = Metadata(
            uid=uid,
            puzzle_id=puzzle_id,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            status=EvaluationStatus.RUNNING
        )
        async with lock:
            save_json(
                metadata.to_dict(), 
                puzzle_output_folder / "metadata.json"
            )

        num_finished_tasks = 0
        all_results = []
        pending = [
            asyncio.create_task(
                sample_worker(
                    semaphore=semaphore,
                    lock=lock,
                    config=config,
                    puzzle_id=puzzle_id,
                    puzzle_json=puzzle_json,
                    puzzle_output_folder=puzzle_output_folder,
                )
            ) for _ in range(min(2, num_samples))
        ]
        while True:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            num_finished_tasks += len(done)
            all_results.extend(
                [t.result() for t in done if not t.cancelled() and t.exception() is None]
            )
            majority_vote_result = get_majority_voted_submission(
                results=all_results, 
                puzzle_id=puzzle_id,
                puzzle_json=puzzle_json
            )
            async with lock:
                submission = json.loads(read_file(submission_file))
                submission = {**submission, **majority_vote_result.submission}
                save_json(submission, submission_file)
            if majority_vote_result.stop or num_finished_tasks >= num_samples:    # early stopping or max samples reached
                break
            elif len(pending) > 0:    # some tasks are still running, wait for them to finish
                continue
            else:    # nothing is running and stopping criteria not reached, start a new sample
                pending.add(
                    asyncio.create_task(
                        sample_worker(
                            semaphore=semaphore,
                            lock=lock,
                            config=config,
                            puzzle_id=puzzle_id,
                            puzzle_json=puzzle_json,
                            puzzle_output_folder=puzzle_output_folder,
                        )
                    )
                )
    except Exception:
        try:
            logger.exception(f"Error in puzzle worker for puzzle {puzzle_id} and UID {uid}")
            metadata.status = EvaluationStatus.ERROR
            metadata.end_time = datetime.now()
            metadata.duration_seconds = (metadata.end_time - metadata.start_time).total_seconds()
            async with lock:
                save_json(
                    metadata.to_dict(), 
                    puzzle_output_folder / "metadata.json"
                )
        except Exception:
            logger.exception(f"Error in saving puzzle metadata for puzzle {puzzle_id} and UID {uid} after puzzle worker ran into an error")
    except asyncio.CancelledError:
        try:
            metadata.status = EvaluationStatus.CANCELLED
            metadata.end_time = datetime.now()
            metadata.duration_seconds = (metadata.end_time - metadata.start_time).total_seconds()
            save_json(
                metadata.to_dict(), 
                puzzle_output_folder / "metadata.json"
            )
        except Exception:
            pass
        finally:
            raise
    else:
        try:
            metadata.status = EvaluationStatus.SUCCESS
            metadata.end_time = datetime.now()
            metadata.duration_seconds = (metadata.end_time - metadata.start_time).total_seconds()
            async with lock:
                save_json(
                    metadata.to_dict(), 
                    puzzle_output_folder / "metadata.json"
                )
        except Exception:
            logger.exception(f"Error in saving puzzle metadata for puzzle {puzzle_id} and UID {uid} after puzzle worker completed successfully")
    finally:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

async def evaluate(
    challenge_file: str,
    config_name: str,
    output_folder: str,  
    vllm_base_url: Optional[str],
    num_samples: int,
    parallel: int,
    submission_folder: str,
    timeout_hours: float,
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
        raise ValueError(f"Output folder {output_folder} or submission file {submission_file} does not exist. Not able to resume evaluation.")

    file_handler = None
    try:
        # Set up logging to file and console
        log_file = output_folder / "run.log"
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        logger.info(f"Created output directory: {output_folder}")
        logger.info(f"Created submission file: {submission_file}")

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
                metadata = PuzzleMetadata.from_dict(
                    json.loads(read_file(metadata_file))
                )
                if metadata.status == EvaluationStatus.SUCCESS:
                    done.append(metadata.puzzle_id)
                elif metadata.status == EvaluationStatus.RUNNING:
                    raise ValueError(f"Puzzle {metadata.puzzle_id} with UID {metadata.uid} is still running. Not able to resume evaluation.")
            challenge = {
                puzzle_id: puzzle_json 
                for puzzle_id, puzzle_json in challenge.items() 
                if puzzle_id not in done
            }

        logger.info(f"Loaded {len(challenge)} puzzles for evaluation")
        logger.info(f"Processing {len(challenge)} puzzles with {num_samples} majority voting samples and max concurrency of {parallel}")

        lock = asyncio.Lock()    # Used for shared access to the submission file and metadata. Enables incremental write to submissions.

        dashboard = StatusDashboard(
            output_folder=output_folder,
            total_puzzles=len(challenge),
            eval_start_time=datetime.now(),
            resume=resume,
            lock=lock,
            poll_interval=60,
            max_rows=5,
        )
        dashboard_task = asyncio.create_task(dashboard.run())

        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(parallel)

        tasks = [
            asyncio.create_task(
                puzzle_worker(
                    puzzle_id=puzzle_id,
                    puzzle_json=puzzle_json,
                    semaphore=semaphore,
                    lock=lock,
                    config=config,
                    evaluation_output_folder=output_folder,
                    submission_file=submission_file,
                    num_samples=num_samples,
                )
            ) for puzzle_id, puzzle_json in challenge.items()
        ]

        try:
            async with asyncio.timeout(timeout_hours * 3600):
                await asyncio.gather(*tasks)
            logger.info("Evaluation completed successfully")
            await dashboard.update()
        except Exception as e:
            dashboard.print_error(e)
            raise
        finally:
            try:
                async with asyncio.timeout(10 * 60):    # Give 10 minutes for tasks to finish gracefully
                    dashboard_task.cancel()
                    await asyncio.gather(dashboard_task, return_exceptions=True)
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.TimeoutError:
                logger.warning("Timeout while waiting for tasks to finish gracefully. Exiting evaluation without completing cleanup.")
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()

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
        "-n", "--num_samples",
        type=int,
        required=True,
        help="Number of samples for majority voting with early stopping e.g. for pass@2 evaluation of plain COT solver or for stabilizing score of interleaved thinking solver."
    )
    parser.add_argument(
        "-p", "--parallel",
        type=int,
        required=True,
        help="Number of tasks to process in parallel. Each task uses max. one sandbox and one model call at a time."
    )
    parser.add_argument(
        "-t", "--timeout_hours",
        type=float,
        required=True,
        help="Timeout in hours for the entire evaluation. Useful in Kaggle, where notebooks can run for max 12 hours and don't write output if they are timed out."
    )
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume evaluation from the last saved state. This works at the puzzle level, not at the sample level."
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
            num_samples=args.num_samples,
            parallel=args.parallel,
            submission_folder=args.submission_folder,
            timeout_hours=args.timeout_hours,
            resume=args.resume,
            env_file=args.env_file,
        )
    )

if __name__ == "__main__":
    main_cli()
