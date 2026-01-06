import argparse
import asyncio
from dataclasses import fields
from datetime import datetime
import json
import logging
from pathlib import Path
import time
import uuid

from dotenv import load_dotenv

import os
# USE_DAYTONA=True: Daytona cloud sandboxes, False (default): ipybox Docker
if os.getenv("USE_DAYTONA", "False") == "True":
    from arcagi2.tools.execution_client_daytona import ExecutionError
else:
    from ipybox import ExecutionError

from arcagi2.api.clients import AbstractAPIClient
from arcagi2.solver.config import SYSTEM_CONFIG
from arcagi2.solver.solver import solver
from arcagi2.utils.logging_utils import (
    setup_logger_for_parallel_processing_script,
    ProgressCounter,
)
from arcagi2.utils.utils import read_file, save_json


logger = logging.getLogger(__name__)


async def main(
    evaluation_data_file,
    config_name,
    output_folder,  
    vllm_base_url,
    max_parallel_requests,
    env_file,
    timeout_minutes,
    resume,
):
    logger.info(f"Loading environment variables from {env_file}")
    load_dotenv(env_file)

    # Load config from SOLVE_CONFIG
    if config_name not in SYSTEM_CONFIG:
        raise ValueError(f"Config '{config_name}' not found in SYSTEM_CONFIG. Available configs: {list(SYSTEM_CONFIG.keys())}")
    
    config = SYSTEM_CONFIG[config_name]
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
    if not resume:
        output_folder.mkdir(parents=True, exist_ok=False)
    elif not output_folder.exists():
        raise ValueError(f"Output folder {output_folder} does not exist. Cannot resume from where we left off.")

    log_file = output_folder / "run.log"
    setup_logger_for_parallel_processing_script(logger, log_file)
    logger.info(f"Created output directory: {output_folder}")
    logger.info(f"Logging to {log_file}")

    logger.info(f"Loading evaluation data from {evaluation_data_file}")
    evaluation_data_file = Path(evaluation_data_file)
    if not evaluation_data_file.is_absolute():
        evaluation_data_file = Path.cwd() / evaluation_data_file

    evaluation_data = json.loads(read_file(evaluation_data_file))
    if resume:
        done = []
        for subfolder in output_folder.iterdir():
            if not subfolder.is_dir():
                continue
            metadata_file = subfolder / "metadata.json"
            if not metadata_file.exists():
                continue
            metadata = json.loads(read_file(metadata_file))
            done.append(metadata["id"])
        evaluation_data = [item for item in evaluation_data if item["metadata"]["id"] not in done]
    logger.info(f"Loaded {len(evaluation_data)} items for evaluation")

    total_items = len(evaluation_data)

    if max_parallel_requests == 1:
        logger.info(
            f"Processing {total_items} items sequentially"
        )
    else:
        logger.info(
            f"Processing {total_items} items with max concurrency of {max_parallel_requests}"
        )

    # Create semaphore to control concurrency
    semaphore = asyncio.Semaphore(max_parallel_requests)
    progress_counter = ProgressCounter(total_items, logger)

    async def worker(item):
        async with semaphore:
            # Create a folder with random ID
            submission_id = uuid.uuid4().hex[:9]
            item_folder = output_folder / submission_id
            item_folder.mkdir(parents=True, exist_ok=False)
            save_json(
                item["metadata"],
                item_folder / "metadata.temp.json"
            )
            start_time = time.time()
            
            try:
                await asyncio.wait_for(
                    solver(
                        config=config,
                        puzzle_json=item["puzzle"],
                        output_folder=item_folder,
                    ),
                    timeout=timeout_minutes * 60
                )
            except asyncio.TimeoutError:
                duration_seconds = time.time() - start_time
                logger.warning(f"Submission {submission_id} timed out after {timeout_minutes} minutes")
            except (Exception, ExecutionError) as e:
                logger.exception(f"Submission {submission_id} failed due to exception: {e}")
                return
            else:
                duration_seconds = time.time() - start_time
                logger.info(f"Submission {submission_id} for eval item {item['metadata']['id']} completed in {duration_seconds / 60} minutes")

            metadata = item["metadata"].copy()
            metadata["duration_seconds"] = duration_seconds
            
            # Save metadata file with timing info
            save_json(metadata, item_folder / "metadata.json")
            (item_folder / "metadata.temp.json").unlink()
            await progress_counter.increment()

    tasks = [
        asyncio.create_task(worker(item)) for item in evaluation_data
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
            f"Processing complete! {total_items} items processed"
        )
        logger.info("Evaluation completed successfully")
    finally:
        # Cleanup: cancel any remaining tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate models on solving ARC AGI puzzles"
    )

    parser.add_argument(
        "-d",
        "--evaluation_data_file",
        type=str,
        required=True,
        help="Path to the JSON file containing evaluation data with grid and metadata.",
    )

    parser.add_argument(
        "-c", "--config_name",
        type=str,
        default="default",
        help="Configuration key. The configuration dict is defined in arcagi2.solver.config.SOLVE_CONFIG."
    )

    parser.add_argument(
        "-o", "--output_folder",
        type=str,
        help="Folder to save the output"
    )
    
    parser.add_argument(
        "-b", "--vllm_base_url",
        type=str,
        help="Base URL of VLLM server"
    )

    parser.add_argument(
        "-p", "--max-parallel-requests",
        type=int,
        default=5,
        help="Maximum number of parallel requests"
    )

    parser.add_argument(
        "-e", "--env_file",
        type=str,
        default=".env",
        help="Path to environment file containing API keys and other configuration. Useful when running multiple instances of the script in parallel with different configurations (e.g. with different OpenAI accounts). Defaults to '.env'."
    )

    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=720,
        help="Timeout in minutes for each puzzle"
    )

    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="Resume from where we left off. If a submission doesn't have metadata, then it means that it didn't finish, so we should retry it."
    )

    args = parser.parse_args()

    if args.resume and args.output_folder is None:
        raise ValueError("Cannot resume from where we left off if output folder is not specified.")

    if args.output_folder is None:
        args.output_folder = (
            Path(args.evaluation_data_file).parent
            / f"evaluation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    return args

def main_cli():
    args = parse_arguments()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Silence noisy HTTP client logs (polling creates a lot of noise)
    # We save all logs to files, so we would need a lot of space. This is especially bad in Kaggle I guess.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    asyncio.run(
        main(
            evaluation_data_file=args.evaluation_data_file,
            config_name=args.config_name,
            output_folder=args.output_folder,
            vllm_base_url=args.vllm_base_url,
            max_parallel_requests=args.max_parallel_requests,
            env_file=args.env_file,
            timeout_minutes=args.timeout,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main_cli()
