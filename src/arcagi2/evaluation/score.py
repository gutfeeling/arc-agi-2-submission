import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from arcagi2.evaluation.utils import EvaluationMetadata
from arcagi2.utils.utils import read_file


logger = logging.getLogger(__name__)

def score_puzzle_submission(
        solutions: list[list[list[int]]], 
        submission: list[dict], 
        ) -> list[list[int]]:
    """
    Score a puzzle submission against the expected outputs.
    
    Args:
        solutions: List of correct output grids, one per test.
        submission: List of dicts, one per test, each with "attempt_1" and "attempt_2" keys.
                   Format: [{"attempt_1": grid_or_none, "attempt_2": grid_or_none}, ...]
    
    Returns:
        List of lists, where each inner list contains scores [attempt_1_score, attempt_2_score]
        for the corresponding test index. Score is 1 if correct, 0 otherwise.
    """
    scores = []
    for test_idx, attempts in enumerate(submission):
        expected_output = solutions[test_idx]
        attempt_scores = [
            1 if attempts.get("attempt_1") == expected_output else 0,
            1 if attempts.get("attempt_2") == expected_output else 0,
        ]
        scores.append(attempt_scores)
    return scores

def compute_score(score_for_attempts: list[list[int]]) -> float:
    if len(score_for_attempts) == 0:
        return 0
    return sum(any(scores) for scores in score_for_attempts) / len(score_for_attempts)

def create_score_summary(solutions: dict[str, list[list[list[int]]]], scoring_results: list[dict]) -> None:
    score_df_data = [
        {
            "puzzle_id": item["puzzle_id"],
            "submission_id": item["submission_id"],
            "score_for_attempts": item["score_for_attempts"],
            "duration_minutes": item["duration_seconds"] / 60,
            "sample_index": item["sample_index"]
        } for item in scoring_results 
        if item["puzzle_id"] in solutions
    ]
    score_df = pd.DataFrame(score_df_data)
    # Add missing evaluation items
    existing_ids = set()
    if len(score_df) > 0:
        existing_ids = set(score_df["puzzle_id"].values)
    missing_ids = [pid for pid in solutions.keys() if pid not in existing_ids]
    if missing_ids:
        missing_df = pd.DataFrame({
            "puzzle_id": missing_ids,
            "submission_id": [None] * len(missing_ids),
            "score_for_attempts": [[]] * len(missing_ids),
            "duration_minutes": [0] * len(missing_ids),
            "sample_index": [None] * len(missing_ids)
        })
        score_df = pd.concat([score_df, missing_df], ignore_index=True)
    score_df["score"] = score_df["score_for_attempts"].apply(compute_score)
    score_df = score_df.sort_values(by=["score", "duration_minutes"], ascending=[False, False])
    logger.info(f"Score dataframe after computing score:\n{score_df.to_string()}")
    logger.info(f"Total score: {score_df['score'].sum()} / {len(score_df)}")
    score = score_df["score"].mean()
    logger.info(f"Score: {score}")

async def main(
    solutions_file: str,
    output_folder: str,
    submission_folder_relative: Optional[str],
):
    output_folder = Path(output_folder)
    if not output_folder.is_absolute():
        output_folder = Path.cwd() / output_folder

    logger.info(f"Loading solutions from {solutions_file}")
    solutions_file = Path(solutions_file)
    if not solutions_file.is_absolute():
        solutions_file = Path.cwd() / solutions_file
    solutions = json.loads(read_file(solutions_file))

    scoring_results = []
    for subfolder in output_folder.iterdir():
        if not subfolder.is_dir():
            continue

        puzzle_submission_file = subfolder / "submission.json"
        if not puzzle_submission_file.exists():
            continue
        puzzle_submission = json.loads(read_file(puzzle_submission_file))
        assert len(puzzle_submission) == 1, "Only one puzzle submission is supported"
        puzzle_id = list(puzzle_submission.keys())[0]
        if puzzle_id not in solutions:
            logger.warning(f"Puzzle {puzzle_id} not found in solutions file")
            continue
        score_for_attempts = score_puzzle_submission(solutions[puzzle_id], puzzle_submission[puzzle_id])

        sample_index = None  # Default value when submission_folder_relative is not provided
        if submission_folder_relative is not None:
            submission_metadata_file = subfolder / submission_folder_relative / "metadata.json"
            if submission_metadata_file.exists():
                submission_metadata = json.loads(read_file(submission_metadata_file))
                sample_index = submission_metadata["sample_index"]

        metadata_file = subfolder / "metadata.json"
        duration_seconds = 0
        if metadata_file.exists():
            metadata = EvaluationMetadata.from_dict(json.loads(read_file(metadata_file)))
            if metadata.duration_seconds is not None:
                duration_seconds = metadata.duration_seconds
        scoring_results.append(
            {
                "puzzle_id": puzzle_id, 
                "submission_id": subfolder.name, 
                "duration_seconds": duration_seconds,
                "sample_index": sample_index,
                "score_for_attempts": score_for_attempts,
            }
        )

    create_score_summary(solutions, scoring_results)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Score grid descriptions in submission files using a judge model"
    )
    parser.add_argument(
        "--solutions_file",
        type=str,
        required=True,
        help="Path to the solutions JSON file in ARC Prize format",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Path to the output folder created by the evaluation script",
    )
    parser.add_argument(
        "-r",
        "--submission_folder_relative",
        type=str,
        help="Relative path of the folder (with respect to the solver's output folder root) containing sample index information. 'submission/extended' or 'submission/core'. Not required for plain COT solvers.",
    )
    args = parser.parse_args()

    return args

def main_cli():
    args = parse_arguments()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    asyncio.run(
        main(
            solutions_file=args.solutions_file,
            output_folder=args.output_folder,
            submission_folder_relative=args.submission_folder_relative,
        )
    )

if __name__ == "__main__":
    main_cli()
