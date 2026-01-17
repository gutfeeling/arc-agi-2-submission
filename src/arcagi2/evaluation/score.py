import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from arcagi2.evaluation.utils import SampleMetadata, SampleStatus
from arcagi2.utils.utils import read_file


logger = logging.getLogger(__name__)

def score_puzzle_submission(
        solutions: list[list[list[int]]], 
        submission: list[dict], 
        num_attempts: int,
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
        try:
            expected_output = solutions[test_idx]
        except IndexError:
            logger.exception(f"IndexError: {test_idx} not found in solutions")
            continue
        attempt_scores = [
            1 if attempts.get(f"attempt_{num + 1}") == expected_output else 0
            for num in range(num_attempts)
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
            "score_for_attempts": item["score_for_attempts"],
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
            "score_for_attempts": [[]] * len(missing_ids),
        })
        score_df = pd.concat([score_df, missing_df], ignore_index=True)
    score_df["score"] = score_df["score_for_attempts"].apply(compute_score)
    score_df = score_df.sort_values(by=["score"], ascending=[False])
    logger.info(f"Score dataframe after computing score:\n{score_df.to_string()}")
    logger.info(f"Total score: {score_df['score'].sum()} / {len(score_df)}")
    score = score_df["score"].mean()
    logger.info(f"Score: {score}")

async def score_submission(
    solutions_file: str,
    submissions_file: str,
    num_attempts: int,
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Loading solutions from {solutions_file}")
    solutions_file = Path(solutions_file)
    if not solutions_file.is_absolute():
        solutions_file = Path.cwd() / solutions_file
    solutions = json.loads(read_file(solutions_file))

    logger.info(f"Loading submissions from {submissions_file}")
    submissions_file = Path(submissions_file)
    if not submissions_file.is_absolute():
        submissions_file = Path.cwd() / submissions_file
    submissions = json.loads(read_file(submissions_file))

    scoring_results = []
    for puzzle_id, submission_data in submissions.items():
        if puzzle_id not in solutions:
            logger.warning(f"Puzzle {puzzle_id} not found in solutions")
            continue
        score_for_attempts = score_puzzle_submission(
            solutions=solutions[puzzle_id], 
            submission=submission_data,
            num_attempts=num_attempts
        )

        scoring_results.append(
            {
                "puzzle_id": puzzle_id, 
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
        "--submissions_file",
        type=str,
        required=True,
        help="Path to the submissions JSON file",
    )
    parser.add_argument(
        "-n",
        "--num_attempts",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of attempts to score. 1 for single attempt, 2 for two attempts.",
    )
    args = parser.parse_args()

    return args

def main_cli():
    args = parse_arguments()

    asyncio.run(
        score_submission(
            solutions_file=args.solutions_file,
            submissions_file=args.submissions_file,
            num_attempts=args.num_attempts,
        )
    )

if __name__ == "__main__":
    main_cli()
