import json
import logging
import random
from pathlib import Path
from typing import Union

from arcagi2.utils.utils import read_file


logger = logging.getLogger(__name__)

def get_first_working_solution(output_folder: Path) -> Union[Path, None]:
    """
    Get the first sample that passes all training examples.
    Used for the minimal core system.
    """
    output_folder = Path(output_folder)
    
    solution_paths = sorted(
        output_folder.rglob("solution.py"),
        key=lambda x: str(x)
    )

    for solution_path in solution_paths:
        solution_dir = solution_path.parent
        verification_dir = solution_dir / "verification"
        verification_json_path = verification_dir / "verification.json"
        
        if not verification_json_path.exists():
            continue
        
        verification_result = json.loads(read_file(verification_json_path))
        
        score_on_training_examples = verification_result.get("score_on_training_examples", [])
        training_failures = sum(1 - score for score in score_on_training_examples)
        if training_failures == 0:
            return solution_path
    
    return None

def choose_best_solution(output_folder: Path) -> Union[Path, None]:
    """
    Choose the best sample from the output folder based on several metrics.
    Used for the leaderboard submission.
    The main metrics are:
    - Hard verification: do all training examples pass?
    - Soft verification based on LLM as a judge
    - Smaller token count as a tie breaker if there is a tie after hard and soft verification.
    - Optional metrics to tie break when training examples don't pass:
        - Shape mismatches: how many training examples have different shapes?
        - Total diff count: how many differences are there in the training examples?
        - Total errored: how many test examples have errors
    """
    output_folder = Path(output_folder)
    
    # Find all solution.py files
    solution_paths = list(output_folder.rglob("solution.py"))
    
    if not solution_paths:
        return None
    
    # Collect metrics for each solution
    candidates = []
    
    for solution_path in solution_paths:
        # Get the verification folder adjacent to the solution
        solution_dir = solution_path.parent
        verification_dir = solution_dir / "verification"
        verification_json_path = verification_dir / "verification.json"
        
        if not verification_json_path.exists():
            continue
        
        verification_result = json.loads(read_file(verification_json_path))
        
        # Extract metrics
        
        # 1. Number of training examples that don't pass (lower is better)
        score_on_training_examples = verification_result.get("score_on_training_examples", [])
        training_failures = sum(1 - score for score in score_on_training_examples)

        # Optional metrics to tie break when training examples don't pass
        shape_mismatches = float("inf")
        total_diff_count = float("inf")
        total_errored = float("inf")
        
        try:
            # 2. Number of training failures where expected shape doesn't match actual shape (lower is better)
            diff_info = verification_result.get("diff_info")
            if diff_info is not None:
                shape_mismatches = sum(
                    1 for example_info in diff_info.values()
                    if example_info.get("pred_shape") != example_info.get("exp_shape")
                )
            else:
                shape_mismatches = float("inf")  # Penalize solutions without diff info
            
            # 3. Total diff count for training examples that don't pass (lower is better)
            if diff_info is not None:
                total_diff_count = sum(
                    example_info.get("diff_count", 0) 
                    for example_info in diff_info.values()
                )
            else:
                total_diff_count = float("inf")  # Penalize solutions without diff info
            
            # 4. Number of test examples with errors (lower is better)
            test_coverage = verification_result.get("test_coverage")
            if test_coverage is not None:
                total_errored = sum(
                    1 for value in test_coverage.values() 
                    if isinstance(value, dict) and value.get("error") is not None
                )
            else:
                total_errored = float("inf")  # Penalize solutions without test coverage info
        except Exception as e:
            logger.exception(f"Error calculating optional metrics for verification file {verification_json_path}")

        # Soft verifier related metrics

        flag_count = float("inf")
        special_casing_count = float("inf")

        soft_verification_file_path = verification_dir / "soft_verification" / "soft_verification.json"
        if soft_verification_file_path.exists():
            soft_verification_result = json.loads(read_file(soft_verification_file_path))
            # 5. Number of soft verifier flags with decision: True (lower is better)
            flag_count = sum(1 for flag in soft_verification_result if flag["decision"])

            # 6. Tie break equal flags by special_casing since about ~10% scoring solutions get flagged for color_arithmetic
            special_casing_count = sum(1 for flag in soft_verification_result if flag["property"] == "special_casing" and flag["decision"])

        # Token count related metrics

        total_tokens = float("inf")

        token_consumption_file_path = solution_dir / "token_consumption.json"
        if token_consumption_file_path.exists():
            token_consumption = json.loads(read_file(token_consumption_file_path))
            total_tokens = token_consumption["total"]
        
        # Store the solution with its metrics
        candidates.append({
            "path": solution_path,
            "metrics": (
                training_failures, 
                flag_count, 
                special_casing_count, 
                # Optional metrics to tie break when training examples don't pass
                shape_mismatches,    # This will be zero if all training examples pass
                total_diff_count,    # This will be zero if all training examples pass
                # This will be the discrimator if there is a tie after hard and soft verification.
                # Overthinking is generally bad, so we prefer solutions with fewer tokens.
                total_tokens,
                total_errored,    # Total errored is the last because we haven't done generalization yet.
            )
        })
    
    if not candidates:
        return None
    
    # Sort by metrics (lower is better for all)
    candidates.sort(key=lambda x: x["metrics"])
    logger.info(f"Sorted candidates: {candidates}")
    
    # Get all solutions with the best metrics
    best_metrics = candidates[0]["metrics"]
    best_solutions = [c for c in candidates if c["metrics"] == best_metrics]
    
    # If there's still a tie, randomly select one
    chosen = random.choice(best_solutions)
    
    return chosen["path"]

def choose_best_output_grids(puzzle_json: dict, output_folder: Path) -> list[dict]:
    """
    Choose the best output grids from the output folder based on majority voting.
    Used for the plain COT baseline.
    
    Returns a list in Kaggle submission format:
    [{"attempt_1": grid, "attempt_2": grid}, ...] where each element corresponds to a test case.
    """
    
    output_folder = Path(output_folder)
    solutions = {
        test_index: [] for test_index in range(len(puzzle_json["test"]))
    }
    
    solution_paths = sorted(
        output_folder.rglob("solution.json"),
        key=lambda x: str(x)
    )

    for solution_path in solution_paths:
        solution_dir = solution_path.parent
        test_index = int(solution_dir.name)
        output_grid = json.loads(read_file(solution_path))
        found = False
        for i in range(len(solutions[test_index])):
            if solutions[test_index][i][0] == output_grid:
                solutions[test_index][i][1] += 1
                found = True
                break
        if not found:
            solutions[test_index].append([output_grid, 1])


    # Sort by majority vote, grids that appear more often are better
    sorted_solutions = {
        test_index: sorted(solutions[test_index], key=lambda x: x[1], reverse=True)
        for test_index in solutions.keys()
    }
    
    # Build submission list in Kaggle format
    submission = [
        {
            "attempt_1": sorted_solutions[test_index][0][0] if len(sorted_solutions[test_index]) > 0 else None,
            "attempt_2": sorted_solutions[test_index][1][0] if len(sorted_solutions[test_index]) > 1 else None,
        }
        for test_index in range(len(puzzle_json["test"]))
    ]
    return submission
