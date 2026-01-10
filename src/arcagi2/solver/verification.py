from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Union

from arcagi2.solver.config.base import SolverConfig
from arcagi2.utils.solving_utils import solution_works_on_training_examples, get_coverage_report
from arcagi2.utils.utils import save_text, save_json

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    score_on_training_examples: list[int]
    diff_str: Union[str, None]
    diff_info: Union[dict, None]
    train_coverage_str: Union[str, None]
    test_coverage_str: Union[str, None]
    train_coverage: Union[dict, None]
    test_coverage: Union[dict, None]

    @property
    def all_training_examples_passed(self) -> bool:
        return all(score == 1 for score in self.score_on_training_examples)

    def save_to_dir(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        info_file_path = dir / "verification.json"
        logger.info(f"Saving info to {info_file_path}")
        info = {
            "score_on_training_examples": self.score_on_training_examples,
            "diff_info": self.diff_info,
            "train_coverage": self.train_coverage,
            "test_coverage": self.test_coverage,
        }
        save_json(info, info_file_path)
        try:
            if self.diff_str is not None:
                diff_str_file_path = dir / "diff.txt"
                logger.info(f"Saving diff string to {diff_str_file_path}")
                save_text(self.diff_str, diff_str_file_path)
            if self.train_coverage_str is not None:
                train_coverage_str_file_path = dir / "train_coverage.txt"
                logger.info(f"Saving train coverage string to {train_coverage_str_file_path}")
                save_text(self.train_coverage_str, train_coverage_str_file_path)
            if self.test_coverage_str is not None:
                test_coverage_str_file_path = dir / "test_coverage.txt"
                logger.info(f"Saving test coverage string to {test_coverage_str_file_path}")
                save_text(self.test_coverage_str, test_coverage_str_file_path)
        except Exception as e:
            logger.exception(f"Error saving additional verification information: {e}")

async def verify_solution(
        config: SolverConfig,
        puzzle_json,
        solution,
        save_to_dir: Union[Path, None]=None,
        ) -> VerificationResult:
        
    # check if solution works on all training examples
    score_on_training_examples = [0 for _ in range(len(puzzle_json["train"]))]
    diff_str = None
    diff_info = None
    train_coverage_str = None
    test_coverage_str = None
    train_coverage = None
    test_coverage = None

    try:
        diff_str, diff_info, score_on_training_examples = await solution_works_on_training_examples(
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
    except Exception:
        logger.exception(f"Error checking if solution works on training examples")

    try:
        train_coverage_str, test_coverage_str, train_coverage, test_coverage = await get_coverage_report(
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
    except Exception:
        logger.exception(f"Error getting coverage report")

    result = VerificationResult(
        score_on_training_examples=score_on_training_examples,
        diff_str=diff_str,
        diff_info=diff_info,
        train_coverage_str=train_coverage_str,
        test_coverage_str=test_coverage_str,
        train_coverage=train_coverage,
        test_coverage=test_coverage,
    )
    if save_to_dir is not None:
        result.save_to_dir(save_to_dir)
    return result
