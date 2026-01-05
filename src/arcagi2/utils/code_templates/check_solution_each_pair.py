import copy
import numbers

import numpy as np


def clean_grid(grid):
    """Sometimes the grids are numpy arrays, or list of lists containing numpy datatypes."""
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()

    output = []
    for row_idx, row in enumerate(grid):
        new_row = []
        for col_idx, x in enumerate(row):
            if not isinstance(x, numbers.Integral):
                raise TypeError(
                    f"Non-integer value {x!r} at position ({row_idx}, {col_idx})"
                )
            new_row.append(int(x))
        output.append(new_row)
    return output


def grid_to_text(grid):
    return "\n".join(" ".join(map(str, row)) for row in grid)


per_test_pair_scores = []
for test_num, test in enumerate(tests):
    grid = clean_grid(test["input"])
    print(f"Test {test_num + 1} input grid:\n{grid_to_text(grid)}")
    print(f"Test {test_num + 1} expected output grid:\n{grid_to_text(test['output'])}")
    try:
        output_grid = solution(copy.deepcopy(grid))
    except Exception as e:
        print(f"Test{test_num + 1} error: {e}")
        per_test_pair_scores.append(0)
        continue
    output_grid = clean_grid(output_grid)
    print(f"Test{test_num + 1} actual output grid:\n{grid_to_text(output_grid)}")
    if output_grid != test["output"]:
        print(f"Test{test_num + 1} expected and actual output grids are different")
        per_test_pair_scores.append(0)
    else:
        per_test_pair_scores.append(1)
        print(f"Test{test_num + 1} expected and actual output grids match")
print(' '.join(str(score) for score in per_test_pair_scores))
