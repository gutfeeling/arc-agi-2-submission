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

out = []
for idx, ex in enumerate(puzzle["test"]):
    grid = ex["input"]
    try:
        predicted = solution(copy.deepcopy(grid))
    except Exception as e:
        print(f"Test example {idx}: ERROR\n{e}\n")
        out.append(None)
        continue
    else:
        out.append(clean_grid(predicted))