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


results = []
diff_info = dict()
for idx, ex in enumerate(puzzle["train"]):
    grid = ex["input"]
    expected = ex["output"]
    try:
        predicted = solution(copy.deepcopy(grid))
    except Exception as e:
        print(f"Train example {idx}: ERROR\n{e}\n")
        results.append(0)
        continue
    predicted = clean_grid(predicted)
    if predicted != expected:
        print(f"Train example {idx}: mismatch")
        print(f"Expected grid:\n{grid_to_text(expected)}")
        print(f"Predicted grid:\n{grid_to_text(predicted)}")
        results.append(0)
    else:
        print(f"Train example {idx}: OK")
        results.append(1)