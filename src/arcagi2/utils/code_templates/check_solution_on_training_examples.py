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


def shape_of(grid):
    return (len(grid), len(grid[0]))


def diff_info_equal_shape(pred, exp, max_coords=10):
    """
    Assumes pred and exp have the same shape.

    Returns (diff_count, first_coords_rc, bbox_rc) where:
      - diff_count: number of differing cells
      - first_coords_rc: up to `max_coords` mismatch coordinates,
        each as (row, col), 0-indexed, top-left origin.
      - bbox_rc: [min_row, min_col, max_row, max_col] covering all mismatches,
        using the same (row, col) convention; None if there are no mismatches.
    """
    P = np.array(pred, dtype=int)
    E = np.array(exp, dtype=int)
    M = (P != E)
    ys, xs = np.where(M)  # ys=rows, xs=cols
    coords_rc = sorted((int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist()))
    diff_count = len(coords_rc)
    if diff_count > 0:
        rows = [r for r, _ in coords_rc]
        cols = [c for _, c in coords_rc]
        bbox_rc = [min(rows), min(cols), max(rows), max(cols)]
    else:
        bbox_rc = None
    return diff_count, coords_rc[:max_coords], bbox_rc

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
        pred_shape = shape_of(predicted)
        exp_shape = shape_of(expected)
        diff_info[idx] = {
            "pred_shape": pred_shape,
            "exp_shape": exp_shape,
        }
        print(f"Train example {idx}: mismatch")
        print(f"Predicted shape (rows, cols): {pred_shape}")
        print(f"Expected shape  (rows, cols): {exp_shape}")

        if pred_shape == exp_shape:
            diff_count, first_coords_rc, bbox_rc = diff_info_equal_shape(
                predicted, expected, max_coords=10
            )
            diff_info[idx]["diff_count"] = diff_count
            print(f"Diff count: {diff_count}")
            # Coordinates reported as (row, col), 0-indexed, top-left origin
            print(f"First 10 coordinates (row, col): {first_coords_rc}")
            print(f"Diff bbox [min_row, min_col, max_row, max_col]: {bbox_rc}")
        print()
        results.append(0)
    else:
        print(f"Train example {idx}: OK")
        results.append(1)