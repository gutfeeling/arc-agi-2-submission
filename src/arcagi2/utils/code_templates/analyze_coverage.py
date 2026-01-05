import importlib
import linecache
import traceback

import coverage

import solution_code_{{random_id}}


filename = solution_code_{{random_id}}.__file__

def analyze_coverage(input_grid):
    error = None

    cov = coverage.Coverage(data_file=None, include=[filename])
    cov.start()

    # Import-time code (helpers, nested defs, docstrings) runs under coverage
    importlib.reload(solution_code_{{random_id}})
    try:
        solution_code_{{random_id}}.solution(input_grid)
    except Exception as e:
        error = traceback.format_exc()

    cov.stop()

    # filename, statements, excluded, missing, missing_str
    fname, statements, excluded, missing, missing_str = cov.analysis2(filename)

    statements = set(statements)      # all executable lines in solution.py
    missing    = set(missing)         # subset of statements
    executed   = statements - missing # lines that actually ran in this run

    total_exec = len(statements)
    coverage_pct = 100.0 * len(executed) / total_exec if total_exec else 0.0

    return {
        "statements": sorted(statements),
        "executed_lines": sorted(executed),
        "coverage_pct": coverage_pct,
        "missing_lines": sorted(missing),
        "error": error,
    }

def generate_coverage_report(puzzle, data_type="train"):
    reports = dict()
    all_statements = set()
    all_executed = set()

    for idx, pair in enumerate(puzzle[data_type]):
        report = analyze_coverage(pair["input"])
        reports[idx] = report
        print(f"=== {data_type.capitalize()} example {idx} ===")
        if report["error"] is not None:
            print(f"Running solution on this input raised the following error:\n{report['error']}\n")
        print(f"Executable lines in code block: {len(report['statements'])}")
        print(f"Executed lines:                 {len(report['executed_lines'])}")
        print(f"Coverage:                       {report['coverage_pct']:.1f}%")
        print("Lines that did not run for this input:")
        if len(report['missing_lines']) == 0:
            print("  <none>")
        else:
            for ln in report['missing_lines']:
                print(f"  {ln}: {linecache.getline(filename, ln).rstrip()}")
        print()
        all_statements.update(report["statements"])
        all_executed.update(report["executed_lines"])

    never_executed = sorted(all_statements - all_executed)

    # Compute dead_code
    if len(never_executed) > 0:
        lines = []
        for ln in never_executed:
            lines.append(f"{ln}: {linecache.getline(filename, ln).rstrip()}")
        dead_code = "\n".join(lines)
    else:
        dead_code = None
    reports["dead_code"] = dead_code
    
    # Print heading and dead code
    print(f"=== Lines in the solution code block never executed by any {data_type} example ===")
    if dead_code is None:
        print("  <none>")
    else:
        print(dead_code)
    
    return reports