import subprocess
import tempfile
from io import StringIO
import pprint as pp
from typing import List, Optional, Union, Any, Tuple, TextIO, cast

from . import solver

# Wrapper for GPLK as subprocess
# we use the simple input format
# Variables are defined
# the number of constraints


def generate_problem(variables, constraints, objective, min_or_max, verbose):
    assert min_or_max in ("min", "max")
    if verbose:
        print("variables")
        pp.pprint(variables)
        print("constraints")
        pp.pprint(constraints)
        print("objective")
        pp.pprint(objective)
        pp.pprint(min_or_max)

    var_dict = dict()
    id = 0
    for var in variables:
        id = id + 1
        (name, lbound, ubound) = var
        if name in var_dict:
            raise Exception("duplicate variable: " + name)
        assert isinstance(name, str)
        var_dict[name] = id

    n = 0
    for c in constraints:
        n = n + len(c[1])

    out = StringIO()
    if min_or_max is None:
        raise "no objective"
    out.write("c Problem\n")
    if min_or_max == "max":
        mm = "max"
    else:
        mm = "min"
    out.write("p lp {3} {0} {1} {2}\n".format(len(constraints), len(variables), n, mm))

    i = 1
    for cons in constraints:
        (name, coefficients, lbound, ubound) = cons
        out.write("n i {0} cons_{1}\n".format(i, i))
        if not lbound is None and not ubound is None:
            if lbound == ubound:
                out.write("i {0} s {1}\n".format(i, lbound))
            else:
                out.write("i {0} d {1} {2}\n".format(i, lbound, ubound))
        elif not lbound is None:
            out.write("i {0} l {1}\n".format(i, lbound))
        elif not ubound is None:
            out.write("i {0} u {1}\n".format(i, ubound))
        else:
            raise Exception("BUG")
        i = i + 1
    for var in variables:
        (name, lbound, ubound) = var
        id = var_dict[name]

        out.write("n j {0} {1}\n".format(id, name))
        if lbound is None and ubound is None:
            out.write("j {0}  f\n".format(id))
        elif not lbound is None and ubound is None:
            out.write("j {0}  l {1} \n".format(id, lbound))
        elif lbound is None and not ubound is None:
            out.write("j {0}  u {1} \n".format(id, ubound))
        elif not lbound is None and not ubound is None and lbound < ubound:
            out.write("j {0} c d {1} {2} \n".format(id, lbound, ubound))
        elif not lbound is None and not ubound is None and lbound >= ubound:
            out.write("j {0} s {1}\n".format(id, lbound))
        else:
            raise Exception("BUG")

    for (coeff, v) in objective:
        id = var_dict[v]
        out.write("a 0 {0} {1}\n".format(id, coeff))
    j = 1
    for cons in constraints:
        (name, coefficients, lbound, ubound) = cons
        for (coeff, v) in coefficients:
            id = var_dict[v]
            out.write("a {0} {1} {2}\n".format(j, id, coeff))
        j = j + 1
    out.write("e o f\n")
    res = (var_dict, out.getvalue())
    out.close()
    return res


def parse_solution(fileName: str) -> List[float]:
    with open(fileName, "r") as input:
        objective: float = 0
        vars: List[float] = []
        for line in input:
            if len(line) == 0:
                continue
            cc = line[0]
            if cc == "c":
                continue
            if cc == "e":
                break
            if cc == "s":  # solution
                (_, _, ncons, nvars, uf1, uf2, obj) = line.split(" ")
                # f == feasible, u == unfeasible?
                if not (uf1 == "f" and uf2 == "f"):
                    return []
                objective = float(obj)
                vars = [0.0] * int(nvars)
            if cc == "j":
                cols = line.split(" ")
                j = int(cols[1])
                v = float(cols[3])
                vars[j - 1] = v
        return vars


def solve_problem(variables, constraints, objective, min_or_max, verbose=False):
    (var_dict, problem) = generate_problem(
        variables, constraints, objective, min_or_max, verbose
    )
    with tempfile.NamedTemporaryFile(
        prefix="glpk-input-", suffix=".txt", mode="w", delete=False
    ) as file_input:
        file_input_name = file_input.name
        file_input.write(problem)
    with tempfile.NamedTemporaryFile(
        prefix="glpk-output-", suffix=".txt", mode="w", delete=False
    ) as file_output:
        file_output_name = file_output.name
    subprocess.call(
        ["glpsol", "--glp", file_input_name, "--write", file_output_name],
        stdin=None,
        stdout=None,
        stderr=None,
        shell=False,
        timeout=None,
    )

    solution_vars = parse_solution(file_output_name)
    if len(solution_vars) == 0:
        return False
    res = {}
    for var_name in var_dict:
        index = var_dict[var_name]
        res[var_name] = solution_vars[index - 1]
    print(res)
    return res


class GLPKSolver(solver.Solver):
    def __init__(self):
        pass

    def solve_problem(
        self, variables, constraints, objective, min_or_max, verbose=False
    ):
        return solve_problem(variables, constraints, objective, min_or_max, verbose)
