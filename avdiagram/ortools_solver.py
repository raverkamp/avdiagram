import pprint as pp
import ortools
from ortools.linear_solver import pywraplp


def solve(variables, constraints, objective, min_or_max):
    assert min_or_max in ("min", "max")
    """print("variables")
    pp.pprint(variables)
    print("constraints")
    pp.pprint(constraints)
    print("objective")
    pp.pprint(objective)
    pp.pprint(min_or_max)"""

    solver = pywraplp.Solver.CreateSolver("GLOP")

    var_dict = dict()

    for var in variables:
        (name, lbound, ubound) = var
        if name in var_dict:
            raise Exception("duplicate variable: " + name)
        assert isinstance(name, str)
        if ubound is None:
            up = solver.infinity()
        else:
            up = ubound

        if lbound is None:
            low = -solver.infinity()
        else:
            low = lbound

        v = solver.NumVar(low, up, name)
        var_dict[name] = v

    cons_dict = {}
    for cons in constraints:
        (name, coefficients, lbound, ubound) = cons
        assert isinstance(name, str)
        if name in cons_dict:
            raise Exception("duplicate constraint name: " + name)
        c = solver.Constraint(name)
        if not lbound is None:
            c.SetLb(lbound)
        if not ubound is None:
            c.SetUb(ubound)

        cons_vars = set()
        for coeff in coefficients:
            (fac, vname) = coeff
            assert isinstance(vname, str)
            assert vname in var_dict
            var = var_dict[vname]
            c.SetCoefficient(var, fac)
            cons_vars.add(vname)

    obj = solver.Objective()
    obj_vars = set()
    for coeff in objective:
        (fac, vname) = coeff
        if vname in obj_vars:
            raise Exception("duplicate variable in objective:" + vname)
        assert vname in var_dict, f"{0} is nit a variable"
        obj.SetCoefficient(var_dict[vname], fac)
    if min_or_max == "min":
        obj.SetMinimization()
    else:
        obj.SetMaximization()

    # print(solver.ExportModelAsLpFormat(False))

    status = solver.Solve()

    if status != solver.OPTIMAL:
        print("The problem does not have an optimal solution!")
        if status == solver.FEASIBLE:
            print("A potentially suboptimal solution was found.")
        else:
            print("The solver could not solve the problem.")
            exit(1)
    res = {}
    for vname in var_dict:
        res[vname] = var_dict[vname].solution_value()
    return res
