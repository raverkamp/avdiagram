"""simple ortools wrapper tests"""
import unittest
from avdiagram import ortools_solver


class Test_simple(unittest.TestCase):
    def test_simple(self):
        vars = [("x", None, None), ("y", None, None)]
        cons = [
            ("c1", [(1, "x"), (2, "y")], None, 14),
            ("c2", [(3, "x"), (-1, "y")], 0, None),
            ("c3", [(1, "x"), (-1, "y")], None, 2),
        ]
        obj = [(3, "x"), (4, "y")]

        res = ortools_solver.solve(vars, cons, obj, "max")
        self.assertAlmostEqual(6, res["x"])
        self.assertAlmostEqual(4, res["y"])


if __name__ == "__main__":
    unittest.main()
