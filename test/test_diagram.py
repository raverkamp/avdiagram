"""simple diagram tests"""
import os
import tempfile
import unittest
from avdiagram import diagram, mm
from tempfile import NamedTemporaryFile


class Test_simple(unittest.TestCase):

    def test_success(self):
        d = diagram.Diagram(mm(200), mm(200))
        t1 = d.text("abc", 12, "t1")
        t2 = d.text("xyz", 12, "t2")
        d.over(t1, mm(2), t2)
        d.show()

    def test_fail(self):
        d = diagram.Diagram(mm(200), mm(200))
        t1 = d.text("abc", 12)
        t2 = d.text("xyz", 12)
        d.over(t1, mm(2), t2)
        d.over(t2, mm(2), t1)
        with self.assertRaises(Exception):
            res = d.solve_problem(False)



if __name__ == "__main__":
    unittest.main()
