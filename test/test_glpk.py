"""simple glpk wrapper tests"""
import unittest
from avdiagram import glpk

class Test_simple(unittest.TestCase):

    def test_success(self):
        p = glpk.Problem()
        v = p.addvar("a",0, 2)
        p.addConstraint([(v,1)],"GE",1)
        p.setObjective("MIN",[(v,1)])
        self.assertTrue(p.run())
        self.assertEqual(v.value,1)

    def test_fail(self):
        p = glpk.Problem()
        v = p.addvar("a",-1, 0)
        p.addConstraint([(v,1)],"GE",1)
        p.setObjective("MIN",[(v,1)])
        self.assertFalse(p.run())

if __name__ == '__main__':
    unittest.main()
