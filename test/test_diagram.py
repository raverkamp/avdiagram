"""simple diagram tests"""
import os
import tempfile
import unittest
from avdiagram import diagram, mm

class Test_simple(unittest.TestCase):

    def test_success(self):
        d = diagram.Diagram(mm(200), mm(200))
        t1 = d.text("t1","abc",12)
        t2 = d.text("t2","xyz",12)
        d.over(t1,mm(2),t2)
        (_,a) = tempfile.mkstemp()
        try:
            x = d.run(a)
        finally:
            os.remove(a)
        self.assertTrue(x)


    def test_fail(self):
        d = diagram.Diagram(mm(200), mm(200))
        t1 = d.text("t1","abc",12)
        t2 = d.text("t2","xyz",12)
        d.over(t1,mm(2),t2)
        d.over(t2,mm(2),t1)
        (_,a) = tempfile.mkstemp()
        try:
            x = d.run(a)
        finally:
            os.remove(a)
        self.assertFalse(x)

if __name__ == '__main__':
    unittest.main()
