"""simple diagram tests"""
import os
import tempfile
import unittest
from avdiagram import diagram, mm
from tempfile import NamedTemporaryFile


class Test_simple(unittest.TestCase):
    def test_success(self):
        d = diagram.Diagram(mm(200), mm(200))
        t1 = d.text("t1", "abc", 12)
        t2 = d.text("t2", "xyz", 12)
        d.over(t1, mm(2), t2)
        f = NamedTemporaryFile(delete=False)
        file_name = f.name
        f.close()
        try:
            x = d.run(file_name)
        finally:
            os.remove(file_name)
        self.assertTrue(x)

    def test_fail(self):
        d = diagram.Diagram(mm(200), mm(200))
        t1 = d.text("t1", "abc", 12)
        t2 = d.text("t2", "xyz", 12)
        d.over(t1, mm(2), t2)
        d.over(t2, mm(2), t1)
        f = NamedTemporaryFile(delete=False)
        file_name = f.name
        f.close()
        try:
            x = d.run(file_name)
        finally:
            os.remove(file_name)
        self.assertFalse(x)

    def test_fail2(self):
        d = diagram.Diagram(mm(200), mm(200))
        t1 = d.text("t1", "abc", 12)
        with self.assertRaises(Exception):
            d.lalign([t1, t1])


if __name__ == "__main__":
    unittest.main()
