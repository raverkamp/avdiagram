import argparse
from avdiagram import (
    cm,
    mm,
    Diagram,
    Table,
    Column,
    Rectangle,
    Relation,
    Line,
    Arrow,
    Inc12,
    DTextLines,
    CLine,
)


def cmd_example1(args) -> None:
    d = Diagram(cm(20), cm(20), True)

    t1 = d.text("text 1", 12)
    t2 = d.text("text 2 laenger", 24)
    t3 = d.text("text 3", 12)

    d.over(t1, mm(10), t2)
    d.over(t2, mm(20), t3)

    d.lalign([t1, t2, t3])

    t4 = d.text("text 4", 18)
    t5 = d.text("text 5 laenger", 10)
    t6 = d.text("text 6", 12)

    d.over(t4, mm(20), t5)
    d.over(t5, mm(10), t6)

    d.ralign([t4, t5, t6])

    d.left(t1, mm(100), t4)

    p1 = d.marker(mm(40), mm(30), True, name="p1_40_30")
    p2 = d.marker(None, None, True, name="p1_50_20_off")
    d.over(p1, mm(50), p2)
    d.left(p1, mm(20), p2)

    r = Rectangle(d, 1, "green")

    tl = d.textl(["A", "B", "C", "D"], 18)

    d.same(r.point(), tl.point())
    d.samev(r.width(), tl.width())
    d.samev(r.height(), tl.height())

    d.over(t3, mm(20), tl)

    d.show()


def cmd_example2(args):
    d = Diagram(cm(20), cm(20), True)
    r = Rectangle(d, mm(2), "#80ff80")

    tl = d.textl(["A", "B", "C", "D"], 18)

    d.same(r.point(), tl.point())
    d.samev(r.width(), tl.width())
    d.samev(r.height(), tl.height())
    d.show(True)


def mk_box(
    d: Diagram,
    txt: str,
    size: float = 12,
    borders=None,
    border_width=1,
    fgcolor="#000000",
    bgcolor="#ffffff",
    border_color="#000000",
):
    r = Rectangle(d, line_width=border_width, color=bgcolor)
    tl = d.textl(txt.split("\n"), size)
    if borders is None:
        (leftb, rightb, upb, lowb) = (size, size, size, size)
    elif isinstance(borders, (float, int)):
        (leftb, rightb, upb, lowb) = (borders, borders, borders, borders)
    else:
        (leftb, rightb, upb, lowb) = borders

    d.diffv(tl.point().x, r.point().x, leftb)
    d.diffv(tl.point().y, r.point().y, upb)

    d.diffv(r.width(), tl.width(), leftb + rightb)
    d.diffv(r.height(), tl.height(), upb + lowb)

    return r


def cmd_example3(args):
    d = Diagram(cm(20), cm(20), True)

    r = mk_box(d, "a\nbbb\ncccccc", 12, borders=mm(20))
    p = d.point(mm(20), mm(40))
    d.same(r.point(), p)

    r = mk_box(
        d, "a\nbbb\ncccccc", 24, borders=mm(10), bgcolor="#aaaaaa", border_width=mm(2)
    )
    p = d.point(mm(100), mm(40))
    d.same(r.point(), p)
    d.show()


def cmd_example4(args):
    d = Diagram(cm(20), cm(20), True)

    r1 = Rectangle(d, line_width=1, color="#ff0000")
    r2 = Rectangle(d, line_width=1, color="#ff00ff")
    r3 = Rectangle(d, line_width=1, color="#ffff00")

    p = d.point(mm(10), mm(10))

    d.same(r1.point(), p)
    d.samev(r1.width(), r2.width())
    d.samev(r1.width(), r3.width())
    d.samev(r1.height(), r2.height())
    d.samev(r1.height(), r3.height())

    d.add_constraint("C1", [(1, r1.width())], Relation.EQ, mm(20))
    d.add_constraint("C2", [(1, r1.height())], Relation.EQ, mm(40))

    d.over(r1, mm(40), r2)
    d.left(r1, mm(40), r3)

    d.lalign([r1, r2])

    d.samev(r1.point().y, r3.point().y)

    l = Line(d, 1)
    d.same(r1.port(12), l.p1())
    d.same(r3.port(32), l.p2())

    l = Line(d, 1)
    d.same(r1.port(27), l.p1())
    d.same(r2.port(2), l.p2())

    d.show()


def cmd_example5(args):
    d = Diagram(cm(20), cm(20), True)

    p1 = d.point(mm(20), mm(20))
    p2 = d.point(mm(90), mm(90))
    p3 = d.point(mm(10), mm(90))

    a1 = Arrow(d, mm(10))

    a2 = Arrow(d, mm(5), color="#ffa0a0", line_width=1, line_color="red")

    d.same(p1, a1.p1())
    d.same(p2, a1.p2())
    d.same(p1, a2.p1())
    d.same(p3, a2.p2())
    d.show()


def cmd_example6(args):
    d = Diagram(cm(20), cm(20), True)

    r = Rectangle(d, line_width=2, color="none")

    r1 = Rectangle(d, line_width=1, color="#ff0000")
    r2 = Rectangle(d, line_width=1, color="#ff0000")

    p = d.point(mm(20), mm(30))

    d.same(r1.point(), p)

    d.add_constraint("C1", r1.width(), Relation.EQ, mm(20))
    d.add_constraint("C2", r1.height(), Relation.EQ, mm(40))

    d.samev(r1.width(), r2.height())
    d.samev(r1.height(), r2.width())

    d.left(r1, mm(10), r2)
    d.over(r1, mm(20), r2)

    d.around(r.p1(), r.p2(), [r1, r2], (mm(2), mm(4), mm(8), mm(16)))

    d.show(True)


def cmd_example7(args):
    d = Diagram(cm(50), cm(50), True)

    r = Rectangle(d, line_width=2, color="#00ff00")

    r1 = Rectangle(d, line_width=1, color="#ff0000")

    p = d.point(mm(20), mm(30))

    d.same(r.point(), p)

    # d.same(r.point(), d.point("AAA", mm(10), mm(20)))

    d.add_constraint("C1", [(1, r1.width())], Relation.EQ, mm(20))
    d.add_constraint("C2", [(1, r1.height())], Relation.EQ, mm(40))

    d.diffv(r.width(), r1.width(), mm(40))
    d.diffv(r.height(), r1.height(), mm(15))

    d.centered(r.p1(), r.p2(), r1)

    r2 = Rectangle(d, color="#aaffaa", line_width=2)

    t = DTextLines(d, ["Hans Dampf", "in allen Gassen"], Inc12, 1)

    d.centered(r2.p1(), r2.p2(), t)
    d.diffv(r2.width(), t.width(), mm(40))
    d.left(r, mm(5), r2)
    d.over(r, mm(10), r2)

    d.show(True)


def cmd_example8(args):
    d = Diagram(cm(50), cm(50), True)

    p1 = d.point(mm(20), mm(30))
    p2 = d.point(mm(200), mm(30))
    p3 = d.point(mm(20), mm(200))

    p4 = d.point(mm(100), mm(100))

    p5 = d.point(mm(150), mm(110))
    p6 = d.point(mm(150), mm(60))

    CLine(d, [p1, p2], (1, 0), (-1, 0), ">")
    CLine(d, [p1, p3], (0, 1), (0, -1), ">")

    CLine(d, [p1, p4, p6, p5, p3], (1, 1), (1, -1), ">")
    CLine(d, [p1, p4, p3], (1, 1), (1, -1), ">")

    CLine(d, [p4, p4], (1, -1), (1, 1), ">")

    d.show(True)


def main():
    parser = argparse.ArgumentParser(
        prog="Examples",
        description="""Examples for Development""",
    )

    subparsers = parser.add_subparsers(required=True)

    p_ex1 = subparsers.add_parser("ex1")
    p_ex1.set_defaults(func=cmd_example1)

    p_ex2 = subparsers.add_parser("ex2")
    p_ex2.set_defaults(func=cmd_example2)

    p_ex3 = subparsers.add_parser("ex3")
    p_ex3.set_defaults(func=cmd_example3)

    p_ex4 = subparsers.add_parser("ex4")
    p_ex4.set_defaults(func=cmd_example4)

    p_ex5 = subparsers.add_parser("ex5")
    p_ex5.set_defaults(func=cmd_example5)

    p_ex6 = subparsers.add_parser("ex6")
    p_ex6.set_defaults(func=cmd_example6)

    p_ex7 = subparsers.add_parser("ex7")
    p_ex7.set_defaults(func=cmd_example7)

    p_ex8 = subparsers.add_parser("ex8")
    p_ex8.set_defaults(func=cmd_example8)

    args = parser.parse_args()
    args.func(args)


main()
