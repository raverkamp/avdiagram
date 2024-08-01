import argparse
import pathlib
from avdiagram import (
    cm,
    mm,
    Diagram,
    Point,
    Table,
    Column,
    Rectangle,
    Relation,
    Line,
    Arrow,
    Inc12,
    DTextLines,
    DText,
    CLine,
    connect,
    Ellipse,
    add,
    mul,
)


class RectWithText(Rectangle):

    def __init__(
        self,
        d: "Diagram",
        text,
        name: str = "",
        line_width=mm(1),
        color="#eee",
        sz=mm(20),
    ):
        super().__init__(d, line_width, color, name)
        te = DText(d, text, sz)
        d.add_constraint("c", te.p1().y(), Relation.EQ, add(self.p1().y(), sz / 2))
        d.add_constraint("c0", te.p1().x(), Relation.GE, add(self.p1().x(), sz / 2))
        d.calign([te, self])
        d.add_constraint("c0", self.p1().x(), Relation.LE, te.p1().x())
        self._text = te
        self._ysplit = d.get_var("ysplit", None, None)
        d.add_constraint("c2", add(te.p2().y(), sz / 2), Relation.EQ, self._ysplit)
        d.add_constraint("c3", self._ysplit, Relation.LE, self.p2().y())

        self._p1_inner = d.point()
        d.samev(self._p1_inner.x(), self.p1().x())
        d.samev(self._p1_inner.y(), self._ysplit)

    def text(self):
        return self.text

    def p1_inner(self):
        return self._p1_inner


class EllipseWithText(Ellipse):

    def __init__(self, d: "Diagram", text, line_width=mm(1), color="#eee", sz=mm(20)):
        super().__init__(d, line_width, color)
        te = DText(d, text, sz)
        self._p1_inner = self.port(315)
        self._p2_inner = self.port(135)
        d.around(self._p1_inner, self._p2_inner, te, sz / 2)

    def p1_inner(self):
        return self._p1_inner

    def p2_inner(self):
        return self._p2_inner


def rtext(d, text, color, border=mm(1), size=8):
    assert isinstance(text, str), "text parameter must be a string"
    assert isinstance(color, str), "color parameter must be a string"
    assert isinstance(border, float) and border >= 0, "border parameter must be >=0"
    assert isinstance(size, float) and size >= 0, "size parameter must be >=0"

    re = Rectangle(d, border, color)
    te = d.text(text, size)
    d.around(re.p1(), re.p2(), te, mm(10))
    d.add_weight(re.width(), 2)
    return re


def cmd_process1(args) -> None:
    d = Diagram(cm(200), cm(100), True)

    staging = RectWithText(d, "STAGING", color="#afa")

    t1 = rtext(d, "Table 1", "#0f0", size=mm(10))
    t2 = rtext(d, "Table 2", "#0f0", size=mm(10))
    t3 = rtext(d, "Table 3 xxxxxxxxxxxxx", "#0f0", size=mm(10))

    d.over([t1], mm(10), [t2])
    d.over([t2], mm(10), [t3])
    d.calign([t1, t2, t3, staging])

    d.around(staging.p1_inner(), staging.p2(), [t1, t2, t3], mm(2))

    psa = RectWithText(d, "PSA", color="#faa")

    pt1 = rtext(d, "PSA Table 1", "#f00", size=mm(10))
    pt2 = rtext(d, "PSA Table 2", "#f00", size=mm(10))

    d.over([pt1], mm(10), [pt2])

    d.calign([psa, pt1, pt2])

    d.around(psa.p1_inner(), psa.p2(), [pt1, pt2], mm(2))
    d.over([staging], mm(20), [psa])

    cleansing = RectWithText(d, "CLEANSING", color="#aaf")

    ct1 = rtext(d, "Table 1", "#00f", size=mm(10))
    ct2 = rtext(d, "Table 2", "#00f", size=mm(10))
    ct3 = rtext(d, "Table 3", "#00f", size=mm(10))

    d.add_weight(ct1.height(), 100)

    cls_tables = [ct1, ct2, ct3]

    d.column(mm(10), cls_tables, 0.5)

    core_names = [
        "KUCHEN",
        "STEAK_MIT_POMMES",
        "SALAT",
        "SUPPE_MIT_MAGGI",
        "KAFFEE",
        "MAGENPUTZER",
    ]

    cls_views = [rtext(d, "VIEW_" + x, "#00f", size=mm(10)) for x in core_names]

    d.column(mm(10), cls_views, 0.5)

    d.left(cls_tables, mm(30), cls_views)

    e2 = EllipseWithText(d, "More Processing", sz=mm(10))

    d.samev(
        add(e2.p1().y(), e2.p2().y()), add(cleansing.p1_inner().y(), cleansing.p2().y())
    )

    d.samev(add(e2.p1().y(), e2.p2().y()), add(ct2.p1().y(), ct2.p2().y()))

    d.left(cls_tables, mm(20), [e2])
    d.left([e2], mm(60), cls_views)

    d.around(cleansing.p1_inner(), cleansing.p2(), cls_tables + cls_views + [e2], mm(2))

    d.left([staging, psa], mm(30), [cleansing])

    d.add_constraint(
        "mittig",
        add(staging.p1().y(), psa.p2().y()),
        Relation.EQ,
        add(cleansing.p1().y(), cleansing.p2().y()),
    )

    e = Ellipse(d, mm(1), "#dda")
    pro = DText(d, "Processing", mm(10))
    d.around(e.port(315), e.port(135), [pro], mm(10))

    d.left([staging, psa], mm(40), [e])

    d.left([e], mm(40), [cleansing])

    d.add_constraint(
        "a",
        add(e.p1().y(), e.p2().y()),
        Relation.EQ,
        add(staging.p1().y(), psa.p2().y()),
    )

    a = Arrow(d, mm(10))
    d.same(a.p1(), staging.port(15))
    d.same(a.p2(), e.port(300))

    a = Arrow(d, mm(10))
    d.same(a.p1(), psa.port(15))
    d.same(a.p2(), e.port(240))

    connect(e, 45, ct1, 35)
    connect(e, 90, ct2, 35)
    connect(e, 135, ct3, 35)

    for x in cls_views:
        connect(e2, 90, x, 35)

    i = 0
    for x in cls_tables:
        connect(x, 15, e2, 290 - i * 20)
        i += 1

    connect(e, 90, ct2, 35)
    connect(e, 135, ct3, 35)

    main = RectWithText(d, "MAIN", color="#aaf")

    main_tables = [rtext(d, "TABLE_" + x, "#00f", size=mm(10)) for x in core_names]

    d.column(mm(10), main_tables, 0.5)

    d.around(main.p1_inner(), main.p2(), main_tables, mm(10))

    d.left([cleansing], mm(20), [main])

    for v, t in zip(cls_views, main_tables):
        connect(v, 15, t, 35)

    dm = RectWithText(d, "DATAMART", color="#fdd")
    dm_tables = [rtext(d, "TEMP_" + x, "#f00", size=mm(10)) for x in core_names]

    d.column(mm(10), dm_tables, 0.5)

    d.left([main], mm(20), [dm])

    dmt = rtext(d, "DATA_TABLE", "#00f", size=mm(10))

    d.left(dm_tables, mm(80), [dmt])

    for v, t in zip(main_tables, dm_tables):
        connect(v, 15, t, 35)
        connect(t, 15, dmt, 35)

    d.around(dm.p1_inner(), dm.p2(), dm_tables + [dmt], mm(10))

    d.samev(add(dmt.p1().y(), dmt.p2().y()), add(dm.p1_inner().y(), dm.p2().y()))

    d.add_weight(dmt.height(), 1)

    d.show(True)


def main():
    parser = argparse.ArgumentParser(
        prog="DWH Process",
        description="""Examples of DWH dataflow""",
    )

    subparsers = parser.add_subparsers(required=True)

    ex = subparsers.add_parser("p1")
    ex.set_defaults(func=cmd_process1)

    args = parser.parse_args()
    args.func(args)


main()
