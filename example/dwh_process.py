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


def rtext(d, text, color, border=mm(1), size=8):
    assert isinstance(text, str), "text parameter must be a string"
    assert isinstance(color, str), "color parameter must be a string"
    assert isinstance(border, float) and border >= 0, "border parameter must be >=0"
    assert isinstance(size, float) and size >= 0, "size parameter must be >=0"

    re = Rectangle(d, border, color)
    te = d.text(text, size)
    d.around(re.p1(), re.p2(), te, mm(10))
    return re


def cmd_process1(args) -> None:
    d = Diagram(cm(100), cm(100), True)

    #    sta = Rectangle(d,mm(1),"#0f0")
    #   sta_h = d.text("Staging",mm(20))

    staging = Rectangle(d, mm(1), color="#afa")

    t = DText(d, "Staging", mm(20))
    t1 = rtext(d, "Table 1", "#0f0", size=mm(10))
    t2 = rtext(d, "Table 2", "#0f0", size=mm(10))
    t3 = rtext(d, "Table 3", "#0f0", size=mm(10))

    d.over([t], mm(10), [t1])
    d.over([t1], mm(10), [t2])
    d.over([t2], mm(10), [t3])
    d.calign([t1, t2, t3])

    d.around(staging.p1(), staging.p2(), [t, t1, t2, t3], mm(2))

    psa = Rectangle(d, mm(1), color="#faa")
    psat = DText(d, "PSA", mm(20))

    pt1 = rtext(d, "PSA Table 1", "#f00", size=mm(10))
    pt2 = rtext(d, "PSA Table 2", "#f00", size=mm(10))

    d.over([psat], mm(10), [pt1])
    d.over([pt1], mm(10), [pt2])

    d.calign([psat, pt1, pt2])

    d.around(psa.p1(), psa.p2(), [psat, pt2, pt2], mm(2))
    d.over([staging], mm(20), [psa])

    cleansing = Rectangle(d, mm(1), color="#aaf")

    ct = DText(d, "CLEANSING", mm(20))

    ct1 = rtext(d, "Table 1", "#00f", size=mm(10))
    ct2 = rtext(d, "Table 2", "#00f", size=mm(10))
    ct3 = rtext(d, "Table 3", "#00f", size=mm(10))

    cls_tables = [ct1, ct2, ct3]

    d.over([ct], mm(10), [ct1])

    d.column(mm(10), cls_tables, 0.5)

    ma = rtext(d, "VIEW_KUCHEN", "#00f", size=mm(10))
    md = rtext(d, "VIEW_STEAK_MIT_POMMES", "#00f", size=mm(10))
    me = rtext(d, "VIEW_SALAT", "#00f", size=mm(10))
    co = rtext(d, "VIEW_SUPPE_MIT_MAGGI", "#00f", size=mm(10))
    sg = rtext(d, "VIEW_KAFFE", "#00f", size=mm(10))
    ps = rtext(d, "VIEW_MAGENPUTZER", "#00f", size=mm(10))

    cls_views = [ma, md, me, co, sg, ps]

    d.column(mm(10), cls_views, 0.5)

    d.left([ct1, ct2, ct3], mm(30), cls_views)

    d.around(cleansing.p1(), cleansing.p2(), cls_tables + cls_views, mm(2))
    d.over([ct], mm(10), [ma])
    d.some_align([cleansing, ct], 0.5)

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

    e2 = Ellipse(d, mm(1), "#dad")
    d.around(e2.port(315), e2.port(135), [DText(d, "More Processing", mm(5))], mm(10))

    d.left(cls_tables, mm(20), [e2])
    d.left([e2], mm(20), cls_views)
    d.over([ct], mm(1), [e2])

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
