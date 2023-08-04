from avdiagram import cm, mm, Diagram, Table, Column, Line


def main() -> None:
    d = Diagram(cm(20), cm(20), True)

    t1 = d.text("t1", "text 1", 12)

    t2 = d.text("t2", "text 2", 12)

    d.left(t1, mm(4), t2)

    l = Line(d, "line", mm(2))

    p = d.point("P", mm(100), mm(200))

    d.same(l.p1(), p)

    d.show()


main()
