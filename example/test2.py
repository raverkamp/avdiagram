from avdiagram import cm, mm, Diagram, Table, Column, Line


def main() -> None:
    d = Diagram(cm(20), cm(20), True)

    t1 = d.text("text 1", 12)

    t2 = d.text("text 2", 12)

    d.left(t1, mm(4), t2)

    l = Line(d, mm(2))

    p = d.point(mm(100), mm(200))

    d.same(l.p1(), p)

    d.show()


main()
