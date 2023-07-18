from avdiagram import cm, mm, Diagram, Table, Column


def main() -> None:
    d = Diagram(cm(20), cm(20), True)

    t1 = d.text("t1", "text 1", 12)
    t2 = d.text("t2", "text 2 laenger", 24)
    t3 = d.text("t3", "text 3", 12)

    d.over(t1, mm(10), t2)
    d.over(t2, mm(20), t3)

    d.lalign([t1, t2, t3])

    t4 = d.text("t4", "text 4", 18)
    t5 = d.text("t5", "text 5 laenger", 10)
    t6 = d.text("t6", "text 6", 12)

    d.over(t4, mm(20), t5)
    d.over(t5, mm(10), t6)

    d.ralign([t4, t5, t6])

    d.left("t1", mm(100), "t4")

    p1 = d.point("p1_40_30", mm(40), mm(30))
    p2 = d.point("p1_50_20_off")
    d.over(p1, mm(50), p2)
    d.left(p1, mm(20), p2)

    tl = d.textl("tl", ["A", "B", "C", "D"], 18)

    d.over(t3, mm(20), tl)

    d.show()


main()
