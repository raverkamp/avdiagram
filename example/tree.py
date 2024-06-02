import argparse
import pathlib
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
    connect,
    add, mul
)



def cmd_example1(args) -> None:
    d = Diagram(cm(50), cm(50), True)

    def create_tree(t):
        (lab, children) = t
        #returns a tree of all componenets
        label = d.text(lab, 12)
        last_t = None
        first_head = None
        last_head = None
        all = [label]
        for child in children:
            (head, compos) = create_tree(child)
            connect(label,25, head, 5)
            if not first_head:
                first_head = head
            last_head = head
            d.over(label, mm(20),compos)
            if last_t:
                d.left(last_t,mm(10),compos)
            last_t = compos
            all.append(compos)
        if not first_head is None:
            d.add_constraint("mittig", add(first_head.p1().x(), last_head.p2().x()),Relation.EQ,
                             add(label.p1().x(), label.p2().x()))
        return (label, all)

    # a tree is a 2-tuple with a label and list of children which are also trees.
    create_tree(("AAAAAAA",[("BBBB",[("EEEEEEE",[]), ("GGGG",[("HHHHH",[])])]),("CCCC",[("DDDD",[("GGGGGGGGG",[])])])]))
    
    d.show(False)

    
def cmd_dir_tree(args) -> None:
    p = pathlib.Path(args.dir).resolve()
    
    d = Diagram(cm(500), cm(500), True)

    # returns head, around points
    def build_tree(p):
        label = d.text(p.name, 12)
        if p.is_dir():
            l = []
            for child in p.iterdir():
                l.append(build_tree(child))
        else:
            l = []
        if not l:
            p1 = label.p1()
            p2 = label.p2()
            return (label, p1, p2)
        p1 = d.point()
        p2 = d.point()
        a = list([label] + [px for (_,px,_) in l] + [py for (_,_,py) in l])

        d.around(p1, p2, a, 0)
       
        d.add_constraint("mittig", add(label.p1().x(), label.p2().x()), Relation.EQ,
                         add(p1.x(), p2.x()))

        for (lab,_,_) in l:
            connect(label,25, lab, 5)
        for ((alab,ap1,ap2),(blab,bp1,bp2)) in zip(l,l[1:]):
            d.left(ap2,mm(3), bp1)
            d.add_constraint("eq",ap1.y(), Relation.EQ, bp1.y())
        d.over(label,mm(20), l[0][1])
        return (label, p1, p2)

    build_tree(p)
    d.show(True)
    
def main():
    parser = argparse.ArgumentParser(
        prog="Examples",
        description="""Examples for Development""",
    )

    subparsers = parser.add_subparsers(required=True)

    ex = subparsers.add_parser("ex1")
    ex.set_defaults(func=cmd_example1)

    ex = subparsers.add_parser("dir")
    ex.add_argument("dir", type=str)
    ex.set_defaults(func=cmd_dir_tree)

    args = parser.parse_args()
    args.func(args)



main()
