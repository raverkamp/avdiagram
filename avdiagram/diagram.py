import collections
from collections.abc import Sequence
from collections import namedtuple
from enum import Enum
import math
import tempfile
from typing import List, Optional, Union, Any, Tuple, Callable, cast
import webbrowser

USE_GLPK = True


from .svg import (
    Font,
    polyline,
    text,
    line,
    translate,
    path,
    render_file,
    Tag,
    Stuff,
    rect,
    points,
    empty_tag,
)

from . import glpk
from . import ortools_solver

from . import solver


def mk_counter() -> Callable[[], str]:
    a: List[int] = [0]

    def newid() -> str:
        a[0] = a[0] + 1
        return str(a[0])

    return newid


_newid = mk_counter()


def newid(s: Optional[str] = None) -> str:
    return (s or "") + "-" + _newid()


Table = namedtuple("Table", ["name", "columns", "pk"])
Column = namedtuple("Column", ["name", "type"])


def cm(x: float) -> float:
    return float(x) / 2.54 * 72.0


def mm(x: float) -> float:
    return cm(x / 10.0)


Inc12 = Font("Inconsolata", 12, "")


def text_width(font: Font, line: str) -> float:
    if font.font_family == "Inconsolata":
        return len(line) * font.font_size * 0.5
    else:
        raise Exception("unknown font:" + repr(font))


class DVar(object):
    def __init__(
        self,
        parent: "Diagram",
        name: str,
        lbound: Optional[float],
        ubound: Optional[float],
        objective: float,
    ):
        self.id = newid(name)
        self.name = name
        self.parent = parent
        self.lbound = lbound
        self.ubound = ubound
        if not ubound is None and not lbound is None:
            assert lbound <= ubound, "fail"
            self.fixed = lbound >= ubound
        self.objective = objective
        self.parent.vars.append(self)


class Relation(Enum):
    EQ = 1
    LE = 2
    GE = 3


class Constraint(object):
    def __init__(
        self, name: str, sums: List[Tuple[float, DVar]], op: Relation, const: float
    ):
        self.name = name
        self.op = op
        self.sums = sums
        self.const = const


class Base(object):
    """everything with at least one coordinate"""

    def __init__(self, d: "Diagram", name: str) -> None:
        self.name = name
        self.diagram = d

    def point(self):
        raise NotImplementedError()

    def to_svg(self, env) -> Tag:
        raise NotImplementedError()

class Point(Base):
    def __init__(self, d: "Diagram", name: str, visible: Optional[bool] = False):
        assert isinstance(d, Diagram)
        assert isinstance(name, str)
        super().__init__(d, name)

        self._xvar = d.get_var(name + "-x", 0, None)
        self._yvar = d.get_var(name + "-y", 0, None)
        self.visible = visible

        d.add_object(name, self)

    def point(self):
        return self

    def set_visible(self, v: bool):
        self.visible = v

    def x(self):
        return self._xvar

    def y(self):
        return self._yvar

    def to_svg(self, env) -> Tag:
        if self.visible:
            return translate(
                env(self.x()),
                env(self.y()),
                [
                    line(-10, 0, 10, 0, color="red"),
                    line(0, -10, 0, 10, color="red"),
                    text(10, -10, self.name),
                ],
            )
        else:
            return empty_tag



def flatten(s: Any) -> List[Base]:
    def flatten_internal(l: Any, res: List[Base]) -> None:
        assert not isinstance(l, str), "string not allowed:" + l
        if isinstance(l, Base):
            res.append(l)
        elif isinstance(l, collections.abc.Iterable):
            for a in l:
                flatten_internal(a, res)
        else:
            raise Exception("Wrong kind of object: " + repr(l))

    res: List[Base] = []
    flatten_internal(s, res)
    return res


class Line(Base):
    def __init__(self, d: "Diagram", name: str, line_width: float):
        super().__init__(d, name)
        self._p1 = d.point(name + "-p1")
        self._p2 = d.point(name + "-p2")
        self.line_width = line_width
        self._width = None
        self._height = None
        d.add_object(name, self)

    def point(self):
        return self.p1()

    def p1(self):
        return self._p1

    def p2(self):
        return self._p2

    def to_svg(self, env) -> Tag:
        return line(
            env(self.p1().x()),
            env(self.p1().y()),
            env(self.p2().x()),
            env(self.p2().y()),
            color="green",
        )

class CLine(Base):

    def __init__(self, d: "Diagram", name: str, points: List[Point], normal1: tuple[float, float],
                 normal2:tuple[float, float]):
        assert isinstance(points, list)
        assert len(points) >=2
        super().__init__(d, name)
        self.points = list(points)
        self.normal1 = normal1
        self.normal2 = normal2
        self.d = 50
        d.add_object(name, self)

    def point(self):
        return self.p1()

    def p1(self):
        return self.points[0]

    def p2(self):
        return self.points[-1]


    def bseg(
        self, x1: float, y1: float, mx: float, my: float, x2: float, y2: float, d: float
    ) -> str:
        (cx, cy) = cpfor(x1, y1, mx, my, x2, y2, d)
        return "{0} {1} ,  {2} {3}   ".format(cx, cy, mx, my)

    def to_svg(self, env) -> Tag:

        p1 = self.points[0]
        p2 = self.points[-1]
        # the start and endpoints
        x1 = round(env(p1.x()))
        y1 = round(env(p1.y()))
        x2 = round(env(p2.x()))
        y2 = round(env(p2.y()))


        # the control points for the endpoints
        #xxx
        (dx,dy) = vnormalize(self.normal1[0], self.normal1[1], self.d)
        cx1 = x1 + dx
        cy1 = y1 + dy

        (dx,dy) = vnormalize(self.normal2[0], self.normal2[1], self.d)
        cx2 = x2 + dx
        cy2 = y2 + dy

        s = "M {0} {1} C {2} {3}, ".format(x1, y1, cx1, cy1)
        ox = x1
        oy = y1
        liste = self.points[1:-1]
        for i in range(len(liste)):
            mx = round(env(liste[i].x()))
            my = round(env(liste[i].y()))

            if i == len(liste) - 1:
                nx = x2
                ny = y2
            else:
                nx = round(env(liste[i + 1].x()))
                ny = round(env(liste[i + 1].y()))
            s = s + self.bseg(ox, oy, mx, my, nx, ny, self.d) + " S "
            
        s = s + "{0} {1}, {2} {3}".format(cx2, cy2, x2, y2)
        return path(s, width=2, marker_end="url(#Triangle)")

class Arrow(Base):
    def __init__(
        self,
        d: "Diagram",
        name: str,
        width: float,
        color: str = "#a0a0a0",
        line_color: str = "#000000",
        line_width: float = 0,
    ):
        super().__init__(d, name)
        self._p1 = d.point(name + "-p1")
        self._p2 = d.point(name + "-p2")
        self._width = width
        self._color = color
        self._line_color = line_color
        self._line_width = line_width
        d.add_object(name, self)

    def point(self):
        return self.p1()

    def p1(self):
        return self._p1

    def p2(self):
        return self._p2

    def to_svg(self, env) -> Tag:
        p1x = env(self.p1().x())
        p1y = env(self.p1().y())
        p2x = env(self.p2().x())
        p2y = env(self.p2().y())
        width = self._width
        wo = width * 2
        ah = wo / 2
        l = math.sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)
        if ah >= l:
            ah = l
            wo = ah * 2
            width = wo
        il = l - ah

        points_ = [
            (0, width / 2),
            (il, width / 2),
            (il, wo / 2),
            (l, 0),
            (il, -wo / 2),
            (il, -width / 2),
            (0, -width / 2),
        ]

        a = Tag(
            "polygon",
            {
                "fill": self._color,
                "stroke": self._line_color,
                "stroke-width": str(self._line_width),
                "points": points(points_),
            },
            None,
        )
        b = Tag(
            "g", {"transform": f"rotate({180/math.pi*math.atan2(p2y-p1y,p2x-p1x)})"}, a
        )
        return Tag("g", {"transform": f"translate({p1x} {p1y})"}, b)


class Thing(Base):
    """something with a point at upper left and a positive witdh and height"""

    def __init__(self, d: "Diagram", name: str):
        super().__init__(d, name)
        self._p1 = d.point(name + "-p1")
        self._p2 = d.point(name + "-p2")
        self._width = d.get_var(name + "-width", 0, None)
        self._height = d.get_var(name + "-height", 0, None)
        d.add_constraint(
            name,
            [(1, self._p2.x()), (-1, self._p1.x()), (-1, self._width)],
            Relation.EQ,
            0,
        )
        d.add_constraint(
            name,
            [(1, self._p2.y()), (-1, self._p1.y()), (-1, self._height)],
            Relation.EQ,
            0,
        )

    def point(self):
        return self._p1

    def p1(self):
        return self._p1

    def p2(self):
        return self._p2

    def width(self):
        return self._width

    def height(self):
        return self._height

    def port(self, pos: float):
        if pos < 0 or pos > 40:
            raise Exception("out of bounds: " + str(pos))
        p = self.diagram.point("a")
        if pos < 10:
            self.diagram.convex_comb(p.x(), pos / 10.0, self.point().x(), self.p2().x())
            self.diagram.samev(p.y(), self.point().y())
        elif pos < 20:
            self.diagram.samev(p.x(), self.p2().x())
            self.diagram.convex_comb(
                p.y(), (pos - 10) / 10, self.point().y(), self.p2().y()
            )
        elif pos < 30:
            self.diagram.convex_comb(
                p.x(), 1 - (pos - 20) / 10.0, self.point().x(), self.p2().x()
            )
            self.diagram.samev(p.y(), self.p2().y())
        else:
            self.diagram.samev(p.x(), self.point().x())
            self.diagram.convex_comb(
                p.y(), 1 - (pos - 30) / 10, self.point().y(), self.p2().y()
            )
        return p


class Rectangle(Thing):
    def __init__(self, d: "Diagram", name: str, line_width: float, color: str):
        super().__init__(d, name)
        self.name = name
        self._line_width = line_width
        self._color = color

        d.add_object(name, self)

    def to_svg(self, env) -> Tag:
        return rect(
            env(self.p1().x()),
            env(self.p1().y()),
            env(self.p2().x()) - env(self.p1().x()),
            env(self.p2().y()) - env(self.p1().y()),
            color=self._color,
            line_width=self._line_width,
        )


#  diagram table
class DTable(Thing):
    def __init__(self, d: "Diagram", name: str, table: Table, font: Font) -> None:
        super().__init__(d, name)
        self.table = table
        self.font = font
        self.tb = mm(2)
        self.diagram = d

        name = table.name
        self.name = name

        tb = mm(1)
        h = (2 + len(table.columns)) * font.font_size * 1.5
        w = text_width(font, name)
        for c in table.columns:
            w = max(w, text_width(font, c.name + " " + c.type))
        width = w + 2 * tb
        height = h
        self.diagram.add_constraint("w", [(1, self.width())], Relation.EQ, width)
        self.diagram.add_constraint("h", [(1, self.height())], Relation.EQ, height)

        d.add_object(name, self)

    def to_svg(self, env) -> Tag:
        w = env(self.width())
        h = env(self.height())
        x = env(self.point().x())
        y = env(self.point().y())

        r = polyline([(0, 0), (w, 0), (w, h), (0, h), (0, 0)])
        bl = 1.5 * self.font.font_size
        th = text(self.tb, bl, self.table.name, self.font)
        li = line(
            0,
            bl + self.font.font_size * 0.2,
            w,
            bl + self.font.font_size * 0.2,
            width=2,
        )
        bl = bl + 2 * self.font.font_size
        lc = []
        for c in self.table.columns:
            tc = text(self.tb, bl, c.name + "  " + c.type, self.font)
            lc.append(tc)
            bl = bl + 1.5 * self.font.font_size
        return translate(x, y, cast(Stuff, [r, li, th, lc]))


class DText(Thing):
    def __init__(self, d: "Diagram", name: str, txt: str, sz: float):
        super().__init__(d, name)
        self.font = Font("Inconsolata", sz, "")
        self.text = txt

        width = text_width(self.font, txt)

        self.diagram.add_constraint("w", [(1, self.width())], Relation.EQ, width)
        self.diagram.add_constraint("h", [(1, self.height())], Relation.EQ, sz)

        d.add_object(name, self)

    def to_svg(self, env) -> Tag:
        return translate(
            env(self.point().x()),
            env(self.point().y()),
            [text(0, self.font.font_size, self.text, self.font)],
        )


class DTextLines(Thing):
    def __init__(
        self, d: "Diagram", name: str, txt: List[str], font: Font, border: float
    ):
        super().__init__(d, name)

        self.font = font
        wi = 0.0
        for l in txt:
            wi = max(wi, text_width(font, l))
        tb = mm(2)
        width = wi + 2 * tb
        height = (len(txt) - 1) * 1.5 * font.font_size + font.font_size + 2 * tb

        self.tb = border
        self.font = font
        self.text_lines = txt

        self.diagram.add_constraint("w", [(1, self.width())], Relation.EQ, width)
        self.diagram.add_constraint("h", [(1, self.height())], Relation.EQ, height)

        d.add_object(name, self)

    def to_svg(self, env):
        x = env(self.point().x())
        y = env(self.point().y())
        w = env(self.width())
        h = env(self.height())

        r = polyline([(0, 0), (w, 0), (w, h), (0, h), (0, 0)])
        bl = self.font.font_size + self.tb
        lc = []
        for line in self.text_lines:
            tc = text(self.tb, bl, line, self.font)
            lc.append(tc)
            bl = bl + 1.5 * self.font.font_size

        return translate(x, y, cast(Stuff, [lc]))


import csv


def load_tables(
    fname: str, tab_prefix: Optional[str] = None, col_prefix: Optional[str] = None
) -> dict[str, Table]:
    with open(fname, "r", newline="") as f:
        f.readline()
        reader = csv.reader(f, delimiter=";", quotechar='"')

        def keyy(r: tuple[str, str, str, str]) -> str:
            return r[1]

        def strip_tab_prefix(x: str) -> str:
            if tab_prefix and x.startswith(tab_prefix):
                return x[len(tab_prefix) :]
            else:
                return x

        def stripcolprefix(x: str) -> str:
            if col_prefix:
                p = x.find("_")
                if p and p > 0:
                    return x[p + 1 :]
                else:
                    return x
            else:
                return x

        lines = list(reader)
        a = filter(
            lambda x: len(x) > 1 and (not tab_prefix or x[1].startswith(tab_prefix)),
            lines,
        )
        a2 = map(
            lambda x: (x[0], strip_tab_prefix(x[1]), stripcolprefix(x[2]), x[3]), a
        )

        import itertools

        b = itertools.groupby(a2, keyy)
        tabs = {}
        for _, rows_ in b:
            rows = list(rows_)
            name = rows[0][1]
            l = []
            for r in rows:
                col = Column(r[2], r[3])
                l.append(col)
            tabs[name.upper()] = Table(name, l, [])

        return tabs


def posfornumbase(wi: float, he: float, p: float) -> Tuple[float, float]:
    if p < 0:
        raise Exception("position number to small")
    elif p < 10:
        return (p / 10.0 * wi, 0)
    elif p < 20:
        return (wi, (p - 10) * he / 10.0)
    elif p < 30:
        return ((30 - p) / 10 * wi, he)
    elif p < 40:
        return (0, (40 - p) / 10 * he)
    else:
        raise Exception("position number to large")


# vector length


def vlen(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)

def vnormalize(x: float,y:float,d:float)->tuple[float, float]:
    le = vlen(x,y)
    return (x*d/le, y*d/le)


# controlpoint computation
# the curve goes from point (x1,x2) over (mx,my) to (x2,y2)
# the controlpoint after (x1,y1) is known
# we need the control point before point (mx,my)
# the direction of the tangent in (mx,my) should be the same as (x1,y1) to (x2,y2)


def cpfor(
    x1: float, y1: float, mx: float, my: float, x2: float, y2: float, d: float
) -> Tuple[float, float]:
    dx = x2 - x1
    dy = y2 - y1
    da = vlen(dx, dy)

    # is this a sane default? better prependicular to point 1 and point m?
    if da == 0:
        return (x1, y1)

    # min distances to middlepoint
    dm = min(vlen(mx - x1, my - y1), vlen(mx - x2, mx - y2))

    # offset for controlpoint, subtract from mx,my
    ddx = dx / da * d
    ddy = dy / da * d
    # length of offset is d

    # the length of the offset should be less than the distances to the middlepoint
    if d > dm / 2.0:
        f = dm / 2.0 / d
    else:
        f = 1
    return (mx - f * ddx, my - f * ddy)


class Diagram(object):
    def __init__(self, w: float, h: float, debug: bool = False) -> None:
        # the base date, the importetd table, not necessarly all displayed

        # the obejcts indexed by name
        # self.objects: dict[str, Base] = {}
        self.object_list: List[Base] = []

        # these are our constaints
        self.constraints: List[Constraint] = []

        # the constraint engine
        self.cons = None

        # the variables
        self.vars: List[DVar] = []

        self.width = w
        self.height = h
        # the border around things
        self.border = mm(10)

        self.clines: List[tuple[Thing, float, Thing, float, List[Point]]] = []

        self.default_font = Inc12

        self.counter = 0
        self.debug = debug
        self.zero_var = self.get_var("zero", 0, 0)

    def add_constraint(
        self, name: str, sums: List[Tuple[float, DVar]], op: Relation, const: float
    ) -> None:
        var_ids = set()
        for _, var in sums:
            if var.id in var_ids:
                raise Exception("duplicate variable in constraint: " + repr(var))
            var_ids.add(var.id)
        c = Constraint(name, sums, op, const)
        self.constraints.append(c)

    def same(self, p1: Point, p2: Point):
        self.add_constraint("SAME", [(1, p1.x()), (-1, p2.x())], Relation.EQ, 0)
        self.add_constraint("SAME", [(1, p1.y()), (-1, p2.y())], Relation.EQ, 0)

    def samev(self, v1: DVar, v2: DVar):
        self.add_constraint("SAMEV", [(1, v1), (-1, v2)], Relation.EQ, 0)

    def diffv(self, v1: DVar, v2: DVar, diff: float):
        self.add_constraint("DIFFV", [(1, v1), (-1, v2)], Relation.EQ, diff)

    def left(self, t1: Any, c: float, t2: Any) -> None:
        l1 = flatten(t1)
        if len(l1) == 0:
            return
        l2 = flatten(t2)
        if len(l2) == 0:
            return

        varr = self.get_var("cons-left", None, None)

        for kobj in l1:
            obj = kobj
            sum1: List[tuple[float, DVar]] = [(1, varr)]
            if isinstance(obj, Point):
                sum1.append((-1, obj.x()))
            elif isinstance(obj, Thing):
                sum1.append((-1, obj.point().x()))
                sum1.append((-1, obj.width()))
            else:
                raise Exception("unknown")
            self.add_constraint("left1", sum1, Relation.GE, c)

        for kobj in l2:
            obj = kobj
            sum2: List[tuple[float, DVar]] = [(-1, varr), (1, obj.point().x())]
            self.add_constraint("left2", sum2, Relation.GE, 0)

    def over(self, t1: Any, c: float, t2: Any) -> None:
        l1 = flatten(t1)
        if len(l1) == 0:
            return
        l2 = flatten(t2)
        if len(l2) == 0:
            return
        varr = self.get_var("cons-over", None, None)

        for kobj in l1:
            obj = kobj
            sum1: List[tuple[float, DVar]] = [(1, varr)]
            if isinstance(obj, Point):
                sum1.append((-1, obj.y()))
            elif isinstance(obj, Thing):
                sum1.append((-1, obj.point().y()))
                sum1.append((-1, obj.height()))
            else:
                raise Exception("unknown:" + repr(obj))
            self.add_constraint("left1", sum1, Relation.GE, c)

        for kobj in l2:
            obj = kobj
            sum2: List[tuple[float, DVar]] = [(-1, varr)]
            if isinstance(obj, Point):
                sum2.append((1, obj.y()))
            elif isinstance(obj, Thing):
                sum2.append((1, obj.point().y()))
            else:
                raise Exception("unknown")
            self.add_constraint("left2", sum2, Relation.GE, 0)

    def lcol(self, dist: float, objs: list[Base]) -> None:
        l = flatten(objs)
        if len(l) == 0:
            return
        self.lalign(l)

        obj0 = l[0]

        for kobj in l[1:]:
            obj = kobj
            lu = [(-1.0, obj0.point().y())]
            if isinstance(obj0, Thing):
                lu.append((-1.0, obj0.height))
            lu.append((1, obj.point().y()))
            self.add_constraint("y dist", lu, Relation.EQ, dist)
            obj0 = obj

    def lalign(self, l: List[Base]) -> None:
        if len(l) <= 1:
            return
        var0 = l[0].point().x()
        for q in l[1:]:
            var1 = q.point().x()
            self.add_constraint("lalign", [(1, var0), (-1, var1)], Relation.EQ, 0)

    def some_align(self, l: List[Base], fac: float) -> None:
        if len(l) <= 1:
            return

        obj0 = l[0]
        if isinstance(obj0, Thing):
            l0 = [(float(1), obj0.point().x()), (fac, obj0.width())]
        else:
            l0 = [(float(1), obj0.point().x())]

        for q in l[1:]:
            obj1 = q
            if isinstance(obj1, Thing):
                l1 = [(-1.0, obj1.point().x()), (-fac, obj1.width())]
            else:
                l1 = [(float(-1), obj1.point().x())]

            self.add_constraint("ralign", l0 + l1, Relation.EQ, 0)

    def ralign(self, l: List[Base]) -> None:
        return self.some_align(l, 1)

    def calign(self, l: List[Base]) -> None:
        return self.some_align(l, 0.5)

    def pos(
        self,
        obj: Point,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        if not x is None:
            self.add_constraint("posx", [(1, obj.x())], Relation.EQ, x)
        if not y is None:
            self.add_constraint("posy", [(1, obj.y())], Relation.EQ, y)

    def get_var(
        self,
        name: str,
        lbound: Optional[float],
        ubound: Optional[float],
        objective: float = 1,
    ) -> DVar:
        return DVar(self, name, lbound, ubound, objective)

    def point(
        self,
        name: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        visible: Optional[bool] = False,
    ) -> Point:
        assert isinstance(name, str)
        point = Point(self, name, visible)
        if not x is None:
            self.add_constraint("FIX", [(1, point.x())], Relation.EQ, x)
        if not y is None:
            self.add_constraint("FIX", [(1, point.y())], Relation.EQ, y)
        return point

    def real_point(
        self, name: str, x: Optional[float] = None, y: Optional[float] = None
    ) -> Point:
        point = Point(self, name)
        self.add_constraint("real", [(1, point.x())], Relation.GE, 0)
        self.add_constraint("real", [(1, point.y())], Relation.GE, 0)

        if not x is None:
            self.add_constraint("FIX", [(1, point.x())], Relation.EQ, x)
        if not y is None:
            self.add_constraint("FIX", [(1, point.y())], Relation.EQ, y)
        return point

    def bound(self, var: DVar, l: List[DVar], up: bool, dist: float):
        rel = Relation.LE if up else Relation.GE
        x = "up" if up else "lower"
        coeff: List[Tuple[float, DVar]] = []
        for v in l:
            self.add_constraint(
                "bounding-" + x, [(1, v), (-1, var)], rel, -dist if up else dist
            )
            coeff.append((1, v))
        if up:
            v2 = self.get_var("boundvar-" + x, None, None, -1000)
        else:
            v2 = self.get_var("boundvar-" + x, None, None, 1000)
        coeff.append((-len(l), var))
        coeff.append((-1, v2))
        self.add_constraint("def-boundvar-" + x, coeff, Relation.EQ, 0)

    def topb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            l.append(obj.point().y())
        self.bound(var, l, False, dist)

    def leftb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            l.append(obj.point().x())
        self.bound(var, l, False, dist)

    def bottomb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            if isinstance(obj, Thing):
                l.append(obj.p2().y())
            else:
                l.append(obj.point().y())
        self.bound(var, l, True, dist)

    def rightb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            if isinstance(obj, Thing):
                l.append(obj.p2().x())
            else:
                l.append(obj.point().x())
        self.bound(var, l, True, dist)

    def around(self, p1: Point, p2: Point, objs: Any, bounds):
        if isinstance(bounds, (float, int)):
            (t, l, b, r) = (bounds, bounds, bounds, bounds)
        else:
            (l, r, t, b) = bounds
        self.topb(p1.y(), objs, t)
        self.leftb(p1.x(), objs, l)
        self.bottomb(p2.y(), objs, b)
        self.rightb(p2.x(), objs, r)

    def varcentered(self, vo1: DVar, vo2: DVar, vi1: DVar, vi2: DVar):
        self.add_constraint("centered", [(1, vi1), (-1, vo1)], Relation.GE, 0)
        self.add_constraint("centered", [(-1, vi2), (1, vo2)], Relation.GE, 0)
        self.add_constraint(
            "centered-same", [(1, vi1), (-1, vo1), (1, vi2), (-1, vo2)], Relation.EQ, 0
        )

    def centered(self, p1: Point, p2: Point, t: Thing):
        self.varcentered(p1.x(), p2.x(), t.p1().x(), t.p2().x())
        self.varcentered(p1.y(), p2.y(), t.p1().y(), t.p2().y())

    def add_object(self, name, o):
        self.object_list.append(o)

    def text(self, name: str, txt: str, sz: float) -> Thing:
        return DText(self, name, txt, sz)

    def textl(self, name: str, txt: List[str], sz: float) -> Thing:
        font = Font("Inconsolata", sz, "")
        tb = mm(2)
        t = DTextLines(self, name, list(txt), font, tb)
        return t

    def table(self, tab: Table) -> Thing:
        font = Font("Inconsolata", 12, "")
        dtab = DTable(self, tab.name, tab, font)
        return dtab

    def convex_comb(self, v: DVar, x: float, v1: DVar, v2: DVar):
        assert isinstance(v1, DVar)
        assert isinstance(v2, DVar)
        assert isinstance(x, (float, int))
        assert x >= 0 and x <= 1
        if x == 0:
            self.samev(v, v1)
        if x == 1:
            self.samev(v, v2)
        self.add_constraint("a", [(1 - x, v1), (x, v2), (-1, v)], Relation.EQ, 0)
        return v

    # a connection line
    def cline(
        self,
        t1: Thing,
        p1: float,
        t2: Thing,
        p2: float,
        liste: List[Point] = [],
    ) -> None:
        a1 = t1
        if not isinstance(a1, Thing):
            raise Exception("wrong type")
        a2 = t2
        if not isinstance(a2, Thing):
            raise Exception("wrong type")

        liste_: List[Point] = []
        for x in liste:
            xx = x
            if not isinstance(xx, Point):
                raise Exception("not a point")
            liste_.append(xx)
        self.clines.append((a1, float(p1), a2, float(p2), liste_))

    def posfornum(self, env, obj: Thing, p: float) -> Tuple[float, float]:
        x = env(obj.point().x())
        y = env(obj.point().y())
        wi = env(obj.width())
        he = env(obj.height())
        (ox, oy) = posfornumbase(wi, he, p)
        return (x + ox, y + oy)

    def cp(self, x: float, y: float, p: float, d: float) -> Tuple[float, float]:
        if 0 <= p < 10:
            return (x, y - d)
        elif p < 20:
            return (x + d, y)
        elif p < 30:
            return (x, y + d)
        elif p < 40:
            return (x - d, y)
        else:
            raise Exception("bad value for p")

    def bseg(
        self, x1: float, y1: float, mx: float, my: float, x2: float, y2: float, d: float
    ) -> str:
        (cx, cy) = cpfor(x1, y1, mx, my, x2, y2, d)
        return "{0} {1} {2} {3}  ".format(cx, cy, mx, my)

    def bpath(
        self,
        env,
        obj1: Thing,
        p1: float,
        obj2: Thing,
        p2: float,
        liste: List[Point],
    ) -> Tag:
        d = 100
        (x1, y1) = self.posfornum(env, obj1, p1)
        (x2, y2) = self.posfornum(env, obj2, p2)
        (cx1, cy1) = self.cp(x1, y1, p1, d)
        (cx2, cy2) = self.cp(x2, y2, p2, d)

        s = "M {0},{1} C {2},{3} ".format(x1, y1, cx1, cy1)
        ox = x1
        oy = x2
        for i in range(len(liste)):
            mx = env(liste[i].x())
            my = env(liste[i].y())

            if i == len(liste) - 1:
                nx = x2
                ny = y2
            else:
                nx = env(liste[i + 1].x())
                ny = env(liste[i + 1].y())
            s = s + self.bseg(ox, oy, mx, my, nx, ny, d) + " S "
        s = s + "{0},{1} {2},{3}".format(cx2, cy2, x2, y2)
        return path(s, width=2, marker_end="url(#Triangle)")

    def solve_problem(self, verbose=False):
        var_list = []
        for v in self.vars:
            var_list.append((v.id, v.lbound, v.ubound))

        cons_list = []
        for cons in self.constraints:
            l = []
            for fac, var in cons.sums:
                l.append((fac, var.id))
            (low, up) = (None, None)
            if cons.op in (Relation.GE, Relation.EQ):
                low = cons.const
            if cons.op in (Relation.LE, Relation.EQ):
                up = cons.const
            cons_list.append((newid(cons.name), l, low, up))

        objective = []
        for v in self.vars:
            objective.append((v.objective, v.id))

        if USE_GLPK:
            solver = glpk.GLPKSolver()
        else:
            solver = ortools_solver.ORToolsSolver()
        sol = solver.solve_problem(
            var_list, cons_list, objective, "min", verbose=verbose
        )
        return sol

    def create_svg(self, values: dict[str, float]) -> list[Stuff]:
        l: list[Stuff] = []
        # the border
        r0 = polyline(
            [
                (0, 0),
                (self.width, 0),
                (self.width, self.height),
                (0, self.width),
                (0, 0),
            ]
        )
        l.append(r0)

        def env(varr):
            return values[varr.id]

        for obj in self.object_list:
            if isinstance(obj, Point):
                if self.debug:
                    l.append(obj.to_svg(env))
            else:
                svgtab = obj.to_svg(env)
                l.append(svgtab)

        for o1, p1, o2, p2, liste in self.clines:
            paa = self.bpath(env, o1, p1, o2, p2, liste)
            l.append(paa)
        return l

    def show(self, verbose=False) -> None:
        res = self.solve_problem(verbose)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".svg", delete=False) as f:
            filename = f.name
            print("SVG Filename:" + filename)
            l = self.create_svg(res)
            render_file(filename, self.width, self.height, [l])
            c = webbrowser.get("firefox")
            c.open("file://" + filename, autoraise=False)
