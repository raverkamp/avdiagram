import collections
from collections.abc import Sequence
from collections import namedtuple

from enum import Enum
import math
from numbers import Number
import tempfile
from typing import List, Optional, Union, Any, Tuple, Callable, cast, NamedTuple
import webbrowser

USE_GLPK = False  # True


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
    ellipse,
)

from . import glpk
from . import ortools_solver

from . import solver


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
        self.id = parent.newid(name)
        self.name = name
        self.parent = parent
        self.lbound = lbound
        self.ubound = ubound
        if not ubound is None and not lbound is None:
            assert lbound <= ubound, "fail"
            self.fixed = lbound >= ubound
        self.objective = objective
        self.parent.vars.append(self)


def simplify_summands(summands: List[Tuple[float, DVar]]) -> List[Tuple[float, DVar]]:
    d = {}
    for fac, var in summands:
        assert isinstance(fac, (int, float))
        assert isinstance(var, DVar)
        if fac == 0:
            continue
        if not var in d:
            d[var] = fac
        else:
            d[var] = d[var] + fac
    res: List[Tuple[float, DVar]] = []
    for var in d:
        res.append((d[var], var))
    return res


class Term:
    def __init__(self, summands: List[Tuple[float, DVar]], const) -> None:
        assert isinstance(const, (int, float))
        self.summands = summands
        self.const = const


def as_term(x) -> Term:
    if isinstance(x, Term):
        return x
    if isinstance(x, (int, float)):
        return Term([], x)
    if isinstance(x, DVar):
        return Term([(1, x)], 0)
    if isinstance(x, Sequence):
        l: List[Tuple[float, DVar]] = []
        for c, v in x:
            assert isinstance(c, (int, float)), "factor must be float"
            assert isinstance(v, DVar), "var must be DVar"
            l.append((c, v))
        return Term(simplify_summands(l), 0)
    raise Exception("can not convert to term: " + repr(x))


def addl(terms) -> Term:
    const = 0
    l = []
    for t in terms:
        term = as_term(t)
        const = const + term.const
        l.extend(term.summands)
    l2 = simplify_summands(l)
    return Term(l2, const)


def add(*terms) -> Term:
    return addl(terms)


def mul(const, term):
    t = as_term(term)
    assert isinstance(const, Number)
    l = [(c * const, v) for (c, v) in t.summands]
    return Term(l, t.const * const)


def sub(t1, t2) -> Term:
    return add(t1, mul(-1, t2))


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
        assert isinstance(name, str)
        self.name = name
        self.diagram = d

    def point(self):
        raise NotImplementedError()

    def to_svg(self, env) -> Tag:
        raise NotImplementedError()


class Point(NamedTuple):
    x: DVar
    y: DVar


class Marker(Base):
    def __init__(self, d: "Diagram", visible: Optional[bool] = False, name: str = ""):
        assert isinstance(d, Diagram)
        assert isinstance(name, str)
        assert visible in (True, False)
        super().__init__(d, name)

        self._xvar = d.get_var(self.name + "-x", 0, None)
        self._yvar = d.get_var(self.name + "-y", 0, None)
        self.visible = visible
        self.p = Point(self._xvar, self._yvar)

        d.add_object(self)

    def point(self):
        return self.p

    def set_visible(self, v: bool):
        self.visible = v

    def to_svg(self, env) -> Tag:
        if self.visible:
            return translate(
                env(self.p.x),
                env(self.p.y),
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
        if isinstance(l, (Base, Point)):
            res.append(l)
        elif isinstance(l, collections.abc.Sequence):
            for a in l:
                flatten_internal(a, res)
        else:
            raise Exception("Wrong kind of object: " + repr(l))

    res: List[Base] = []
    flatten_internal(s, res)
    return res


class Line(Base):
    def __init__(self, d: "Diagram", line_width: float, name: str = ""):
        assert isinstance(line_width, (float, int))
        super().__init__(d, name)
        self._p1 = d.point(name=self.name + "-p1")
        self._p2 = d.point(name=self.name + "-p2")
        self.line_width = line_width
        self._width = None
        self._height = None
        d.add_object(self)

    def point(self):
        return self.p1()

    def p1(self):
        return self._p1

    def p2(self):
        return self._p2

    def to_svg(self, env) -> Tag:
        return line(
            env(self.p1().x),
            env(self.p1().y),
            env(self.p2().x),
            env(self.p2().y),
            color="green",
        )


class CLine(Base):
    def __init__(
        self,
        d: "Diagram",
        points: List[Point],
        normal1: tuple[float, float],
        normal2: tuple[float, float],
        connector2: str,
        name: str = "",
    ):
        assert isinstance(points, list)
        assert len(points) >= 2
        super().__init__(d, name)
        self.points = list(points)
        self.normal1 = normal1
        self.normal2 = normal2
        self.connector2 = connector2
        self.d = 50
        d.add_object(self)

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
        x1 = round(env(p1.x))
        y1 = round(env(p1.y))
        x2 = round(env(p2.x))
        y2 = round(env(p2.y))

        # the control points for the endpoints
        # xxx
        (dx, dy) = vnormalize(self.normal1[0], self.normal1[1], self.d)
        cx1 = x1 + dx
        cy1 = y1 + dy

        (dx, dy) = vnormalize(self.normal2[0], self.normal2[1], self.d)
        cx2 = x2 + dx
        cy2 = y2 + dy

        s = "M {0} {1} C {2} {3}, ".format(x1, y1, cx1, cy1)
        ox = x1
        oy = y1
        liste = self.points[1:-1]
        for i in range(len(liste)):
            mx = round(env(liste[i].x))
            my = round(env(liste[i].y))

            if i == len(liste) - 1:
                nx = x2
                ny = y2
            else:
                nx = round(env(liste[i + 1].x))
                ny = round(env(liste[i + 1].y))
            s = s + self.bseg(ox, oy, mx, my, nx, ny, self.d) + " S "

        s = s + "{0} {1}, {2} {3}".format(cx2, cy2, x2, y2)
        if self.connector2 == ">":
            marker_end = "url(#Triangle)"
        elif self.connector2 == "":
            marker_end = ""
        else:
            raise Exception(f"unkown end marker: {self.connector2}")
        return path(s, width=2, marker_end=marker_end)


class Arrow(Base):
    def __init__(
        self,
        d: "Diagram",
        width: float,
        color: str = "#a0a0a0",
        line_color: str = "#000000",
        line_width: float = 0,
        name: str = "",
    ):
        super().__init__(d, name)
        self._p1 = d.point(name=self.name + "-p1")
        self._p2 = d.point(name=self.name + "-p2")
        self._width = width
        self._color = color
        self._line_color = line_color
        self._line_width = line_width
        d.add_object(self)

    def point(self):
        return self.p1()

    def p1(self):
        return self._p1

    def p2(self):
        return self._p2

    def to_svg(self, env) -> Tag:
        p1x = env(self.p1().x)
        p1y = env(self.p1().y)
        p2x = env(self.p2().x)
        p2y = env(self.p2().y)
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
        self._p1 = d.point(name=self.name + "-p1")
        self._p2 = d.point(name=self.name + "-p2")
        self._width = d.def_var(sub(self._p2.x, self._p1.x))
        self._height = d.def_var(sub(self._p2.y, self._p1.y))
        d.add_constraint(name, self._width, Relation.GE, 0)
        d.add_constraint(name, self._height, Relation.GE, 0)

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

    def left(self):
        return self._p1.x

    def right(self):
        return self._p2.x

    def top(self):
        return self._p1.y

    def bottom(self):
        return self._p2.y

    def port(self, pos: float):
        if pos < 0 or pos > 40:
            raise Exception("out of bounds: " + str(pos))
        p = self.diagram.point(name=self.name + "port-" + str(pos))
        if pos < 10:
            self.diagram.convex_comb(p.x, pos / 10.0, self.point().x, self.p2().x)
            self.diagram.samev(p.y, self.point().y)
        elif pos < 20:
            self.diagram.samev(p.x, self.p2().x)
            self.diagram.convex_comb(p.y, (pos - 10) / 10, self.point().y, self.p2().y)
        elif pos < 30:
            self.diagram.convex_comb(
                p.x, 1 - (pos - 20) / 10.0, self.point().x, self.p2().x
            )
            self.diagram.samev(p.y, self.p2().y)
        else:
            self.diagram.samev(p.x, self.point().x)
            self.diagram.convex_comb(
                p.y, 1 - (pos - 30) / 10, self.point().y, self.p2().y
            )
        return p

    def port_normal(self, pos: float):
        if pos < 0 or pos > 40:
            raise Exception("out of bounds: " + str(pos))
        if pos < 10:
            return (0, -1)
        if pos < 20:
            return (1, 0)
        if pos < 30:
            return (0, 1)
        return (-1, 0)


class Rectangle(Thing):
    def __init__(self, d: "Diagram", line_width: float, color: str, name: str = ""):
        super().__init__(d, name)
        assert isinstance(line_width, (float, int))
        self.name = name
        self._line_width = line_width
        self._color = color

        d.add_object(self)

    def to_svg(self, env) -> Tag:
        return rect(
            env(self.p1().x),
            env(self.p1().y),
            env(self.p2().x) - env(self.p1().x),
            env(self.p2().y) - env(self.p1().y),
            color=self._color,
            line_width=self._line_width,
        )


class Ellipse(Thing):
    def __init__(self, d: "Diagram", line_width: float, color: str, name: str = ""):
        super().__init__(d, name)
        assert isinstance(line_width, (float, int))
        self.name = name
        self._line_width = line_width
        self._color = color

        d.add_object(self)

    def port(self, degree):
        p = self.diagram.point()
        rad = -(degree - 90) / 360 * math.pi * 2
        self.diagram.add_constraint(
            "x",
            p.x,
            Relation.EQ,
            add(
                mul(math.cos(rad) * 0.5, self.width()),
                mul(0.5, add(self.p1().x, self.p2().x)),
            ),
        )
        self.diagram.add_constraint(
            "y",
            p.y,
            Relation.EQ,
            add(
                mul(-math.sin(rad) * 0.5, self.height()),
                mul(0.5, add(self.p1().y, self.p2().y)),
            ),
        )
        return p

    def port_normal(self, degree):
        rad = -(degree - 90) / 360 * math.pi * 2
        nx = math.cos(rad)  # /env(self.p1().width())
        ny = -math.sin(rad)  # /env(self.p1().height())
        return (nx, ny)

    def to_svg(self, env) -> Tag:
        return ellipse(
            env(self.p1().x),
            env(self.p1().y),
            env(self.p2().x) - env(self.p1().x),
            env(self.p2().y) - env(self.p1().y),
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

        d.add_object(self)

    def to_svg(self, env) -> Tag:
        w = env(self.width())
        h = env(self.height())
        x = env(self.point().x)
        y = env(self.point().y)

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
    def __init__(self, d: "Diagram", txt: str, sz: float, name: str = ""):
        super().__init__(d, name)
        self.font = Font("Inconsolata", sz, "")
        self.text = txt

        width = text_width(self.font, txt)

        self.diagram.add_constraint("w", [(1, self.width())], Relation.EQ, width)
        self.diagram.add_constraint("h", [(1, self.height())], Relation.EQ, sz)

        d.add_object(self)

    def to_svg(self, env) -> Tag:
        return translate(
            env(self.point().x),
            env(self.point().y),
            [text(0, self.font.font_size, self.text, self.font)],
        )


class DTextLines(Thing):
    def __init__(
        self, d: "Diagram", txt: List[str], font: Font, border: float, name: str = ""
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

        d.add_object(self)

    def to_svg(self, env):
        x = env(self.point().x)
        y = env(self.point().y)
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


def connect(
    t1: Thing, p1: float, t2: Thing, p2: float, d=100, n=0, connector2=">"
) -> CLine:
    start_point = t1.port(p1)
    end_point = t2.port(p2)
    normal1 = t1.port_normal(p1)
    normal2 = t2.port_normal(p2)

    l = [start_point]
    for i in range(n):
        l.append(d.point("p-" + str(i)))
    l.append(end_point)
    return CLine(
        t1.diagram,
        l,
        normal1,
        normal2,
        connector2=connector2,
        name=f"connect{t1.name}-{t2.name}",
    )


# vector length


def vlen(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def vnormalize(x: float, y: float, d: float) -> tuple[float, float]:
    le = vlen(x, y)
    return (x * d / le, y * d / le)


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

        # the variables
        self.vars: List[DVar] = []

        self.width = w
        self.height = h
        # the border around things
        self.border = mm(10)

        self.default_font = Inc12

        self.debug = debug
        self.counter = 0
        self.zero_var = self.get_var("zero", 0, 0)

    def newid(self, s: Optional[str] = None) -> str:
        self.counter = self.counter + 1
        return (s or "") + "-" + str(self.counter)

    def add_constraint(self, name: str, t1: Any, op: Relation, t2: Any) -> None:
        term1 = as_term(t1)
        term2 = as_term(t2)
        term = sub(term1, term2)
        sums = term.summands
        const = -term.const
        var_ids = set()
        for _, var in sums:
            if var.id in var_ids:
                raise Exception("duplicate variable in constraint: " + repr(var))
            var_ids.add(var.id)
        c = Constraint(name, sums, op, const)
        self.constraints.append(c)

    def same(self, p1: Point, p2: Point):
        self.add_constraint("SAME", p1.x, Relation.EQ, p2.x)
        self.add_constraint("SAME", p1.y, Relation.EQ, p2.y)

    def samev(self, v1: DVar, v2: DVar):
        self.add_constraint("SAMEV", v1, Relation.EQ, v2)

    def diffv(self, v1: DVar, v2: DVar, diff: float):
        self.add_constraint("DIFFV", v1, Relation.EQ, add(v2, diff))

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
                sum1.append((-1, obj.x))
            elif isinstance(obj, Marker):
                sum1.append((-1, obj.point().x))
            elif isinstance(obj, Thing):
                sum1.append((-1, obj.point().x))
                sum1.append((-1, obj.width()))
            else:
                raise Exception("unknown: " + repr(obj))
            self.add_constraint("left1", sum1, Relation.GE, c)

        for kobj in l2:
            obj = kobj
            sum2: List[tuple[float, DVar]] = [
                (-1, varr),
                (1, obj.x if isinstance(obj, Point) else obj.point().x),
            ]
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
                sum1.append((-1, obj.y))
            elif isinstance(obj, Marker):
                sum1.append((-1, obj.point().y))
            elif isinstance(obj, Thing):
                sum1.append((-1, obj.point().y))
                sum1.append((-1, obj.height()))
            else:
                raise Exception("unknown:" + repr(obj))
            self.add_constraint("left1", sum1, Relation.GE, c)

        for kobj in l2:
            obj = kobj
            sum2: List[tuple[float, DVar]] = [(-1, varr)]
            if isinstance(obj, Point):
                sum2.append((1, obj.y))
            elif isinstance(obj, Marker):
                sum2.append((1, obj.point().y))
            elif isinstance(obj, Thing):
                sum2.append((1, obj.point().y))
            else:
                raise Exception("unknown")
            self.add_constraint("left2", sum2, Relation.GE, 0)

    def column(self, dist: float, objs: list[Base], align: float = 0) -> None:
        l = flatten(objs)
        if len(l) == 0:
            return
        self.some_align(l, align)

        obj0 = l[0]

        for kobj in l[1:]:
            obj = kobj
            x = add(dist, obj0.point().y)
            if isinstance(obj0, Thing):
                x = add(x, obj0.height())
            self.add_constraint("y dist", x, Relation.EQ, obj.point().y)
            obj0 = obj

    def lalign(self, l: List[Base]) -> None:
        if len(l) <= 1:
            return
        var0 = l[0].point().x
        for q in l[1:]:
            var1 = q.point().x
            self.add_constraint("lalign", [(1, var0), (-1, var1)], Relation.EQ, 0)

    def some_align(self, l: List[Base], fac: float) -> None:
        if len(l) <= 1:
            return

        obj0 = l[0]
        if isinstance(obj0, Thing):
            l0 = [(float(1), obj0.point().x), (fac, obj0.width())]
        else:
            l0 = [(float(1), obj0.point().x)]

        for q in l[1:]:
            obj1 = q
            if isinstance(obj1, Thing):
                l1 = [(-1.0, obj1.point().x), (-fac, obj1.width())]
            else:
                l1 = [(float(-1), obj1.point().x)]

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
            self.add_constraint("posx", [(1, obj.x)], Relation.EQ, x)
        if not y is None:
            self.add_constraint("posy", [(1, obj.y)], Relation.EQ, y)

    def get_var(
        self,
        name: str,
        lbound: Optional[float],
        ubound: Optional[float],
        objective: float = 1,
    ) -> DVar:
        return DVar(self, name, lbound, ubound, objective)

    def def_var(self, term):
        term = as_term(term)
        v = self.get_var("dummy", None, None, 0)
        self.add_constraint("dummy", v, Relation.EQ, term)
        return v

    def add_weight(self, term, w):
        term = as_term(term)
        v = self.get_var("dummy", None, None, w)
        self.add_constraint("dummy", v, Relation.EQ, term)

    def point(
        self, x: Optional[float] = None, y: Optional[float] = None, name: str = ""
    ) -> Point:
        assert isinstance(name, str)
        xvar = self.get_var(name + "-x", 0, None)
        yvar = self.get_var(name + "-y", 0, None)

        point = Point(xvar, yvar)
        if not x is None:
            self.add_constraint("FIX", [(1, point.x)], Relation.EQ, x)
        if not y is None:
            self.add_constraint("FIX", [(1, point.y)], Relation.EQ, y)
        return point

    def marker(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        visible: Optional[bool] = True,
        name: str = "",
    ):
        m = Marker(self, visible, name)
        return m

    def real_point(
        self, name: str, x: Optional[float] = None, y: Optional[float] = None
    ) -> Point:
        point = Point(self)
        self.add_constraint("real", [(1, point.x)], Relation.GE, 0)
        self.add_constraint("real", [(1, point.y)], Relation.GE, 0)

        if not x is None:
            self.add_constraint("FIX", [(1, point.x)], Relation.EQ, x)
        if not y is None:
            self.add_constraint("FIX", [(1, point.y)], Relation.EQ, y)
        return point

    def bound(self, var: DVar, l: List[DVar], up: bool, dist: float):
        clamper = self.get_var("clamper", 0, None, 1)
        if up:
            for v in l:
                self.add_constraint("bounding-up", var, Relation.GE, add(v, dist))
                self.add_constraint("clamp", clamper, Relation.GE, add(var, mul(-1, v)))
        else:
            for v in l:
                self.add_constraint("bounding-down", var, Relation.LE, add(v, -dist))
                self.add_constraint("clamp", clamper, Relation.GE, add(v, mul(-1, var)))

    def topb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            if isinstance(obj, Point):
                l.append(obj.y)
            else:
                l.append(obj.point().y)
        self.bound(var, l, False, dist)

    def leftb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            if isinstance(obj, Point):
                l.append(obj.x)
            else:
                l.append(obj.point().x)
        self.bound(var, l, False, dist)

    def bottomb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            if isinstance(obj, Thing):
                l.append(obj.p2().y)
            elif isinstance(obj, Point):
                l.append(obj.y)
            else:
                l.append(obj.point().y)
        self.bound(var, l, True, dist)

    def rightb(self, var: DVar, objs, dist: float):
        l = []
        for obj in flatten(objs):
            if isinstance(obj, Thing):
                l.append(obj.p2().x)
            elif isinstance(obj, Point):
                l.append(obj.x)
            else:
                l.append(obj.point().x)
        self.bound(var, l, True, dist)

    def around(self, p1: Point, p2: Point, objs: Any, bounds):
        if isinstance(bounds, (float, int)):
            (t, l, b, r) = (bounds, bounds, bounds, bounds)
        else:
            (l, r, t, b) = bounds
        self.topb(p1.y, objs, t)
        self.leftb(p1.x, objs, l)
        self.bottomb(p2.y, objs, b)
        self.rightb(p2.x, objs, r)

    def varcentered(self, vo1: DVar, vo2: DVar, vi1: DVar, vi2: DVar):
        self.add_constraint("centered", [(1, vi1), (-1, vo1)], Relation.GE, 0)
        self.add_constraint("centered", [(-1, vi2), (1, vo2)], Relation.GE, 0)
        self.add_constraint(
            "centered-same", [(1, vi1), (-1, vo1), (1, vi2), (-1, vo2)], Relation.EQ, 0
        )

    def centered(self, p1: Point, p2: Point, t: Thing):
        self.varcentered(p1.x, p2.x, t.p1().x, t.p2().x)
        self.varcentered(p1.y, p2.y, t.p1().y, t.p2().y)

    def add_object(self, o):
        self.object_list.append(o)

    def text(self, txt: str, sz: float, name: str = "") -> Thing:
        assert isinstance(name, str)
        return DText(self, txt, sz, name)

    def textl(self, txt: List[str], sz: float, name: str = "") -> Thing:
        font = Font("Inconsolata", sz, "")
        tb = mm(2)
        t = DTextLines(self, list(txt), font, tb, name)
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
            cons_list.append((self.newid(cons.name), l, low, up))

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
        assert isinstance(values, dict)
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
