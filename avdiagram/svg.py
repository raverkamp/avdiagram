from collections import namedtuple
from typing import List, Optional, Union, Any, Tuple, Callable, NamedTuple

"""a module to generate SVG"""


def merge_dicts(d1: dict[str, Any], d2: dict[str, Any]) -> dict[str, Any]:
    res = {}
    d1 = d1 or {}
    d2 = d2 or {}
    for k in d1:
        res[k] = d1[k]
    for k in d2:
        res[k] = d2[k]
    return res


SVG_HEADER = """<?xml version="1.0"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
         "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">"""


class Font(NamedTuple):
    font_family: str
    font_size: float
    font_weight: str


def points(l: list[Tuple[Any, Any]]) -> str:
    l2 = map(lambda p: repr(p[0]) + ", " + repr(p[1]), l)
    return " ".join(l2)


class Raw(NamedTuple):
    text: str


class Tag(NamedTuple):
    name: str
    ats: dict[str, str]
    content: Optional["Stuff"]


empty_tag = Tag("g", {}, None)

Stuff = Union[str, List["Stuff"], Tag, Raw]

defs = Raw(
    """<defs>
    <marker id="Triangle"
      viewBox="0 0 20 20" refX="20" refY="10"
      markerUnits="userSpaceOnUse"
      markerWidth="20" markerHeight="20"
      orient="auto">
      <line x1="0" y1="0" x2 ="20" y2 ="10" style="stroke:#000; stroke-width:2"/>
      <line x1="0" y1="20" x2 ="20" y2 ="10" style="stroke:#000; stroke-width:2"/>
      <!--path d="M 0 0 L 20 10 L 0 20 z" /-->
    </marker>
    <style type="text/css">
      @import url('https://fonts.googleapis.com/css?family=Inconsolata');
      text {
        font-sizex: 10pt;
        font-family: "Inconsolata";
      }
    </style>
  </defs>"""
)
# "


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    line_color: str = "#000000",
    line_width: float = 1,
) -> Tag:
    return Tag(
        "polygon",
        {
            "fill": color,
            "stroke": line_color,
            "stroke-width": str(line_width),
            "points": points([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]),
        },
        None,
    )


def text(x: float, y: float, txt: str, font: Optional[Font] = None) -> Tag:
    attrs = {"x": str(x), "y": str(y)}
    if font:
        if font.font_family:
            attrs["font-famliy"] = str(font.font_family)
        if font.font_size:
            attrs["font-size"] = str(font.font_size)
        if font.font_weight:
            attrs["font-weight"] = str(font.font_weight)
    return Tag("text", attrs, txt)


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#000",
    width: float = 1,
    marker_end: Optional[str] = None,
) -> Tag:
    attrs = {"x1": str(x1), "x2": str(x2), "y1": str(y1), "y2": str(y2)}
    sty = "stroke:" + str(color) + "; stroke-width:" + str(width)
    attrs["style"] = sty
    if marker_end:
        attrs["marker-end"] = marker_end
    return Tag("line", attrs, None)


def path(
    pa: str, color: str = "#000", width: float = 1, marker_end: Optional[str] = None
) -> Tag:
    assert isinstance(pa, str)
    attrs = {"d": pa}
    sty = "fill:none;stroke:" + str(color) + "; stroke-width:" + str(width)
    attrs["style"] = sty
    if marker_end:
        attrs["marker-end"] = marker_end
    return Tag("path", attrs, None)


def translate(x: float, y: float, what: Stuff) -> Tag:
    return Tag("g", {"transform": "translate(" + str(x) + "," + str(y) + ")"}, what)


def polyline(pts: list[Tuple[float, float]], style: Optional[str] = None) -> Tag:
    style = style or "fill:none;stroke:rgb(0,0,0);stroke-width:2"
    return Tag("polyline", {"points": points(pts), "style": style}, None)


def renderattr(l: list[str], a: str) -> None:
    l.append('"' + a + '" ')


def rendertag(l: list[str], tag: Tag) -> None:
    l.append("<" + tag.name + " ")
    for k in tag.ats.keys():
        v = tag.ats[k]
        l.append(k)
        l.append("=")
        renderattr(l, v)
    if tag.content is None:
        l.append("/>\n")
    else:
        l.append(">\n")
        render(l, tag.content)
        l.append("</" + tag.name + ">\n")


def render(l: list[str], x: Stuff) -> None:
    if isinstance(x, str):
        l.append(x)
    elif isinstance(x, Raw):
        l.append(x.text)
    elif isinstance(x, list):
        for y in x:
            render(l, y)
    elif isinstance(x, Tag):
        if x != empty_tag:
            rendertag(l, x)
    else:
        raise Exception("unknown content: " + repr(x))


def render_file(name: str, w: float, h: float, things: Stuff) -> None:
    with open(name, "w", encoding="UTF-8") as f:
        f.write(SVG_HEADER + "\n")
        svg = Tag(
            "svg",
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "xmlns:xlink": "http://www.w3.org/1999/xlink",
                "width": str(w),
                "height": str(h),
            },
            [defs, things],
        )
        l: List[str] = []
        render(l, svg)
        f.write("".join(l))
