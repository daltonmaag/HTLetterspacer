import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import fontTools.misc.arrayTools as arrayTools
import fontTools.misc.bezierTools as bezierTools
import fontTools.pens.basePen as basePen
import numpy as np
from fontTools.misc.transform import Identity, Transform
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPointPen
from ufoLib2.objects import Font, Glyph, Layer
from ufoLib2.objects.misc import BoundingBox
from ufoLib2.objects.point import Point
from ufoLib2.typing import GlyphSet

LOGGER = logging.Logger(__name__)

GLYPHS_LEFT_METRICS_KEY = "com.schriftgestaltung.Glyphs.glyph.leftMetricsKey"
GLYPHS_RIGHT_METRICS_KEY = "com.schriftgestaltung.Glyphs.glyph.rightMetricsKey"


@dataclass
class NSPoint:
    __slots__ = "x", "y"
    x: float
    y: float


def space_main(
    layer: Glyph,
    reference_layer_bounds: BoundingBox,
    glyphset: Union[Font, Layer],
    angle: float,
    compute_lsb: bool,
    compute_rsb: bool,
    factor: float,
    param_area: int,
    param_depth: int,
    param_freq: int,
    param_over: int,
    tabular_width: Optional[int],
    upm: int,
    xheight: int,
) -> None:
    if not layer.contours and not layer.components:
        LOGGER.warning("No paths in glyph %s.", layer.name)
        return

    if layer.components:
        dpen = DecomposingRecordingPen(glyphset)
        layer.draw(dpen)
        layer_decomposed = layer.copy()
        layer_decomposed.components.clear()
        dpen.replay(layer_decomposed.getPen())
        layer_measure = layer_decomposed
    else:
        layer_measure = layer

    if tabular_width is None and (".tosf" in layer.name or ".tf" in layer.name):
        layer_width = layer.width
        assert layer_width is not None
        tabular_width = round(layer_width)

    new_left, new_right, new_width = calculate_spacing(
        layer_measure,
        reference_layer_bounds,
        angle,
        compute_lsb,
        compute_rsb,
        factor,
        param_area,
        param_depth,
        param_freq,
        param_over,
        tabular_width,
        upm,
        xheight,
    )
    set_sidebearings(
        layer,
        glyphset,
        new_left,
        new_right,
        new_width,
        angle,
        xheight,
    )


def calculate_spacing(
    layer: Glyph,
    reference_layer_bounds: BoundingBox,
    angle: float,
    compute_lsb: bool,
    compute_rsb: bool,
    factor: float,
    param_area: int,
    param_depth: int,
    param_freq: int,
    param_over: int,
    tabular_width: Optional[int],
    upm: int,
    xheight: int,
) -> tuple[int, int, int]:
    # TODO: compute lsb/rsb separately?

    # get reference glyph maximum points
    overshoot = xheight * param_over / 100

    # store min and max y
    min_yref = reference_layer_bounds.yMin - overshoot
    max_yref = reference_layer_bounds.yMax + overshoot

    # bounds
    margins_left_full, margins_right_full = margin_list(layer, param_freq)
    if angle:
        margins_left_full = deslant(margins_left_full, angle, xheight)
        margins_right_full = deslant(margins_right_full, angle, xheight)

    margins_left = [p for p in margins_left_full if min_yref <= p.y <= max_yref]
    margins_right = [p for p in margins_right_full if min_yref <= p.y <= max_yref]

    # create a closed polygon
    polygon_left, polygon_right = process_margins(
        margins_left,
        margins_right,
        xheight,
        min_yref,
        max_yref,
        param_depth,
        param_freq,
    )

    # get extreme points deitalized
    layer_bounds = layer.getBounds()
    assert layer_bounds is not None
    extreme_left_full, extreme_right_full = max_points(
        margins_left_full + margins_right_full, layer_bounds.yMin, layer_bounds.yMax
    )
    # get zone extreme points
    extreme_left, extreme_right = max_points(
        margins_left + margins_right, min_yref, max_yref
    )

    # dif between extremes full and zone
    distance_left = math.ceil(extreme_left.x - extreme_left_full.x)
    distance_right = math.ceil(extreme_right_full.x - extreme_right.x)

    # set new sidebearings
    new_left: int = math.ceil(
        0
        - distance_left
        + calculate_sidebearing_value(
            factor,
            max_yref,
            min_yref,
            param_area,
            polygon_left,
            upm,
            xheight,
        )
    )
    new_right: int = math.ceil(
        0
        - distance_right
        + calculate_sidebearing_value(
            factor,
            max_yref,
            min_yref,
            param_area,
            polygon_right,
            upm,
            xheight,
        )
    )
    new_width: int = 0

    if tabular_width is not None:
        width_shape = extreme_right_full.x - extreme_left_full.x
        width_actual = width_shape + new_left + new_right
        width_diff = round((tabular_width - width_actual) / 2)

        new_left += width_diff
        new_right += width_diff
        new_width = tabular_width

        LOGGER.warning(
            "%s is tabular and adjusted at width = %s", layer.name, str(tabular_width)
        )
    # TODO: Decide earlier whether to compute lsb/rsb.
    else:
        if not compute_lsb:
            margin_left = layer.getLeftMargin()
            assert margin_left is not None
            new_left = round(margin_left)
        if not compute_rsb:
            margin_right = layer.getRightMargin()
            assert margin_right is not None
            new_right = round(margin_right)

    return new_left, new_right, new_width


def set_sidebearings(
    layer: Glyph,
    glyphset: Union[Font, Layer],
    new_left: float,
    new_right: float,
    width: float,
    angle: float,
    xheight: float,
) -> None:
    if angle:
        new_left, new_right = deslant_sidebearings(
            layer, glyphset, new_left, new_right, angle, xheight
        )
    set_left_margin_rounded(layer, new_left, glyphset)
    set_right_margin_rounded(layer, new_right, glyphset)

    # adjusts the tabular miscalculation
    if width:
        layer.width = width

    # TODO: Handle this outside the core.
    if "com.schriftgestaltung.Glyphs.originalWidth" in layer.lib:
        layer.lib["com.schriftgestaltung.Glyphs.originalWidth"] = layer.width
        layer.width = 0


def set_left_margin_rounded(
    glyph: Glyph, value: float, layer: Optional[GlyphSet] = None
) -> None:
    """Sets the the rounded space in font units from the point of origin to the
    left side of the glyph.

    Args:
        value: The desired left margin in font units.
        layer: The layer of the glyph to look up components, if any. Not needed for
            pure-contour glyphs.
    """
    bounds = glyph.getBounds(layer)
    if bounds is None:
        return None
    diff = round(value - bounds.xMin)
    if diff:
        glyph.width += diff
        glyph.move((diff, 0))


def set_right_margin_rounded(
    glyph: Glyph, value: float, layer: Optional[GlyphSet] = None
) -> None:
    """Sets the the rounded space in font units from the glyph's advance width to
    the right side of the glyph.

    Args:
        value: The desired right margin in font units.
        layer: The layer of the glyph to look up components, if any. Not needed for
            pure-contour glyphs.
    """
    bounds = glyph.getBounds(layer)
    if bounds is None:
        return None
    glyph.width = round(bounds.xMax + value)


def calculate_sidebearing_value(
    factor: float,
    max_yref: float,
    min_yref: float,
    param_area: int,
    polygon: list[NSPoint],
    upm: int,
    xheight: int,
) -> float:
    amplitude_y = max_yref - min_yref

    # recalculates area based on UPM
    area_upm = param_area * ((upm / 1000) ** 2)

    # calculates proportional area
    white_area = area_upm * factor * 100

    prop_area = (amplitude_y * white_area) / xheight
    valor = prop_area - area(polygon)
    return valor / amplitude_y


def deslant(margin: list[NSPoint], angle: float, xheight: int) -> list[NSPoint]:
    """De-slant a list of points (contour) at angle with the point of origin at half the xheight."""
    mline = xheight / 2
    return [
        NSPoint(p.x - (p.y - mline) * math.tan(math.radians(angle)), p.y)
        for p in margin
    ]


def close_open_counters(
    margin: list[NSPoint], extreme: NSPoint, max_yref: float, min_yref: float
) -> list[NSPoint]:
    """close counterforms, creating a polygon"""
    initPoint = NSPoint(extreme.x, min_yref)
    endPoint = NSPoint(extreme.x, max_yref)
    margin.insert(0, initPoint)
    margin.append(endPoint)
    return margin


def max_points(
    points: list[NSPoint], min_y: float, max_y: float
) -> tuple[NSPoint, NSPoint]:
    right = -10000
    righty = None
    left = 10000
    lefty = None
    for p in points:
        if p.y >= min_y and p.y <= max_y:
            if p.x > right:
                right = p.x
                righty = p.y
            if p.x < left:
                left = p.x
                lefty = p.y
    assert lefty is not None
    assert righty is not None
    return NSPoint(left, lefty), NSPoint(right, righty)


def set_depth(
    margins_left: list[NSPoint],
    margins_right: list[NSPoint],
    extreme_left: NSPoint,
    extreme_right: NSPoint,
    xheight: int,
    min_yref: float,
    max_yref: float,
    param_depth: int,
    param_freq: int,
) -> tuple[list[NSPoint], list[NSPoint]]:
    """process lists with depth, proportional to xheight"""
    depth = xheight * param_depth / 100
    max_depth = extreme_left.x + depth
    min_depth = extreme_right.x - depth
    margins_left = [NSPoint(min(p.x, max_depth), p.y) for p in margins_left]
    margins_right = [NSPoint(max(p.x, min_depth), p.y) for p in margins_right]

    # add all the points at maximum depth if glyph is shorter than overshoot
    y = margins_left[0].y - param_freq
    while y > min_yref:
        margins_left.insert(0, NSPoint(max_depth, y))
        margins_right.insert(0, NSPoint(min_depth, y))
        y -= param_freq

    y = margins_left[-1].y + param_freq
    while y < max_yref:
        margins_left.append(NSPoint(max_depth, y))
        margins_right.append(NSPoint(min_depth, y))
        y += param_freq

    return margins_left, margins_right


def diagonize(
    margins_left: list[NSPoint],
    margins_right: list[NSPoint],
    param_freq: int,
) -> tuple[list[NSPoint], list[NSPoint]]:
    """close counters at 45 degrees"""
    # TODO: Use https://github.com/huertatipografica/HTLetterspacer/issues/45
    total = len(margins_left) - 1

    frequency = param_freq * 1.5
    for index in range(total):
        # left
        actual_point = margins_left[index]
        next_point = margins_left[index + 1]
        diff = next_point.y - actual_point.y
        if next_point.x > (actual_point.x + diff) and next_point.y > actual_point.y:
            margins_left[index + 1].x = actual_point.x + diff
        # right
        actual_point = margins_right[index]
        next_point = margins_right[index + 1]
        # if nextPoint.x < (actualPoint.x - valueFreq) and nextPoint.y > actualPoint.y:
        if next_point.x < (actual_point.x - diff) and next_point.y > actual_point.y:
            margins_right[index + 1].x = actual_point.x - diff

        # left
        actual_point = margins_left[total - index]
        next_point = margins_left[total - index - 1]
        diff = actual_point.y - next_point.y
        if (
            next_point.x > (actual_point.x + frequency)
            and next_point.y < actual_point.y
        ):
            margins_left[total - index - 1].x = actual_point.x + diff
        # right
        actual_point = margins_right[total - index]
        next_point = margins_right[total - index - 1]
        if next_point.x < (actual_point.x - diff) and next_point.y < actual_point.y:
            margins_right[total - index - 1].x = actual_point.x - diff

    return margins_left, margins_right


def process_margins(
    margins_left: list[NSPoint],
    margins_right: list[NSPoint],
    xheight: int,
    min_yref: float,
    max_yref: float,
    param_depth: int,
    param_freq: int,
) -> tuple[list[NSPoint], list[NSPoint]]:
    # get extremes
    extreme_left, extreme_right = max_points(
        margins_left + margins_right, min_yref, max_yref
    )

    # set depth
    margins_left, margins_right = set_depth(
        margins_left,
        margins_right,
        extreme_left,
        extreme_right,
        xheight,
        min_yref,
        max_yref,
        param_depth,
        param_freq,
    )

    # close open counterforms at 45 degrees
    margins_left, margins_right = diagonize(margins_left, margins_right, param_freq)
    margins_left = close_open_counters(margins_left, extreme_left, max_yref, min_yref)
    margins_right = close_open_counters(
        margins_right, extreme_right, max_yref, min_yref
    )

    return margins_left, margins_right


def deslant_sidebearings(
    layer: Glyph,
    glyphset: Union[Font, Layer],
    l: float,
    r: float,
    a: float,
    xheight: float,
) -> tuple[int, int]:
    bounds = layer.getBounds(glyphset)
    assert bounds is not None
    left, _, _, _ = bounds
    m = skew_matrix((-a, 0), offset=(left, xheight / 2))
    backslant = Glyph()
    backslant.width = layer.width
    layer.drawPoints(TransformPointPen(backslant.getPointPen(), m))
    backslant.setLeftMargin(l, glyphset)
    backslant.setRightMargin(r, glyphset)

    boundsback = backslant.getBounds(glyphset)
    assert boundsback is not None
    left, _, _, _ = boundsback
    mf = skew_matrix((a, 0), offset=(left, xheight / 2))
    forwardslant = Glyph()
    forwardslant.width = backslant.width
    backslant.drawPoints(TransformPointPen(forwardslant.getPointPen(), mf))

    left_margin = forwardslant.getLeftMargin(glyphset)
    assert left_margin is not None
    right_margin = forwardslant.getRightMargin(glyphset)
    assert right_margin is not None

    return round(left_margin), round(right_margin)


def skew_matrix(
    angle: tuple[float, float], offset: tuple[float, float] = (0, 0)
) -> Transform:
    dx, dy = offset
    x, y = angle
    x, y = math.radians(x), math.radians(y)
    sT = Identity.translate(dx, dy)
    sT = sT.skew(x, y)
    sT = sT.translate(-dx, -dy)
    return sT


# point list area
def area(points: list[NSPoint]) -> float:
    s = 0
    for ii in np.arange(len(points)) - 1:
        s = s + (points[ii].x * points[ii + 1].y - points[ii + 1].x * points[ii].y)
    return abs(s) * 0.5


# get margins in Glyphs
def get_margins(
    layer: Glyph, measurement_line: tuple[float, float, float, float]
) -> tuple[Optional[float], Optional[float]]:
    # TODO: intersection returns a reversed list?
    result = sorted(intersections(layer, measurement_line))
    if not result:
        return (None, None)

    # Only take the outermost hits.
    left = 0
    right = -1
    return (result[left][0], result[right][0])


# a list of margins
def margin_list(layer: Glyph, param_freq: int) -> tuple[list[Any], list[Any]]:
    layer_bounds = layer.getBounds()
    assert layer_bounds is not None
    y = layer_bounds.yMin
    list_left = []
    list_right = []
    # works over glyph copy
    while y <= layer_bounds.yMax:
        measurement_line = layer_bounds.xMin, y, layer_bounds.xMax, y
        lpos, rpos = get_margins(layer, measurement_line)
        if lpos is not None:
            list_left.append(NSPoint(lpos, y))
        if rpos is not None:
            list_right.append(NSPoint(rpos, y))
        y += param_freq
    return list_left, list_right


####


def segments(contour: list[Point]) -> list[list[Point]]:
    if not contour:
        return []
    segments = [[]]
    last_was_off_curve = False
    for point in contour:
        segments[-1].append(point)
        if point.segmentType is not None:
            segments.append([])
        last_was_off_curve = point.segmentType is None
    if len(segments[-1]) == 0:
        del segments[-1]
    if last_was_off_curve:
        lastSegment = segments[-1]
        segment = segments.pop(0)
        lastSegment.extend(segment)
    elif segments[0][-1].segmentType != "move":
        segment = segments.pop(0)
        segments.append(segment)
    return segments


def intersections(
    layer: Glyph, measurement_line: tuple[float, float, float, float]
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    x1, y1, x2, y2 = measurement_line
    for contour in layer.contours:
        for segment_pair in arrayTools.pairwise(segments(contour.points)):
            last, curr = segment_pair
            curr_type = curr[-1].type
            if curr_type == "line":
                i = line_intersection(x1, y1, x2, y2, last[-1], curr[-1])
                if i is not None:
                    intersections.append(i)
            elif curr_type == "curve":
                i = curve_intersections(
                    x1, y1, x2, y2, last[-1], curr[0], curr[1], curr[2]
                )
                if i:
                    intersections.extend(i)
            elif curr_type == "qcurve":
                i = qcurve_intersections(x1, y1, x2, y2, last[-1], *curr)
                if i:
                    intersections.extend(i)
            else:
                raise ValueError(f"Cannot deal with segment type {curr_type}")
    return intersections


def curve_intersections(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> list[tuple[float, float]]:
    """
    Computes intersection between a line and a cubic curve.
    Adapted from: https://www.particleincell.com/2013/cubic-line-intersection/
    Takes four scalars describing line parameters and four points describing
    curve.
    Returns a list of intersections in Tuples of the format (coordinate_x,
    coordinate_y, term, subsegment_index == 0). The subsegment_index is only
    used for quadratic curves.
    """
    bx, by = x1 - x2, y2 - y1
    m = x1 * (y1 - y2) + y1 * (x2 - x1)
    a, b, c, d = bezierTools.calcCubicParameters(
        (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
    )
    pc0 = by * a[0] + bx * a[1]
    pc1 = by * b[0] + bx * b[1]
    pc2 = by * c[0] + bx * c[1]
    pc3 = by * d[0] + bx * d[1] + m
    r = bezierTools.solveCubic(pc0, pc1, pc2, pc3)
    sol = []
    for t in r:
        if t < 0 or t > 1:
            continue
        s0 = ((a[0] * t + b[0]) * t + c[0]) * t + d[0]
        s1 = ((a[1] * t + b[1]) * t + c[1]) * t + d[1]
        if (x2 - x1) != 0:
            s = (s0 - x1) / (x2 - x1)
        else:
            s = (s1 - y1) / (y2 - y1)
        if s < 0 or s > 1:
            continue
        sol.append((s0, s1))
    return sol


def qcurve_intersections(
    x1: float, y1: float, x2: float, y2: float, *pts: Point
) -> list[tuple[float, float]]:
    """
    Computes intersection between a quadratic spline and a line segment.
    Adapted from curveIntersections(). Takes four scalars describing line and an
    Iterable of points describing a quadratic curve, including the first (==
    anchor) point. Quadatric curves are special in that they can consist of
    implied on-curve points, which is why this function returns a
    `subsegment_index` to associate a `t` with the correct subsegment.
    Returns a list of intersections in Tuples of the format (coordinate_x,
    coordinate_y, term, subsegment_index).
    """

    sol = []
    points = [(p.x, p.y) for p in pts]

    nx, ny = x1 - x2, y2 - y1
    m = x1 * (y1 - y2) + y1 * (x2 - x1)
    # Decompose a segment with potentially implied on-curve points into subsegments.
    # p1 is the anchor, p2 the control handle, p3 the (implied) on-curve point in the
    # subsegment.
    p1 = points[0]
    # TODO: skia pathops also has a SegmentPenIterator
    # https://github.com/googlefonts/nanoemoji/blob/9adfff414b1ba32a816d722936421c52d4827d8a/src/nanoemoji/svg_path.py#L98-L101
    for p2, p3 in basePen.decomposeQuadraticSegment(points[1:]):
        (ax, ay), (bx, by), (cx, cy) = bezierTools.calcQuadraticParameters(p1, p2, p3)
        p1 = p3  # prepare for next turn

        pc0 = ny * ax + nx * ay
        pc1 = ny * bx + nx * by
        pc2 = ny * cx + nx * cy + m
        r = bezierTools.solveQuadratic(pc0, pc1, pc2)

        for t in r:
            if t < 0 or t > 1:
                continue
            sx = (ax * t + bx) * t + cx
            sy = (ay * t + by) * t + cy
            if (x2 - x1) != 0:
                s = (sx - x1) / (x2 - x1)
            else:
                s = (sy - y1) / (y2 - y1)
            if s < 0 or s > 1:
                continue
            sol.append((sx, sy))
    return sol


def line_intersection(
    x1: float, y1: float, x2: float, y2: float, p3: Point, p4: Point
) -> Optional[tuple[float, float]]:
    """
    Computes intersection between two lines.
    Adapted from Andre LaMothe, "Tricks of the Windows Game Programming Gurus".
    G. Bach, http://stackoverflow.com/a/1968345
    Takes four scalars describing line parameters and two points describing
    line.
    Returns a list of intersections in Tuples of the format (coordinate_x,
    coordinate_y, term, subsegment_index == 0). The subsegment_index is only
    used for quadratic curves.
    """
    x3, y3 = p3.x, p3.y
    x4, y4 = p4.x, p4.y
    Bx_Ax = x4 - x3
    By_Ay = y4 - y3
    Dx_Cx = x2 - x1
    Dy_Cy = y2 - y1
    determinant = -Dx_Cx * By_Ay + Bx_Ax * Dy_Cy
    if abs(determinant) < 1e-12:
        return None
    s = (-By_Ay * (x3 - x1) + Bx_Ax * (y3 - y1)) / determinant
    t = (Dx_Cx * (y3 - y1) - Dy_Cy * (x3 - x1)) / determinant
    if 0 <= s <= 1 and 0 <= t <= 1:
        return (x3 + (t * Bx_Ax), y3 + (t * By_Ay))
    return None
