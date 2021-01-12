import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import fontTools.misc.arrayTools as arrayTools
import fontTools.misc.bezierTools as bezierTools
import fontTools.pens.basePen as basePen
import numpy as np
from fontTools.misc.transform import Identity
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPointPen
from ufoLib2.objects import Font, Glyph, Layer
from ufoLib2.objects.point import Point

LOGGER = logging.Logger(__name__)

GLYPHS_LEFT_METRICS_KEY = "com.schriftgestaltung.Glyphs.glyph.leftMetricsKey"
GLYPHS_RIGHT_METRICS_KEY = "com.schriftgestaltung.Glyphs.glyph.rightMetricsKey"


@dataclass
class NSPoint:
    __slots__ = "x", "y"
    x: float
    y: float


def NSMakePoint(x: float, y: float) -> NSPoint:
    return NSPoint(x, y)


class HTLetterspacerLib:
    def __init__(self, upm, angle, xHeight, LSB, RSB, factor, width):
        self.param_area = 400  # white area in thousand units
        self.param_depth = 15  # depth in open counterforms, from extreme points.
        self.param_over = 0  # overshoot in spacing vertical range
        self.tab_version = False
        self.upm = upm
        self.angle = angle
        self.xheight = xHeight
        self.LSB = LSB
        self.RSB = RSB
        self.factor = factor
        self.width = width
        self.new_width = 0.0
        self.color = False  # mark color, False for no mark
        self.param_freq = 5  # frequency of vertical measuring. Higher values are faster but less accurate

    def set_space(self, layer: Glyph, reference_layer: Glyph) -> None:
        # get reference glyph maximum points
        overshoot = calculate_overshoot(self.xheight, self.param_over)

        # store min and max y
        reference_layer_bounds = reference_layer.getBounds()
        self.min_yref = reference_layer_bounds.yMin - overshoot
        self.max_yref = reference_layer_bounds.yMax + overshoot

        # bounds
        margins_left_full, margins_right_full = marginList(layer, self.param_freq)

        margins_left = [
            p for p in margins_left_full if self.min_yref <= p.y <= self.max_yref
        ]
        margins_right = [
            p for p in margins_right_full if self.min_yref <= p.y <= self.max_yref
        ]

        # create a closed polygon
        polygon_left, polygon_right = process_margins(
            margins_left,
            margins_right,
            self.angle,
            self.xheight,
            self.min_yref,
            self.max_yref,
            self.param_depth,
            self.param_freq,
        )
        margins_left = deslant(margins_left, self.angle, self.xheight)
        margins_right = deslant(margins_right, self.angle, self.xheight)

        margins_left_full = deslant(margins_left_full, self.angle, self.xheight)
        margins_right_full = deslant(margins_right_full, self.angle, self.xheight)

        # get extreme points deitalized
        layer_bounds = layer.getBounds()
        extreme_left_full, extreme_right_full = max_points(
            margins_left_full + margins_right_full, layer_bounds.yMin, layer_bounds.yMax
        )
        # get zone extreme points
        extreme_left, extreme_right = max_points(
            margins_left + margins_right, self.min_yref, self.max_yref
        )

        # dif between extremes full and zone
        distance_left = math.ceil(extreme_left.x - extreme_left_full.x)
        distance_right = math.ceil(extreme_right_full.x - extreme_right.x)

        # set new sidebearings
        self.new_left = math.ceil(
            0
            - distance_left
            + calculate_sidebearing_value(
                self.factor,
                self.max_yref,
                self.min_yref,
                self.param_area,
                polygon_left,
                self.upm,
                self.xheight,
            )
        )
        self.new_right = math.ceil(
            0
            - distance_right
            + calculate_sidebearing_value(
                self.factor,
                self.max_yref,
                self.min_yref,
                self.param_area,
                polygon_right,
                self.upm,
                self.xheight,
            )
        )

        # tabVersion
        if ".tosf" in layer.name or ".tf" in layer.name or self.tab_version:
            if self.width:
                layer_width = self.width
            else:
                layer_width = layer.width

            width_shape = extreme_right_full.x - extreme_left_full.x
            width_actual = width_shape + self.new_left + self.new_right
            width_diff = (layer_width - width_actual) / 2

            self.new_left += width_diff
            self.new_right += width_diff
            self.new_width = layer_width

            LOGGER.warning(
                "%s is tabular and adjusted at width = %s", layer.name, str(layer_width)
            )
        # end tabVersion

        # if there is a metric rule
        else:
            # TODO: coverage test and remove
            if layer.lib.get(GLYPHS_LEFT_METRICS_KEY) is not None or self.LSB == False:
                self.new_left = layer.getLeftMargin()
            if layer.lib.get(GLYPHS_RIGHT_METRICS_KEY) is not None or self.RSB == False:
                self.new_right = layer.getRightMargin()

    def spaceMain(
        self, layer: Glyph, reference_layer: Glyph, glyphset: Union[Font, Layer]
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

        if reference_layer.components:
            dpen = DecomposingRecordingPen(glyphset)
            reference_layer.draw(dpen)
            reference_layer_decomposed = reference_layer.copy()
            reference_layer_decomposed.components.clear()
            dpen.replay(reference_layer_decomposed.getPen())
            reference_layer_measure = reference_layer_decomposed
        else:
            reference_layer_measure = reference_layer

        self.set_space(layer_measure, reference_layer_measure)
        set_sidebearings(
            layer,
            glyphset,
            self.new_left,
            self.new_right,
            self.new_width,
            self.color,
            self.angle,
            self.xheight,
        )


#  Functions


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


def italic_on_off_point(p, make_italic: bool, angle, xHeight):
    mline = xHeight / 2
    cateto = -p.y + mline
    if not make_italic:
        cateto = -cateto
    xvar = -rectCateto(angle, cateto)
    return NSMakePoint(p.x + xvar, p.y)


def deslant(margin: list[NSPoint], angle: float, xHeight: int):
    return [italic_on_off_point(p, False, angle, xHeight) for p in margin]


def slant(margin: list[NSPoint], angle: float, xHeight: int):
    return [italic_on_off_point(p, True, angle, xHeight) for p in margin]


def close_open_counters(
    margin: list[NSPoint], extreme: NSPoint, max_yref: float, min_yref: float
) -> list[NSPoint]:
    """close counterforms, creating a polygon"""
    initPoint = NSMakePoint(extreme.x, min_yref)
    endPoint = NSMakePoint(extreme.x, max_yref)
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
    return NSMakePoint(left, lefty), NSMakePoint(right, righty)


def calculate_overshoot(xHeight: int, paramOver: int) -> float:
    return xHeight * paramOver / 100


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
    margins_left = [NSMakePoint(min(p.x, max_depth), p.y) for p in margins_left]
    margins_right = [NSMakePoint(max(p.x, min_depth), p.y) for p in margins_right]

    # add all the points at maximum depth if glyph is shorter than overshoot
    y = margins_left[0].y - param_freq
    while y > min_yref:
        margins_left.insert(0, NSMakePoint(max_depth, y))
        margins_right.insert(0, NSMakePoint(min_depth, y))
        y -= param_freq

    y = margins_left[-1].y + param_freq
    while y < max_yref:
        margins_left.append(NSMakePoint(max_depth, y))
        margins_right.append(NSMakePoint(min_depth, y))
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
    angle: float,
    xheight: int,
    min_yref: float,
    max_yref: float,
    param_depth: int,
    param_freq: int,
) -> tuple[list[NSPoint], list[NSPoint]]:
    # deSlant if is italic
    margins_left = deslant(margins_left, angle, xheight)
    margins_right = deslant(margins_right, angle, xheight)

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

    margins_left = slant(margins_left, angle, xheight)
    margins_right = slant(margins_right, angle, xheight)
    return margins_left, margins_right


def set_sidebearings(
    layer: Glyph,
    glyphset: Union[Font, Layer],
    new_left: float,
    new_right: float,
    width: float,
    color: Any,
    angle: float,
    xheight: float,
) -> None:
    if angle:
        set_sidebearings_slanted(layer, glyphset, new_left, new_right, angle, xheight)
    else:
        layer.setLeftMargin(new_left, glyphset)
        layer.setRightMargin(new_right, glyphset)

    # adjusts the tabular miscalculation
    if width:
        layer.width = width

    if color:
        layer.lib["public.markColor"] = color


def set_sidebearings_slanted(
    layer: Glyph,
    glyphset: Union[Font, Layer],
    l: float,
    r: float,
    a: float,
    xheight: float,
) -> None:
    # TODO: Handle this outside the core.
    original_width = (
        layer.lib.get("com.schriftgestaltung.Glyphs.originalWidth") or layer.width
    )

    bounds = layer.getBounds(glyphset)
    assert bounds is not None
    left, _, _, _ = bounds
    m = skew_matrix((-a, 0), offset=(left, xheight / 2))
    backslant = Glyph(name="backslant")
    backslant.width = original_width
    layer.drawPoints(TransformPointPen(backslant.getPointPen(), m))
    backslant.setLeftMargin(l, glyphset)
    backslant.setRightMargin(r, glyphset)

    boundsback = backslant.getBounds(glyphset)
    assert boundsback is not None
    left, _, _, _ = boundsback
    mf = skew_matrix((a, 0), offset=(left, xheight / 2))
    forwardslant = Glyph(name="forwardslant")
    forwardslant.width = backslant.width
    backslant.drawPoints(TransformPointPen(forwardslant.getPointPen(), mf))

    left_margin = forwardslant.getLeftMargin(glyphset)
    assert left_margin is not None
    right_margin = forwardslant.getRightMargin(glyphset)
    assert right_margin is not None
    layer.setLeftMargin(round(left_margin), glyphset)
    layer.setRightMargin(round(right_margin), glyphset)
    layer.width = round(layer.width)
    for contour in layer.contours:
        for point in contour:
            point.x = round(point.x)
            point.y = round(point.y)

    # TODO: Handle this outside the core.
    if "com.schriftgestaltung.Glyphs.originalWidth" in layer.lib:
        layer.lib["com.schriftgestaltung.Glyphs.originalWidth"] = layer.width
        layer.width = 0


def skew_matrix(angle, offset=(0, 0)):
    dx, dy = offset
    x, y = angle
    x, y = math.radians(x), math.radians(y)
    sT = Identity.translate(dx, dy)
    sT = sT.skew(x, y)
    sT = sT.translate(-dx, -dy)
    return sT


def save_to_temp_ufo(*glyphs):
    import ufoLib2

    output_ufo = ufoLib2.Font()
    for g in glyphs:
        output_ufo.layers.defaultLayer.insertGlyph(g)
    output_ufo.save("/tmp/test.ufo", overwrite=True)


# shape calculations
def rectCateto(angle, cat):
    angle = math.radians(angle)
    result = cat * (math.tan(angle))
    # result = round(result)
    return result


# point list area
def area(points: list[NSPoint]) -> float:
    s = 0
    for ii in np.arange(len(points)) - 1:
        s = s + (points[ii].x * points[ii + 1].y - points[ii + 1].x * points[ii].y)
    return abs(s) * 0.5


# get margins in Glyphs
def getMargins(
    layer: Glyph, measurement_line: Tuple[float, float, float, float]
) -> Tuple[Optional[float], Optional[float]]:
    # TODO: intersection returns a reversed list?
    result = sorted(intersections(layer, measurement_line))
    if not result:
        return (None, None)

    # Only take the outermost hits.
    left = 0
    right = -1
    return (result[left][0], result[right][0])


# a list of margins
def marginList(layer: Glyph, param_freq: int) -> Tuple[List[Any], List[Any]]:
    layer_bounds = layer.getBounds()
    assert layer_bounds is not None
    y = layer_bounds.yMin
    listL = []
    listR = []
    # works over glyph copy
    while y <= layer_bounds.yMax:
        measurement_line = layer_bounds.xMin, y, layer_bounds.xMax, y
        lpos, rpos = getMargins(layer, measurement_line)
        if lpos is not None:
            listL.append(NSMakePoint(lpos, y))
        if rpos is not None:
            listR.append(NSMakePoint(rpos, y))
        y += param_freq
    return listL, listR


####


def segments(contour: List[Point]) -> List[List[Point]]:
    if not contour:
        return []
    segments = [[]]
    lastWasOffCurve = False
    for point in contour:
        segments[-1].append(point)
        if point.segmentType is not None:
            segments.append([])
        lastWasOffCurve = point.segmentType is None
    if len(segments[-1]) == 0:
        del segments[-1]
    if lastWasOffCurve:
        lastSegment = segments[-1]
        segment = segments.pop(0)
        lastSegment.extend(segment)
    elif segments[0][-1].segmentType != "move":
        segment = segments.pop(0)
        segments.append(segment)
    return segments


def intersections(
    layer: Glyph, measurement_line: Tuple[float, float, float, float]
) -> List[Tuple[float, float]]:
    intersections: List[Tuple[float, float]] = []
    x1, y1, x2, y2 = measurement_line
    for contour in layer.contours:
        for segment_pair in arrayTools.pairwise(segments(contour.points)):
            last, curr = segment_pair
            curr_type = curr[-1].type
            if curr_type == "line":
                i = lineIntersection(x1, y1, x2, y2, last[-1], curr[-1])
                if i is not None:
                    intersections.append(i)
            elif curr_type == "curve":
                i = curveIntersections(
                    x1, y1, x2, y2, last[-1], curr[0], curr[1], curr[2]
                )
                if i:
                    intersections.extend(i)
            elif curr_type == "qcurve":
                i = qcurveIntersections(x1, y1, x2, y2, last[-1], *curr)
                if i:
                    intersections.extend(i)
            else:
                raise ValueError(f"Cannot deal with segment type {curr_type}")
    return intersections


def curveIntersections(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> List[Tuple[float, float]]:
    """
    Computes intersection between a line and a cubic curve.
    Adapted from: https://www.particleincell.com/2013/cubic-line-intersection/
    Takes four scalars describing line parameters and four points describing
    curve.
    Returns a List of intersections in Tuples of the format (coordinate_x,
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


def qcurveIntersections(
    x1: float, y1: float, x2: float, y2: float, *pts: Point
) -> List[Tuple[float, float]]:
    """
    Computes intersection between a quadratic spline and a line segment.
    Adapted from curveIntersections(). Takes four scalars describing line and an
    Iterable of points describing a quadratic curve, including the first (==
    anchor) point. Quadatric curves are special in that they can consist of
    implied on-curve points, which is why this function returns a
    `subsegment_index` to associate a `t` with the correct subsegment.
    Returns a List of intersections in Tuples of the format (coordinate_x,
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
    for index, (p2, p3) in enumerate(basePen.decomposeQuadraticSegment(points[1:])):
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


def lineIntersection(
    x1: float, y1: float, x2: float, y2: float, p3: Point, p4: Point
) -> Optional[Tuple[float, float]]:
    """
    Computes intersection between two lines.
    Adapted from Andre LaMothe, "Tricks of the Windows Game Programming Gurus".
    G. Bach, http://stackoverflow.com/a/1968345
    Takes four scalars describing line parameters and two points describing
    line.
    Returns a List of intersections in Tuples of the format (coordinate_x,
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
