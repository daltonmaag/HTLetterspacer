import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import fontTools.misc.arrayTools as arrayTools
import fontTools.misc.bezierTools as bezierTools
import fontTools.pens.basePen as basePen
import numpy as np
from ufoLib2.objects import Glyph
from ufoLib2.objects.misc import BoundingBox
from ufoLib2.objects.point import Point

LOGGER = logging.Logger(__name__)

# Default parameters
paramArea = 400  # white area in thousand units
paramDepth = 15  # depth in open counterforms, from extreme points.
paramOver = 0  # overshoot in spacing vertical range
color = False  # mark color, False for no mark
paramFreq = (
    5  # frequency of vertical measuring. Higher values are faster but less accurate
)

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
        self.paramArea = paramArea
        self.paramDepth = paramDepth
        self.paramOver = paramOver
        self.tabVersion = False
        self.upm = upm
        self.angle = angle
        self.xHeight = xHeight
        self.LSB = LSB
        self.RSB = RSB
        self.factor = factor
        self.width = width
        self.newWidth = 0.0

    def overshoot(self):
        return self.xHeight * self.paramOver / 100

    def maxPoints(self, points, minY, maxY):
        right = -10000
        left = 10000
        for p in points:
            if p.y >= minY and p.y <= maxY:
                if p.x > right:
                    right = p.x
                    righty = p.y
                if p.x < left:
                    left = p.x
                    lefty = p.y
        return NSMakePoint(left, lefty), NSMakePoint(right, righty)

    def processMargins(self, lMargin, rMargin):
        # deSlant if is italic
        lMargin = self.deSlant(lMargin)
        rMargin = self.deSlant(rMargin)

        # get extremes
        # lExtreme, rExtreme = self.maxPoints(lMargin + rMargin, self.minYref, self.maxYref)
        lExtreme, rExtreme = self.maxPoints(
            lMargin + rMargin, self.minYref, self.maxYref
        )

        # set depth
        lMargin, rMargin = self.setDepth(lMargin, rMargin, lExtreme, rExtreme)

        # close open counterforms at 45 degrees
        lMargin, rMargin = self.diagonize(lMargin, rMargin)
        lMargin = self.closeOpenCounters(lMargin, lExtreme)
        rMargin = self.closeOpenCounters(rMargin, rExtreme)

        lMargin = self.slant(lMargin)
        rMargin = self.slant(rMargin)
        return lMargin, rMargin

    # process lists with depth, proportional to xheight
    def setDepth(self, marginsL, marginsR, lExtreme, rExtreme):
        depth = self.xHeight * self.paramDepth / 100
        maxdepth = lExtreme.x + depth
        mindepth = rExtreme.x - depth
        marginsL = [NSMakePoint(min(p.x, maxdepth), p.y) for p in marginsL]
        marginsR = [NSMakePoint(max(p.x, mindepth), p.y) for p in marginsR]

        # add all the points at maximum depth if glyph is shorter than overshoot
        y = marginsL[0].y - paramFreq
        while y > self.minYref:
            marginsL.insert(0, NSMakePoint(maxdepth, y))
            marginsR.insert(0, NSMakePoint(mindepth, y))
            y -= paramFreq

        y = marginsL[-1].y + paramFreq
        while y < self.maxYref:
            marginsL.append(NSMakePoint(maxdepth, y))
            marginsR.append(NSMakePoint(mindepth, y))
            y += paramFreq

        # if marginsL[-1].y<(self.maxYref-paramFreq):
        # 	marginsL.append(NSMakePoint(min(p.x, maxdepth), self.maxYref))
        # 	marginsR.append(NSMakePoint(max(p.x, mindepth), self.maxYref))
        # if marginsL[0].y>(self.minYref):
        # 	marginsL.insert(0,NSMakePoint(min(p.x, maxdepth), self.minYref))
        # 	marginsR.insert(0,NSMakePoint(max(p.x, mindepth), self.minYref))

        return marginsL, marginsR

    # close counters at 45 degrees
    def diagonize(self, marginsL, marginsR):
        # TODO: Use https://github.com/huertatipografica/HTLetterspacer/issues/45
        total = len(marginsL) - 1

        valueFreq = paramFreq * 1.5
        for index in range(total):
            # left
            actualPoint = marginsL[index]
            nextPoint = marginsL[index + 1]
            diff = nextPoint.y - actualPoint.y
            if nextPoint.x > (actualPoint.x + diff) and nextPoint.y > actualPoint.y:
                marginsL[index + 1].x = actualPoint.x + diff
            # right
            actualPoint = marginsR[index]
            nextPoint = marginsR[index + 1]
            # if nextPoint.x < (actualPoint.x - valueFreq) and nextPoint.y > actualPoint.y:
            if nextPoint.x < (actualPoint.x - diff) and nextPoint.y > actualPoint.y:
                marginsR[index + 1].x = actualPoint.x - diff

            # left
            actualPoint = marginsL[total - index]
            nextPoint = marginsL[total - index - 1]
            diff = actualPoint.y - nextPoint.y
            if (
                nextPoint.x > (actualPoint.x + valueFreq)
                and nextPoint.y < actualPoint.y
            ):
                marginsL[total - index - 1].x = actualPoint.x + diff
            # right
            actualPoint = marginsR[total - index]
            nextPoint = marginsR[total - index - 1]
            if nextPoint.x < (actualPoint.x - diff) and nextPoint.y < actualPoint.y:
                marginsR[total - index - 1].x = actualPoint.x - diff

        return marginsL, marginsR

    # close counterforms, creating a polygon
    def closeOpenCounters(self, margin, extreme):
        initPoint = NSMakePoint(extreme.x, self.minYref)
        endPoint = NSMakePoint(extreme.x, self.maxYref)
        margin.insert(0, initPoint)
        margin.append(endPoint)
        return margin

    def _italicOnOffPoint(self, p, onoff):
        mline = self.xHeight / 2
        cateto = -p.y + mline
        if onoff == "off":
            cateto = -cateto
        xvar = -rectCateto(self.angle, cateto)
        return NSMakePoint(p.x + xvar, p.y)

    def deSlant(self, margin):
        return [self._italicOnOffPoint(p, "off") for p in margin]

    def slant(self, margin):
        return [self._italicOnOffPoint(p, "on") for p in margin]

    def calculateSBValue(self, polygon):
        amplitudeY = self.maxYref - self.minYref

        # recalculates area based on UPM
        areaUPM = self.paramArea * ((self.upm / 1000) ** 2)

        # calculates proportional area
        whiteArea = areaUPM * self.factor * 100

        propArea = (amplitudeY * whiteArea) / self.xHeight

        valor = propArea - area(polygon)
        return valor / amplitudeY

    def setSpace(self, layer: Glyph, referenceLayer: Glyph) -> None:
        # get reference glyph maximum points
        overshoot = self.overshoot()

        # store min and max y
        reference_layer_bounds = referenceLayer.getBounds()
        self.minYref = reference_layer_bounds.yMin - overshoot
        self.maxYref = reference_layer_bounds.yMax + overshoot

        # bounds
        lFullMargin, rFullMargin = marginList(layer)

        lMargins = [p for p in lFullMargin if self.minYref <= p.y <= self.maxYref]
        rMargins = [p for p in rFullMargin if self.minYref <= p.y <= self.maxYref]

        # create a closed polygon
        lPolygon, rPolygon = self.processMargins(lMargins, rMargins)
        lMargins = self.deSlant(lMargins)
        rMargins = self.deSlant(rMargins)

        lFullMargin = self.deSlant(lFullMargin)
        rFullMargin = self.deSlant(rFullMargin)

        # get extreme points deitalized
        layer_bounds = layer.getBounds()
        lFullExtreme, rFullExtreme = self.maxPoints(
            lFullMargin + rFullMargin,
            layer_bounds.yMin,
            layer_bounds.yMax,
        )
        # get zone extreme points
        lExtreme, rExtreme = self.maxPoints(
            lMargins + rMargins, self.minYref, self.maxYref
        )

        # dif between extremes full and zone
        distanceL = math.ceil(lExtreme.x - lFullExtreme.x)
        distanceR = math.ceil(rFullExtreme.x - rExtreme.x)

        # set new sidebearings
        self.newL = math.ceil(0 - distanceL + self.calculateSBValue(lPolygon))
        self.newR = math.ceil(0 - distanceR + self.calculateSBValue(rPolygon))

        # tabVersion
        if ".tosf" in layer.name or ".tf" in layer.name or self.tabVersion:
            if self.width:
                self.layerWidth = self.width
            else:
                self.layerWidth = layer.width

            widthShape = rFullExtreme.x - lFullExtreme.x
            widthActual = widthShape + self.newL + self.newR
            widthDiff = (self.layerWidth - widthActual) / 2

            self.newL += widthDiff
            self.newR += widthDiff
            self.newWidth = self.layerWidth

            LOGGER.warning(
                "%s is tabular and adjusted at width = %s",
                layer.name,
                str(self.layerWidth),
            )
        # end tabVersion

        # if there is a metric rule
        else:
            if layer.lib.get(GLYPHS_LEFT_METRICS_KEY) is not None or self.LSB == False:
                self.newL = layer.getLeftMargin()
            if layer.lib.get(GLYPHS_RIGHT_METRICS_KEY) is not None or self.RSB == False:
                self.newR = layer.getRightMargin()

    def spaceMain(self, layer: Glyph, referenceLayer: Glyph) -> None:
        # TODO: decompose glyphs
        assert not layer.components

        if not layer.name:
            LOGGER.warning("Glyph has no name.")
        elif len(layer.contours) < 1 and len(layer.components) < 1:
            LOGGER.warning("No paths in glyph %s.", layer.name)
        # both sidebearings with metric keys
        # elif layer.hasAlignedWidth():
        #     self.output += (
        #         "Glyph "
        #         + layer.name
        #         + " has automatic alignment. Spacing not set.\n"
        #     )
        elif (
            layer.lib.get(GLYPHS_LEFT_METRICS_KEY) is not None
            and layer.lib.get(GLYPHS_RIGHT_METRICS_KEY) is not None
        ):
            LOGGER.warning("Glyph %s has metric keys. Spacing not set.", layer.name)
        # if it is tabular
        elif ".tosf" in layer.name or ".tf" in layer.name:
            LOGGER.warning("Glyph %s is supposed to be tabular.", layer.name)
        # if it is fraction / silly condition
        elif "fraction" in layer.name:
            LOGGER.warning("Glyph %s should be checked and done manually.", layer.name)
        # if not...
        else:
            self.setSpace(layer, referenceLayer)
            setSidebearings(
                layer,
                self.newL,
                self.newR,
                self.newWidth,
                color,
                self.angle,
                self.xHeight,
            )


#  Functions
def setSidebearings(
    layer: Glyph,
    newL: float,
    newR: float,
    width: float,
    color: Any,
    angle: float,
    xheight: float,
) -> None:
    if angle:
        setSidebearingsSlanted(layer, newL, newR, angle, xheight)
    else:
        layer.setLeftMargin(newL)
        layer.setRightMargin(newR)

    # adjusts the tabular miscalculation
    if width:
        layer.width = width

    if color:
        layer.lib["public.markColor"] = color


from fontTools.misc.transform import Identity
from fontTools.pens.transformPen import TransformPointPen
import math


def setSidebearingsSlanted(
    layer: Glyph, l: float, r: float, a: float, xheight: float
) -> None:
    bounds = layer.getControlBounds()
    assert bounds is not None
    left, bottom, right, top = bounds
    origin = (left, xheight / 2)
    m = skew_matrix((-a, 0), offset=origin)

    original_width = (
        layer.lib.get("com.schriftgestaltung.Glyphs.originalWidth") or layer.width
    )

    backslant = Glyph()
    backslant.width = original_width
    layer.drawPoints(TransformPointPen(backslant.getPointPen(), m))
    backslant.setLeftMargin(l)
    backslant.setRightMargin(r)

    boundsback = backslant.getControlBounds()
    assert boundsback is not None
    left, bottom, right, top = boundsback
    origin = (left, xheight / 2)
    mf = skew_matrix((a, 0), offset=origin)
    forwardslant = Glyph()
    forwardslant.width = backslant.width
    backslant.drawPoints(TransformPointPen(forwardslant.getPointPen(), mf))
    # forwardslant.width = round(forwardslant.width)

    if GLYPHS_LEFT_METRICS_KEY not in layer.lib:
        layer.setLeftMargin(round(forwardslant.getLeftMargin()))
    if GLYPHS_RIGHT_METRICS_KEY not in layer.lib:
        layer.setRightMargin(round(forwardslant.getRightMargin()))
    layer.width = round(layer.width)

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


# shape calculations
def rectCateto(angle, cat):
    angle = math.radians(angle)
    result = cat * (math.tan(angle))
    # result = round(result)
    return result


# point list area
def area(points):
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
def marginList(layer: Glyph) -> Tuple[List[Any], List[Any]]:
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
        y += paramFreq
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
