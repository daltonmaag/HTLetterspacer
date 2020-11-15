import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import fontTools.misc.bezierTools as bezierTools
import fontTools.misc.arrayTools as arrayTools
import fontTools.pens.basePen as basePen
import numpy as np
from ufoLib2.objects import Glyph
from ufoLib2.objects.misc import BoundingBox
from ufoLib2.objects.point import Point

# Default parameters
paramArea = 400  # white area in thousand units
paramDepth = 15  # depth in open counterforms, from extreme points.
paramOver = 0  # overshoot in spacing vertical range
color = False  # mark color, False for no mark
paramFreq = (
    5  # frequency of vertical measuring. Higher values are faster but less accurate
)


@dataclass
class NSPoint:
    __slots__ = "x", "y"
    x: float
    y: float


def NSMakePoint(x: float, y: float) -> NSPoint:
    return NSPoint(x, y)


# @dataclass
# class NSRect:
#     __slots__ = "x", "y", "width", "height"
#     x: float
#     y: float
#     width: float
#     height: float


def NSMinX(r: Optional[BoundingBox]) -> float:
    if r is None:
        raise ValueError("`r` is None!")
    return r.xMin


def NSMinY(r: Optional[BoundingBox]) -> float:
    if r is None:
        raise ValueError("`r` is None!")
    return r.yMin


def NSMaxX(r: Optional[BoundingBox]) -> float:
    if r is None:
        raise ValueError("`r` is None!")
    return r.xMax


def NSMaxY(r: Optional[BoundingBox]) -> float:
    if r is None:
        raise ValueError("`r` is None!")
    return r.yMax


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

    def setSpace(
        self, layer: Glyph, referenceLayer: Glyph
    ) -> Tuple[List[NSPoint], List[NSPoint]]:
        # get reference glyph maximum points
        overshoot = self.overshoot()

        # store min and max y
        self.minYref = NSMinY(referenceLayer.getBounds()) - overshoot
        self.maxYref = NSMaxY(referenceLayer.getBounds()) + overshoot

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
        lFullExtreme, rFullExtreme = self.maxPoints(
            lFullMargin + rFullMargin,
            NSMinY(layer.getBounds()),
            NSMaxY(layer.getBounds()),
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

            self.output += (
                layer.name
                + " is tabular and adjusted at width = "
                + str(self.layerWidth)
            )
        # end tabVersion

        # if there is a metric rule
        # else:
        #     if layer.parent.leftMetricsKey is not None or self.LSB == False:
        #         self.newL = layer.LSB

        #     if layer.parent.rightMetricsKey is not None or self.RSB == False:
        #         self.newR = layer.RSB
        return lPolygon, rPolygon

    def spaceMain(
        self, layer: Glyph, referenceLayer: Glyph
    ) -> Tuple[Optional[List[NSPoint]], Optional[List[NSPoint]]]:
        # TODO: decompose glyphs
        assert not layer.components

        lp, rp = None, None
        self.output = ""
        if not layer.name:
            self.output += "Something went wrong!"
        elif len(layer.contours) < 1 and len(layer.components) < 1:
            self.output += "No paths in glyph " + layer.name + "\n"
        # both sidebearings with metric keys
        # elif layer.hasAlignedWidth():
        #     self.output += (
        #         "Glyph "
        #         + layer.name
        #         + " has automatic alignment. Spacing not set.\n"
        #     )
        # elif (
        #     layer.parent.leftMetricsKey is not None
        #     and layer.parent.rightMetricsKey is not None
        # ):
        #     self.output += (
        #         "Glyph " + layer.name + " has metric keys. Spacing not set.\n"
        #     )
        # if it is tabular
        # elif '.tosf' in layer.name or '.tf' in layer.name:
        # self.output+='Glyph '+layer.name +' se supone tabular..'+"\n"
        # if it is fraction / silly condition
        elif "fraction" in layer.name:
            self.output += (
                "Glyph " + layer.name + ": should be checked and done manually.\n"
            )
        # if not...
        else:
            lp, rp = self.setSpace(layer, referenceLayer)
            # store values in a list
            setSidebearings(layer, self.newL, self.newR, self.newWidth, color)

        print(self.output)
        self.output = ""

        return lp, rp


#  Functions
def setSidebearings(
    layer: Glyph, newL: float, newR: float, width: float, color: Any
) -> None:
    layer.setLeftMargin(newL)
    layer.setRightMargin(newR)

    # adjusts the tabular miscalculation
    if width:
        layer.width = width

    if color:
        layer.lib["public.markColor"] = color


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
def getMargins(layer: Glyph, y: float) -> Tuple[Optional[float], Optional[float]]:
    startPoint = NSMakePoint(NSMinX(layer.getBounds()), y)
    endPoint = NSMakePoint(NSMaxX(layer.getBounds()), y)

    result = sorted(intersections(layer, startPoint, endPoint))
    if not result:
        return (None, None)

    left = 0
    right = -1
    return (result[left][0], result[right][0])


# a list of margins
def marginList(layer: Glyph) -> Tuple[List[Any], List[Any]]:
    y = NSMinY(layer.getBounds())
    listL = []
    listR = []
    # works over glyph copy
    # cleanLayer = layer.copyDecomposedLayer()
    while y <= NSMaxY(layer.getBounds()):
        # lpos, rpos = getMargins(cleanLayer, y)
        lpos, rpos = getMargins(layer, y)
        if y <= 0:
            print(lpos, rpos)
        if lpos is not None:
            listL.append(NSMakePoint(lpos, y))
        if rpos is not None:
            listR.append(NSMakePoint(rpos, y))
        y += paramFreq
    return listL, listR


####


# TODO: skia pathops also has a SegmentPenIterator
def segments(contour: List[Point]) -> Any:
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


def intersections(layer: Glyph, startPoint: NSPoint, endPoint: NSPoint) -> List[Any]:
    intersections = []
    x1, y1, x2, y2 = startPoint.x, startPoint.y, endPoint.x, endPoint.y
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
                if i is not None:
                    intersections.extend(i)
            elif curr_type == "qcurve":
                i = qcurveIntersections(x1, y1, x2, y2, last[-1], *curr)
                if i is not None:
                    intersections.extend(i)
            else:
                raise ValueError(f"Cannot deal with segment type {curr_type}")
    return intersections


def curveIntersections(x1, y1, x2, y2, p1, p2, p3, p4):
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


def qcurveIntersections(x1, y1, x2, y2, *pts) -> List[Tuple[int, int, int, int]]:
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


def lineIntersection(x1, y1, x2, y2, p3, p4):
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
