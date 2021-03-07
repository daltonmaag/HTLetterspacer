from typing import Any

import htletterspacer.core
from hypothesis import given, strategies as st
from ufoLib2.objects.point import Point

coordinates = st.floats() | st.integers()
point_types = st.none() | st.sampled_from(["move", "line", "qcurve", "curve"])


@st.composite
def ufo_point(draw: Any) -> Point:
    return Point(
        x=draw(coordinates),
        y=draw(coordinates),
        type=draw(point_types),
    )


@given(st.lists(ufo_point()))
def test_segmentation(points: list[Point]):
    segments = htletterspacer.core.segments(points)

    # Invariant 1: no point left behind.
    assert len(points) == sum(len(segment) for segment in segments)

    # Invariant 1.5: it's the same points, really.
    point_ids = {id(point) for point in points}
    assert all(id(point) in point_ids for segment in segments for point in segment)

    # Invariant 2:
    assert all(
        segment  # No empty segment and...
        and (
            # ...all segments end on on-curve point...
            segment[-1].type is not None
            # ...or all points in segment are off-curves.
            or all(point.type is None for point in segment)
        )
        for segment in segments
    )
