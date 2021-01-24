import argparse
import collections
import graphlib
import logging
import sys
from pathlib import Path
from typing import Optional

import ufoLib2
import fontTools.misc.transform

import htletterspacer.config
import htletterspacer.core

LOGGER = logging.Logger(__name__)

AREA_KEY = "com.ht.spacer.area"
DEPTH_KEY = "com.ht.spacer.depth"
OVERSHOOT_KEY = "com.ht.spacer.overshoot"

# TODO: build graph of composite dependencies and space everything, including composites
# TODO: respect metrics keys by skipping that side or by interpreting them?
# TODO: pull in glyphConstruction to rebuild components?
def main(args: Optional[list[str]] = None) -> Optional[int]:
    parser = argparse.ArgumentParser(description="Respace all glyphs with contours.")
    parser.add_argument("ufo", type=ufoLib2.Font.open)
    parser.add_argument(
        "--area",
        type=int,
        help="Set the UFO-wide area parameter (can be overriden on the glyph level).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Set the UFO-wide depth parameter (can be overriden on the glyph level).",
    )
    parser.add_argument(
        "--overshoot",
        type=int,
        help="Set the UFO-wide overshoot parameter (can be overriden on the glyph level).",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output")
    parsed_args = parser.parse_args(args)

    space_ufo(parsed_args)

    if parsed_args.output:
        parsed_args.ufo.save(parsed_args.output, overwrite=True)
    else:
        parsed_args.ufo.save()


def space_ufo(args: argparse.Namespace) -> None:
    ufo: ufoLib2.Font = args.ufo
    assert ufo.info.italicAngle is not None
    assert isinstance(ufo.info.unitsPerEm, int)
    assert isinstance(ufo.info.xHeight, int)

    param_area: int = args.area or ufo.lib.get(AREA_KEY, 400)
    param_depth: int = args.depth or ufo.lib.get(DEPTH_KEY, 15)
    param_over: int = args.overshoot or ufo.lib.get(OVERSHOOT_KEY, 0)

    if args.config is not None:
        config = htletterspacer.config.parse_config(args.config.read_text())
    else:
        config = htletterspacer.config.parse_config(
            htletterspacer.config.DEFAULT_CONFIGURATION
        )

    glyph_graph = {
        g.name: {c.baseGlyph for c in g.components} for g in ufo if g.name is not None
    }
    composite_graph: collections.defaultdict[str, set[str]]
    composite_graph = collections.defaultdict(set)
    for g in ufo:
        if g.name is None:
            continue
        glyph_graph[g.name] = set()
        for c in g.components:
            glyph_graph[g.name].add(c.baseGlyph)
            composite_graph[c.baseGlyph].add(g.name)

    # Composites come last because their spacing depends on their components.
    ts = graphlib.TopologicalSorter(glyph_graph)
    for glyph_name in tuple(ts.static_order()):
        glyph = ufo[glyph_name]
        assert glyph.name is not None

        if not glyph.contours and not glyph.components:
            LOGGER.warning(
                "Skipping glyph %s because it has neither contours nor components.",
                glyph.name,
            )
            continue
        if glyph.width == 0 and any(a.name.startswith("_") for a in glyph.anchors):
            LOGGER.warning("Skipping glyph %s because it is a mark.", glyph.name)
            continue

        ref_name, factor = htletterspacer.config.reference_and_factor(config, glyph)

        try:
            glyph_ref = ufo[ref_name]
        except KeyError as e:
            LOGGER.warning(
                "Reference glyph %s does not exist, spacing %s with own bounds.",
                ref_name,
                glyph_name,
            )
            glyph_ref = glyph
        assert glyph_ref.name is not None
        ref_bounds = glyph_ref.getBounds(ufo)
        assert ref_bounds is not None

        glyph_param_area: int = glyph.lib.get(AREA_KEY, param_area)
        glyph_param_depth: int = glyph.lib.get(DEPTH_KEY, param_depth)
        glyph_param_over: int = glyph.lib.get(OVERSHOOT_KEY, param_over)

        left_before = glyph.getLeftMargin(ufo)
        assert left_before is not None

        htletterspacer.core.space_main(
            glyph,
            ref_bounds,
            ufo,
            angle=-ufo.info.italicAngle,
            compute_lsb=True,
            compute_rsb=True,
            factor=factor,
            param_area=glyph_param_area,
            param_depth=glyph_param_depth,
            param_freq=5,
            param_over=glyph_param_over,
            tabular_width=None,
            upm=ufo.info.unitsPerEm,
            xheight=ufo.info.xHeight,
        )

        # If the glyph is used as a component in any other glyph, move that component
        # in the opposite direction (measured to the left, to the origin) to ensure
        # that existing components stay as before.
        if glyph_name in composite_graph:
            left_after = glyph.getLeftMargin(ufo)
            assert left_after is not None
            left_diff = left_before - left_after
            for composite_name in composite_graph.get(glyph_name, []):
                composite = ufo[composite_name]
                for c in composite.components:
                    if c.baseGlyph != glyph_name:
                        continue
                    c.transformation = c.transformation.translate(left_diff, 0)


if __name__ == "__main__":
    sys.exit(main())
