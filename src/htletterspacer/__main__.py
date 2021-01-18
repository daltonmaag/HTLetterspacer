import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import ufoLib2

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

    ufo: ufoLib2.Font = parsed_args.ufo
    assert ufo.info.italicAngle is not None
    assert isinstance(ufo.info.unitsPerEm, int)
    assert isinstance(ufo.info.xHeight, int)

    param_area: int = parsed_args.area or ufo.lib.get(AREA_KEY, 400)
    param_depth: int = parsed_args.depth or ufo.lib.get(DEPTH_KEY, 15)
    param_over: int = parsed_args.overshoot or ufo.lib.get(OVERSHOOT_KEY, 0)

    if parsed_args.config is not None:
        config = htletterspacer.config.parse_config(parsed_args.config.read_text())
    else:
        config = htletterspacer.config.parse_config(
            htletterspacer.config.DEFAULT_CONFIGURATION
        )

    # Composites come last because their spacing depends on their components.
    for glyph in sorted((g for g in ufo), key=lambda g: len(g.components)):
        assert glyph.name is not None
        if glyph.components:
            LOGGER.warning("Skipping glyph %s because it has components.", glyph.name)
            continue
        if glyph.width == 0 and any(a.name.startswith("_") for a in glyph.anchors):
            LOGGER.warning("Skipping glyph %s because it is a mark.", glyph.name)
            continue
        if not glyph.contours:
            LOGGER.warning("Skipping glyph %s because it has no contours.", glyph.name)
            continue

        ref_name, factor = htletterspacer.config.reference_and_factor(config, glyph)

        glyph_ref = ufo[ref_name]
        assert glyph_ref.name is not None
        ref_bounds = glyph_ref.getBounds(ufo)
        assert ref_bounds is not None

        glyph_param_area: int = glyph.lib.get(AREA_KEY, param_area)
        glyph_param_depth: int = glyph.lib.get(DEPTH_KEY, param_depth)
        glyph_param_over: int = glyph.lib.get(OVERSHOOT_KEY, param_over)

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

    if parsed_args.output:
        ufo.save(parsed_args.output, overwrite=True)
    else:
        ufo.save()


if __name__ == "__main__":
    sys.exit(main())
