import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import ufoLib2
from ufoLib2.objects.misc import BoundingBox

import htletterspacer.config
import htletterspacer.core

LOGGER = logging.Logger(__name__)


def main(args: Optional[list[str]] = None) -> Optional[int]:
    parser = argparse.ArgumentParser(description="build some fonts")
    parser.add_argument("ufo", type=ufoLib2.Font.open)
    parser.add_argument("--area", type=int, default=400)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--over", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output")
    parsed_args = parser.parse_args(args)

    ufo: ufoLib2.Font = parsed_args.ufo
    assert ufo.info.italicAngle is not None
    assert isinstance(ufo.info.unitsPerEm, int)
    assert isinstance(ufo.info.xHeight, int)

    param_area: int = parsed_args.area
    param_depth: int = parsed_args.depth
    param_over: int = parsed_args.over

    if parsed_args.config is not None:
        config = htletterspacer.config.parse_config(parsed_args.config.read_text())
    else:
        config = htletterspacer.config.parse_config(
            htletterspacer.config.DEFAULT_CONFIGURATION
        )

    # Composites come last because their spacing depends on their components.
    for glyph in sorted((g for g in ufo), key=lambda g: len(g.components)):
        assert glyph.name is not None
        if not glyph.contours:
            LOGGER.warning("Skipping glyph %s because it has not contours.", glyph.name)
            continue
        if glyph.components:
            LOGGER.warning("Skipping glyph %s because it has components.", glyph.name)
            continue
        if glyph.width == 0 and any(a.name.startswith("_") for a in glyph.anchors):
            LOGGER.warning("Skipping glyph %s because it is a mark.", glyph.name)
            continue

        ref_name, factor = htletterspacer.config.reference_and_factor(config, glyph)

        # Cache bounds of reference glyphs for performance.
        glyph_ref = ufo[ref_name]
        assert glyph_ref.name is not None
        ref_bounds = glyph_ref.getBounds(ufo)
        assert ref_bounds is not None

        htletterspacer.core.space_main(
            glyph,
            ref_bounds,
            ufo,
            angle=ufo.info.italicAngle,
            compute_lsb=True,
            compute_rsb=True,
            factor=factor,
            param_area=param_area,
            param_depth=param_depth,
            param_freq=5,
            param_over=param_over,
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
