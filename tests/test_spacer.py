import argparse

import pytest
import ufoLib2

import htletterspacer.__main__
import htletterspacer.config
import htletterspacer.core


def test_spacer(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed.ufo")
    assert ufo_orig.info.italicAngle is not None
    assert isinstance(ufo_orig.info.unitsPerEm, int)
    assert isinstance(ufo_orig.info.xHeight, int)

    glyph_O = ufo_orig["O"]
    glyph_H = ufo_orig["H"]
    ref_bounds = glyph_H.getBounds(ufo_orig)
    assert ref_bounds is not None

    assert glyph_O.getLeftMargin(ufo_orig) == 20
    assert glyph_O.getRightMargin(ufo_orig) == 20

    htletterspacer.core.space_main(
        glyph_O,
        ref_bounds,
        ufo_orig,
        angle=ufo_orig.info.italicAngle,
        compute_lsb=True,
        compute_rsb=True,
        factor=1.25,
        param_area=120,
        param_depth=5,
        param_freq=5,
        param_over=0,
        tabular_width=None,
        upm=ufo_orig.info.unitsPerEm,
        xheight=ufo_orig.info.xHeight,
    )

    assert glyph_O.getLeftMargin(ufo_orig) == 13
    assert glyph_O.getRightMargin(ufo_orig) == 13


def test_spacer_mutatorsans(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed.ufo")
    assert ufo_orig.info.italicAngle is not None
    assert isinstance(ufo_orig.info.unitsPerEm, int)
    assert isinstance(ufo_orig.info.xHeight, int)
    ufo_rspc = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed-Respaced.ufo")

    config = htletterspacer.config.parse_config(
        htletterspacer.config.DEFAULT_CONFIGURATION
    )

    # Composites come last because their spacing depends on their components.
    for glyph_orig in sorted((g for g in ufo_orig), key=lambda g: len(g.components)):
        assert glyph_orig.name is not None
        if not glyph_orig.contours and not glyph_orig.components:
            continue

        if glyph_orig.name == "Aacute":
            continue  # "Automatic Alignment, space not set."

        glyph_ref, factor = htletterspacer.config.reference_and_factor(
            config, glyph_orig
        )
        glyph_ref_orig = ufo_orig[glyph_ref]
        assert glyph_ref_orig.name is not None
        ref_bounds = glyph_ref_orig.getBounds(ufo_orig)
        assert ref_bounds is not None

        # Manual fixups because our get_data works differently from Glyphs.app's...
        if glyph_ref == "dot":
            factor = 1.5

        htletterspacer.core.space_main(
            glyph_orig,
            ref_bounds,
            ufo_orig,
            angle=ufo_orig.info.italicAngle,
            compute_lsb=True,
            compute_rsb=True,
            factor=factor,
            param_area=120,
            param_depth=5,
            param_freq=5,
            param_over=0,
            tabular_width=None,
            upm=ufo_orig.info.unitsPerEm,
            xheight=ufo_orig.info.xHeight,
        )

        glyph_rspc = ufo_rspc[glyph_orig.name]
        assert glyph_orig.getLeftMargin(ufo_orig) == glyph_rspc.getLeftMargin(
            ufo_rspc
        ), (glyph_orig.name, glyph_ref, factor)
        assert glyph_orig.getRightMargin(ufo_orig) == glyph_rspc.getRightMargin(
            ufo_rspc
        ), (glyph_orig.name, glyph_ref, factor)


def test_spacer_merriweather(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "Merriweather-LightItalic.ufo")
    assert ufo_orig.info.italicAngle is not None
    assert isinstance(ufo_orig.info.unitsPerEm, int)
    assert isinstance(ufo_orig.info.xHeight, int)
    ufo_rspc = ufoLib2.Font.open(datadir / "Merriweather-LightItalic-Respaced.ufo")

    config = htletterspacer.config.parse_config(
        htletterspacer.config.DEFAULT_CONFIGURATION
    )

    for glyph_orig in ufo_orig:
        assert glyph_orig.name is not None
        if not glyph_orig.contours and not glyph_orig.components:
            continue
        if glyph_orig.name == "fraction":
            continue  # Skipped in original code.
        if glyph_orig.components:
            continue  # composites in Merriweather are complicated: change other glyphs, ...
            # would have to space them first and then all dependents.

        glyph_ref, factor = htletterspacer.config.reference_and_factor(
            config, glyph_orig
        )
        glyph_ref_orig = ufo_orig[glyph_ref]
        assert glyph_ref_orig.name is not None
        ref_bounds = glyph_ref_orig.getBounds(ufo_orig)
        assert ref_bounds is not None

        htletterspacer.core.space_main(
            glyph_orig,
            ref_bounds,
            ufo_orig,
            angle=-ufo_orig.info.italicAngle,
            compute_lsb=True,
            compute_rsb=True,
            factor=factor,
            param_area=400,
            param_depth=15,
            param_freq=5,
            param_over=0,
            tabular_width=None,
            upm=ufo_orig.info.unitsPerEm,
            xheight=ufo_orig.info.xHeight,
        )

        glyph_rspc = ufo_rspc[glyph_orig.name]

        left_should = glyph_rspc.getLeftMargin(ufo_rspc)
        if left_should is None:
            continue  # skip emquad, etc.
        left_is = glyph_orig.getLeftMargin(ufo_orig)
        assert left_is is not None
        assert round(left_is) == pytest.approx(round(left_should), abs=1), (
            glyph_orig.name,
            glyph_ref,
            factor,
        )

        right_should = glyph_rspc.getRightMargin(ufo_rspc)
        assert right_should is not None
        right_is = glyph_orig.getRightMargin(ufo_orig)
        assert right_is is not None
        assert round(right_is) == pytest.approx(round(right_should), abs=1), (
            glyph_orig.name,
            glyph_ref,
            factor,
        )


def test_spacer_components(datadir):
    """Test that spacing glyphs leaves their components in other glyphs where
    they are."""

    ufo = ufoLib2.Font.open(datadir / "NestedComponents.ufo")

    glyph_C = ufo["C"]
    glyph_D = ufo["D"]
    glyph_E = ufo["E"]

    bbox_C_1 = glyph_C.components[0].getBounds(ufo)
    bbox_C_2 = glyph_C.components[1].getBounds(ufo)
    bbox_C_x_offset = bbox_C_1.xMin - bbox_C_2.xMin

    bbox_D = glyph_D.getBounds(ufo)
    bbox_D_length = bbox_D.xMax - bbox_D.xMin

    bbox_E_1 = glyph_E.components[0].getBounds(ufo)
    bbox_E_2 = glyph_E.components[1].getBounds(ufo)
    bbox_E_3 = glyph_E.components[2].getBounds(ufo)
    bbox_E_4 = glyph_E.components[3].getBounds(ufo)
    bbox_E_x_offset1 = bbox_E_1.xMin - bbox_E_2.xMin
    bbox_E_x_offset2 = bbox_E_1.xMin - bbox_E_3.xMin
    bbox_E_x_offset3 = bbox_E_1.xMin - bbox_E_4.xMin

    htletterspacer.__main__.space_ufo(
        argparse.Namespace(ufo=ufo, area=400, depth=15, overshoot=0, config=None)
    )

    # C: test offset of components from each other stays the same.
    bbox_C_1_new = glyph_C.components[0].getBounds(ufo)
    bbox_C_2_new = glyph_C.components[1].getBounds(ufo)
    bbox_C_x_offset_new = bbox_C_1_new.xMin - bbox_C_2_new.xMin
    assert bbox_C_x_offset == bbox_C_x_offset_new

    # D: test glyph bbox x-length stays the same to test x-moving of flipped
    # component.
    bbox_D_new = glyph_D.getBounds(ufo)
    bbox_D_new_length = bbox_D_new.xMax - bbox_D_new.xMin
    assert bbox_D_length == bbox_D_new_length

    # E: test offset of components from each other stays the same.
    bbox_E_new_1 = glyph_E.components[0].getBounds(ufo)
    bbox_E_new_2 = glyph_E.components[1].getBounds(ufo)
    bbox_E_new_3 = glyph_E.components[2].getBounds(ufo)
    bbox_E_new_4 = glyph_E.components[3].getBounds(ufo)
    bbox_E_new_x_offset1 = bbox_E_new_1.xMin - bbox_E_new_2.xMin
    bbox_E_new_x_offset2 = bbox_E_new_1.xMin - bbox_E_new_3.xMin
    bbox_E_new_x_offset3 = bbox_E_new_1.xMin - bbox_E_new_4.xMin
    assert bbox_E_x_offset1 == bbox_E_new_x_offset1
    assert bbox_E_x_offset2 == bbox_E_new_x_offset2
    assert bbox_E_x_offset3 == bbox_E_new_x_offset3
