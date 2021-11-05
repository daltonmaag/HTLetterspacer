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

    htletterspacer.__main__.space_ufo(
        argparse.Namespace(
            ufo=ufo_orig,
            area=120,
            depth=5,
            overshoot=0,
            config=None,
            debug_polygons_in_background=None,
        )
    )

    for glyph in ufo_orig:
        assert glyph.name
        glyph_rspc = ufo_rspc[glyph.name]
        left_is = glyph.getLeftMargin(ufo_orig)
        left_expected = glyph_rspc.getLeftMargin(ufo_rspc)
        assert left_is == left_expected, glyph.name
        right_is = glyph.getRightMargin(ufo_orig)
        right_expected = glyph_rspc.getRightMargin(ufo_rspc)
        assert right_is == right_expected, glyph.name


def test_spacer_merriweather(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "Merriweather-LightItalic.ufo")
    assert ufo_orig.info.italicAngle is not None
    assert isinstance(ufo_orig.info.unitsPerEm, int)
    assert isinstance(ufo_orig.info.xHeight, int)
    ufo_rspc = ufoLib2.Font.open(datadir / "Merriweather-LightItalic-Respaced.ufo")

    htletterspacer.__main__.space_ufo(
        argparse.Namespace(
            ufo=ufo_orig,
            area=400,
            depth=15,
            overshoot=0,
            config=None,
            debug_polygons_in_background=None,
        )
    )

    for glyph in ufo_orig:
        assert glyph.name
        glyph_rspc = ufo_rspc[glyph.name]
        left_is = glyph.getLeftMargin(ufo_orig)
        left_expected = glyph_rspc.getLeftMargin(ufo_rspc)
        assert left_is == left_expected, glyph.name
        right_is = glyph.getRightMargin(ufo_orig)
        right_expected = glyph_rspc.getRightMargin(ufo_rspc)
        if isinstance(right_is, float) and isinstance(right_expected, float):
            assert right_is == pytest.approx(right_expected), glyph.name
        else:
            assert right_is == right_expected, glyph.name


def test_spacer_components(datadir):
    """Test that spacing glyphs leaves their components in other glyphs where
    they are."""

    ufo = ufoLib2.Font.open(datadir / "NestedComponents.ufo")

    glyph_C = ufo["C"]
    glyph_D = ufo["D"]
    glyph_E = ufo["E"]
    glyph_F = ufo["F"]

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

    bbox_F_1 = glyph_F.contours[0].getBounds()
    bbox_F_2 = glyph_F.components[0].getBounds(ufo)
    bbox_F_x_offset = bbox_F_1.xMin - bbox_F_2.xMin

    htletterspacer.__main__.space_ufo(
        argparse.Namespace(
            ufo=ufo,
            area=400,
            depth=15,
            overshoot=0,
            config=None,
            debug_polygons_in_background=None,
        )
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

    # F: test offset of component and outline from each other stays the same.
    bbox_F_new_1 = glyph_F.contours[0].getBounds()
    bbox_F_new_2 = glyph_F.components[0].getBounds(ufo)
    bbox_F_new_x_offset = bbox_F_new_1.xMin - bbox_F_new_2.xMin
    assert bbox_F_x_offset == bbox_F_new_x_offset
