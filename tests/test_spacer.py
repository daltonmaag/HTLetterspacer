import htletterspacer.core
import ufoLib2
from fontTools.pens.recordingPen import DecomposingRecordingPen


def test_spacer(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed.ufo")
    glyph_O = ufo_orig["O"]
    glyph_H = ufo_orig["H"]
    # ufo_rspc = ufoLib2.Font.open(datadir / "Respaced.ufo")

    o = htletterspacer.core.HTLetterspacerLib(
        1000, 0, ufo_orig.info.xHeight, True, True, 1.25, glyph_O.width
    )
    o.paramArea = 120
    o.paramDepth = 5
    o.paramOver = 0

    assert glyph_O.getLeftMargin() == 20
    assert glyph_O.getRightMargin() == 20

    o.spaceMain(glyph_O, glyph_H)
    assert o.minYref == 0
    assert o.maxYref == 800
    assert o.newL == 13
    assert o.newR == 13
    assert glyph_O.getLeftMargin() == 13
    assert glyph_O.getRightMargin() == 13


def test_spacer_full(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed.ufo")
    ufo_rspc = ufoLib2.Font.open(datadir / "Respaced.ufo")

    o = htletterspacer.core.HTLetterspacerLib(
        1000, 0, ufo_orig.info.xHeight, True, True, 0.0, 0.0
    )
    o.paramArea = 120
    o.paramDepth = 5
    o.paramOver = 0

    for glyph, glyph_ref, factor in (
        ("A", "H", 1.25),
        # ("Aacute", "H", 1.25),  # "Automatic Alignment, space not set."
        ("acute", "acute", 1.0),
        ("Adieresis", "H", 1.25),
        ("arrowdown", "arrowdown", 1.5),
        ("arrowleft", "arrowleft", 1.5),
        ("arrowright", "arrowright", 1.5),
        ("arrowup", "arrowup", 1.5),
        ("B", "H", 1.25),
        ("C", "H", 1.25),
        ("colon", "colon", 1.4),
        ("comma", "comma", 1.4),
        ("D", "H", 1.25),
        ("dieresis", "dieresis", 1.0),
        ("dot", "dot", 1.5),
        ("E", "H", 1.25),
        ("F", "H", 1.25),
        ("G", "H", 1.25),
        ("H", "H", 1.25),
        ("I.narrow", "H", 1.25),
        ("I", "H", 1.25),
        ("IJ", "H", 1.25),
        ("J.narrow", "H", 1.25),
        ("J", "H", 1.25),
        ("K", "H", 1.25),
        ("L", "H", 1.25),
        ("M", "H", 1.25),
        ("N", "H", 1.25),
        ("O", "H", 1.25),
        ("P", "H", 1.25),
        ("period", "period", 1.4),
        ("Q", "H", 1.25),
        ("quotedblbase", "quotedblbase", 1.2),
        ("quotedblleft", "quotedblleft", 1.2),
        ("quotedblright", "quotedblright", 1.2),
        ("quotesinglbase", "quotesinglbase", 1.2),
        ("R", "H", 1.25),
        ("S.closed", "H", 1.25),
        ("S", "H", 1.25),
        ("semicolon", "semicolon", 1.4),
        ("space", "space", 1.25),
        ("T", "H", 1.25),
        ("", "H", 1.25),
        ("V", "H", 1.25),
        ("W", "H", 1.25),
        ("X", "H", 1.25),
        ("Y", "H", 1.25),
        ("Z", "H", 1.25),
    ):
        glyph_orig = ufo_orig[glyph]
        if glyph_orig.components:
            dpen = DecomposingRecordingPen(ufo_orig)
            glyph_orig.draw(dpen)
            dpen.replay(glyph_orig.getPen())
            glyph_orig.components.clear()
        glyph_ref_orig = ufo_orig[glyph_ref]
        if glyph_ref_orig.components:
            dpen = DecomposingRecordingPen(ufo_orig)
            glyph_ref_orig.draw(dpen)
            dpen.replay(glyph_ref_orig.getPen())
            glyph_ref_orig.components.clear()
        assert not glyph_orig.components
        assert not glyph_ref_orig.components

        o.width = glyph_orig.width
        o.factor = factor
        o.spaceMain(glyph_orig, glyph_ref_orig)

        glyph_rspc = ufo_rspc[glyph]
        assert glyph_orig.getLeftMargin() == glyph_rspc.getLeftMargin(ufo_rspc), glyph
        assert glyph_orig.getRightMargin() == glyph_rspc.getRightMargin(ufo_rspc), glyph
