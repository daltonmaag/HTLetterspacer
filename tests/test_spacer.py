import htletterspacer.core
import ufoLib2


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
