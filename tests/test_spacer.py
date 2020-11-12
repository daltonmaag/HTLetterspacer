import htletterspacer.core
import ufoLib2


def test_spacer(datadir):
    ufo = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed.ufo")
    o = htletterspacer.core.HTLetterspacerLib(
        1000, 0, ufo.info.xHeight, True, True, 1.25, 500
    )
    glyph_O = ufo["O"]
    glyph_H = ufo["H"]
    lp, rp = o.spaceMain(glyph_O, glyph_H)
    assert lp
    assert rp
