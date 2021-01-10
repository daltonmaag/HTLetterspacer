import pytest
import ufoLib2

import htletterspacer.core


def test_spacer(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed.ufo")
    glyph_O = ufo_orig["O"]
    glyph_H = ufo_orig["H"]
    # ufo_rspc = ufoLib2.Font.open(datadir / "Respaced.ufo")

    o = htletterspacer.core.HTLetterspacerLib(
        ufo_orig.info.unitsPerEm,
        ufo_orig.info.italicAngle,
        ufo_orig.info.xHeight,
        True,
        True,
        1.25,
        glyph_O.width,
    )
    o.paramArea = 120
    o.paramDepth = 5
    o.paramOver = 0

    assert glyph_O.getLeftMargin(ufo_orig) == 20
    assert glyph_O.getRightMargin(ufo_orig) == 20

    o.spaceMain(glyph_O, glyph_H, ufo_orig)
    assert o.minYref == 0
    assert o.maxYref == 800
    assert o.newL == 13
    assert o.newR == 13
    assert glyph_O.getLeftMargin(ufo_orig) == 13
    assert glyph_O.getRightMargin(ufo_orig) == 13


def test_spacer_mutatorsans(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed.ufo")
    ufo_rspc = ufoLib2.Font.open(datadir / "MutatorSansBoldCondensed-Respaced.ufo")

    o = htletterspacer.core.HTLetterspacerLib(
        ufo_orig.info.unitsPerEm,
        ufo_orig.info.italicAngle,
        ufo_orig.info.xHeight,
        True,
        True,
        0.0,
        0.0,
    )
    o.paramArea = 120
    o.paramDepth = 5
    o.paramOver = 0

    for glyph, glyph_ref, factor in (
        ("A", "H", 1.25),
        ("acute", "acute", 1.0),
        ("arrowdown", "arrowdown", 1.5),
        ("arrowleft", "arrowleft", 1.5),
        ("arrowright", "arrowright", 1.5),
        ("arrowup", "arrowup", 1.5),
        ("B", "H", 1.25),
        ("C", "H", 1.25),
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
        ("R", "H", 1.25),
        ("S.closed", "H", 1.25),
        ("S", "H", 1.25),
        ("space", "space", 1.25),
        ("T", "H", 1.25),
        ("U", "H", 1.25),
        ("V", "H", 1.25),
        ("W", "H", 1.25),
        ("X", "H", 1.25),
        ("Y", "H", 1.25),
        ("Z", "H", 1.25),
        # Composites come last because their spacing depends on their components.
        # ("Aacute", "H", 1.25),  # "Automatic Alignment, space not set."
        ("Adieresis", "H", 1.25),
        ("colon", "colon", 1.4),
        ("Q", "H", 1.25),
        ("quotedblbase", "quotedblbase", 1.2),
        ("quotedblleft", "quotedblleft", 1.2),
        ("quotedblright", "quotedblright", 1.2),
        ("quotesinglbase", "quotesinglbase", 1.2),
        ("semicolon", "semicolon", 1.4),
    ):
        glyph_orig = ufo_orig[glyph]
        glyph_ref_orig = ufo_orig[glyph_ref]

        o.width = glyph_orig.width
        o.factor = factor
        o.spaceMain(glyph_orig, glyph_ref_orig, ufo_orig)

        glyph_rspc = ufo_rspc[glyph]
        assert glyph_orig.getLeftMargin(ufo_orig) == glyph_rspc.getLeftMargin(
            ufo_rspc
        ), glyph
        assert glyph_orig.getRightMargin(ufo_orig) == glyph_rspc.getRightMargin(
            ufo_rspc
        ), glyph


def test_spacer_merriweather(datadir):
    ufo_orig = ufoLib2.Font.open(datadir / "Merriweather-LightItalic.ufo")
    ufo_rspc = ufoLib2.Font.open(datadir / "Merriweather-LightItalic-Respaced.ufo")

    o = htletterspacer.core.HTLetterspacerLib(
        ufo_orig.info.unitsPerEm,
        -ufo_orig.info.italicAngle,
        ufo_orig.info.xHeight,
        True,
        True,
        0.0,
        0.0,
    )
    o.paramArea = 400
    o.paramDepth = 15
    o.paramOver = 0

    for glyph, glyph_ref, factor in (
        # (".notdef", ".notdef", 1.1),
        ("dzhe_desc.part", "dzhe_desc.part", 1.0),
        ("a.sc", "h.sc", 1.1),
        ("A", "H", 1.25),
        ("a", "x", 1.0),
        ("accountof", "accountof", 1.5),
        ("acute", "acute", 1.0),
        ("acutecomb.case", "acutecomb.case", 1.0),
        ("acutecomb", "acutecomb", 1.0),
        ("addressedtothesubject", "addressedtothesubject", 1.5),
        ("ae.sc", "h.sc", 1.1),
        ("AE", "H", 1.25),
        ("ae", "x", 1.0),
        ("aie-cy.sc", "h.sc", 1.1),
        ("ampersand", "ampersand", 1.5),
        ("apostrophemod", "apostrophemod", 1.0),
        ("approxequal.case", "approxequal.case", 1.5),
        ("approxequal", "approxequal", 1.5),
        ("asciicircum", "asciicircum", 1.5),
        ("asciitilde.case", "asciitilde.case", 1.5),
        ("asciitilde", "asciitilde", 1.5),
        ("asterisk", "asterisk", 1.4),
        ("at.case", "at.case", 1.5),
        ("at", "at", 1.5),
        ("b.sc", "h.sc", 1.1),
        ("B", "H", 1.25),
        ("b", "x", 1.0),
        ("backslash.case", "backslash.case", 1.0),
        ("backslash", "backslash", 1.0),
        ("bar", "bar", 1.5),
        ("barlc", "barlc", 1.0),
        ("be-cy.loclSRB", "x", 1.0),
        ("be-cy.sc", "h.sc", 1.1),
        ("Be-cy", "H", 1.25),
        ("be-cy", "x", 1.0),
        ("bitcoin", "bitcoin", 1.6),
        ("braceleft.case", "braceleft.case", 1.2),
        ("braceleft.sc", "braceleft.sc", 1.0),
        ("braceleft", "braceleft", 1.2),
        ("braceright.case", "braceright.case", 1.2),
        ("braceright.sc", "braceright.sc", 1.0),
        ("braceright", "braceright", 1.2),
        ("bracketleft.case", "bracketleft.case", 1.2),
        ("bracketleft.sc", "bracketleft.sc", 1.0),
        ("bracketleft", "bracketleft", 1.2),
        ("bracketright.case", "bracketright.case", 1.2),
        ("bracketright.sc", "bracketright.sc", 1.0),
        ("bracketright", "bracketright", 1.2),
        ("breve", "breve", 1.0),
        ("brevebelowcomb.case", "brevebelowcomb.case", 1.0),
        ("brevebelowcomb", "brevebelowcomb", 1.0),
        ("brevecomb_acutecomb.case", "brevecomb_acutecomb.case", 1.0),
        ("brevecomb_acutecomb", "brevecomb_acutecomb", 1.0),
        ("brevecomb_gravecomb.case", "brevecomb_gravecomb.case", 1.0),
        ("brevecomb_gravecomb", "brevecomb_gravecomb", 1.0),
        ("brevecomb_hookabovecomb.case", "brevecomb_hookabovecomb.case", 1.0),
        ("brevecomb_hookabovecomb", "brevecomb_hookabovecomb", 1.0),
        ("brevecomb_tildecomb.case", "brevecomb_tildecomb.case", 1.0),
        ("brevecomb_tildecomb", "brevecomb_tildecomb", 1.0),
        ("brevecomb-cy.case", "brevecomb-cy.case", 1.0),
        ("brevecomb-cy", "brevecomb-cy", 1.0),
        ("brevecomb.case", "brevecomb.case", 1.0),
        ("brevecomb", "brevecomb", 1.0),
        ("breveinvertedcomb.case", "breveinvertedcomb.case", 1.0),
        ("breveinvertedcomb", "breveinvertedcomb", 1.0),
        ("brokenbar", "brokenbar", 1.5),
        ("bullet.case", "bullet.case", 1.4),
        ("bullet", "bullet", 1.4),
        ("bulletoperator", "bulletoperator", 1.5),
        ("c.sc", "h.sc", 1.1),
        ("C", "H", 1.25),
        ("c", "x", 1.0),
        ("careof", "careof", 1.5),
        ("caron", "caron", 1.0),
        ("caroncomb.alt", "caroncomb.alt", 1.0),
        ("caroncomb.case", "caroncomb.case", 1.0),
        ("caroncomb", "caroncomb", 1.0),
        ("cedi", "cedi", 1.6),
        ("cedilla", "cedilla", 1.0),
        ("cedillacomb.case", "cedillacomb.case", 1.0),
        ("cedillacomb", "cedillacomb", 1.0),
        ("cent", "cent", 1.6),
        ("che-cy.sc", "h.sc", 1.1),
        ("Che-cy", "H", 1.25),
        ("che-cy", "x", 1.0),
        ("cheabkhasian-cy.sc", "h.sc", 1.1),
        ("Cheabkhasian-cy", "H", 1.25),
        ("cheabkhasian-cy", "x", 1.0),
        ("chekhakassian-cy", "x", 1.0),
        ("Cheverticalstroke-cy", "H", 1.25),
        ("circumflex", "circumflex", 1.0),
        ("circumflexcomb_acutecomb.case", "circumflexcomb_acutecomb.case", 1.0),
        ("circumflexcomb_acutecomb", "circumflexcomb_acutecomb", 1.0),
        ("circumflexcomb_gravecomb.case", "circumflexcomb_gravecomb.case", 1.0),
        ("circumflexcomb_gravecomb", "circumflexcomb_gravecomb", 1.0),
        ("circumflexcomb_hookabovecomb.case", "circumflexcomb_hookabovecomb.case", 1.0),
        ("circumflexcomb_hookabovecomb", "circumflexcomb_hookabovecomb", 1.0),
        ("circumflexcomb_tildecomb.case", "circumflexcomb_tildecomb.case", 1.0),
        ("circumflexcomb_tildecomb", "circumflexcomb_tildecomb", 1.0),
        ("circumflexcomb.case", "circumflexcomb.case", 1.0),
        ("circumflexcomb", "circumflexcomb", 1.0),
        ("colon", "colon", 1.4),
        ("colonsign", "colonsign", 1.6),
        ("comma", "comma", 1.4),
        ("commaaccentcomb.case", "commaaccentcomb.case", 1.0),
        ("commaaccentcomb", "commaaccentcomb", 1.0),
        ("commaturnedabovecomb.case", "commaturnedabovecomb.case", 1.0),
        ("commaturnedabovecomb", "commaturnedabovecomb", 1.0),
        ("commaturnedmod", "commaturnedmod", 1.0),
        ("commercialMinusSign", "commercialMinusSign", 1.5),
        ("copyright", "copyright", 1.5),
        ("currency", "currency", 1.6),
        ("d.sc", "h.sc", 1.1),
        ("D", "H", 1.25),
        ("d", "x", 1.0),
        ("dagger", "dagger", 1.5),
        ("daggerdbl", "daggerdbl", 1.5),
        ("dblgravecomb.case", "dblgravecomb.case", 1.0),
        ("dblgravecomb", "dblgravecomb", 1.0),
        ("dcaron", "x", 1.0),
        ("dcroat", "x", 1.0),
        ("de-cy.loclBGR.sc", "h.sc", 1.1),
        ("De-cy.loclBGR", "H", 1.25),
        ("de-cy.sc", "h.sc", 1.1),
        ("De-cy", "H", 1.25),
        ("de-cy", "x", 1.0),
        ("degree", "degree", 1.5),
        ("delta.sc", "h.sc", 1.1),
        ("Delta", "H", 1.25),
        ("desccy_part.", "desccy_part.", 1.0),
        ("descender", "descender", 1.0),
        ("dieresis-cy", "dieresis-cy", 1.0),
        ("dieresis", "dieresis", 1.0),
        ("dieresisbelowcomb.case", "dieresisbelowcomb.case", 1.0),
        ("dieresisbelowcomb", "dieresisbelowcomb", 1.0),
        ("dieresiscomb-cy.case", "dieresiscomb-cy.case", 1.0),
        ("dieresiscomb.case", "dieresiscomb.case", 1.0),
        ("dieresiscomb", "dieresiscomb", 1.0),
        ("divide.case", "divide.case", 1.5),
        ("divide", "divide", 1.5),
        ("dje-cy.sc", "h.sc", 1.1),
        ("Dje-cy", "H", 1.25),
        ("dje-cy", "x", 1.0),
        ("dollar.ss01", "dollar.ss01", 1.6),
        ("dollar", "dollar", 1.6),
        ("dong", "dong", 1.6),
        ("dotaccent", "dotaccent", 1.0),
        ("dotaccentcomb.case", "dotaccentcomb.case", 1.0),
        ("dotaccentcomb", "dotaccentcomb", 1.0),
        ("dotbelowcomb.case", "dotbelowcomb.case", 1.0),
        ("dotbelowcomb", "dotbelowcomb", 1.0),
        ("doubleprimemod", "doubleprimemod", 1.0),
        ("downArrow", "downArrow", 1.5),
        ("dram-arm", "dram-arm", 1.6),
        ("Dz", "H", 1.25),
        ("DZ", "H", 1.25),
        ("DZcaron", "H", 1.25),
        ("dzhe-cy.sc", "h.sc", 1.1),
        ("Dzhe-cy", "H", 1.25),
        ("e-cy.sc", "h.sc", 1.1),
        ("E-cy", "H", 1.25),
        ("e-cy", "x", 1.0),
        ("e.sc", "h.sc", 1.1),
        ("E", "H", 1.25),
        ("e", "x", 1.0),
        ("ef-cy.loclBGR.sc", "h.sc", 1.1),
        ("Ef-cy.loclBGR", "H", 1.25),
        ("ef-cy.sc", "h.sc", 1.1),
        ("Ef-cy", "H", 1.25),
        ("ef-cy", "x", 1.0),
        ("eight.dnom", "eight.dnom", 0.8),
        ("eight.lf", "one", 1.2),
        ("eight.osf", "zero.osf", 1.2),
        ("eight.sc", "eight.sc", 0.8),
        ("eight.tf", "one", 1.2),
        ("eight.tosf", "one", 1.2),
        ("eight", "one", 1.2),
        ("el-cy.loclBGR.sc", "h.sc", 1.1),
        ("El-cy.loclBGR", "H", 1.25),
        ("el-cy.loclBGR", "x", 1.0),
        ("el-cy.sc", "h.sc", 1.1),
        ("El-cy", "H", 1.25),
        ("el-cy", "x", 1.0),
        ("Eldescender-cy", "H", 1.25),
        ("elhook-cy.sc", "h.sc", 1.1),
        ("Elhook-cy", "H", 1.25),
        ("elhook-cy", "x", 1.0),
        ("ellipsis", "ellipsis", 1.4),
        ("eltail-cy", "x", 1.0),
        ("em-cy", "x", 1.0),
        ("emdash.case", "emdash.case", 1.0),
        ("emdash", "emdash", 1.0),
        ("emptyset.case", "emptyset.case", 1.5),
        ("emptyset", "emptyset", 1.5),
        ("emquad", "emquad", 1.25),
        ("emspace", "emspace", 1.0),
        ("emtail-cy", "x", 1.0),
        ("en-cy", "x", 1.0),
        ("endash.case", "endash.case", 1.0),
        ("endash", "endash", 1.0),
        ("eng.sc", "h.sc", 1.1),
        ("Eng", "H", 1.25),
        ("eng", "x", 1.0),
        ("enghe-cy.sc", "h.sc", 1.1),
        ("Enghe-cy", "H", 1.25),
        ("enghe-cy", "x", 1.0),
        ("enhook-cy.sc", "h.sc", 1.1),
        ("Enhook-cy", "H", 1.25),
        ("enhook-cy", "x", 1.0),
        ("enlefthook-cy.sc", "h.sc", 1.1),
        ("EnLeftHook-cy", "H", 1.25),
        ("enlefthook-cy", "x", 1.0),
        ("enquad", "enquad", 1.0),
        ("enspace", "enspace", 1.1),
        ("entail-cy", "x", 1.0),
        ("equal.case", "equal.case", 1.5),
        ("equal", "equal", 1.5),
        ("ereversed-cy.sc", "h.sc", 1.1),
        ("Ereversed-cy", "H", 1.25),
        ("ereversed-cy", "x", 1.0),
        ("Esdescender-cy.loclBSH", "H", 1.25),
        ("esdescender-cy.loclCHU.sc", "h.sc", 1.1),
        ("esdescender-cy.sc", "h.sc", 1.1),
        ("esdescender-cy", "x", 1.0),
        ("estimated", "estimated", 1.5),
        ("eth", "x", 1.0),
        ("euro", "euro", 1.6),
        ("exclam.sc", "exclam.sc", 1.0),
        ("exclam", "exclam", 1.4),
        ("exclamdouble", "exclamdouble", 1.4),
        ("exclamdown.case", "exclamdown.case", 1.4),
        ("exclamdown.sc", "exclamdown.sc", 1.0),
        ("exclamdown", "exclamdown", 1.4),
        ("ezh.sc", "h.sc", 1.1),
        ("Ezh", "H", 1.25),
        ("ezh", "x", 1.0),
        ("Ezhcaron", "H", 1.25),
        ("f.sc", "h.sc", 1.1),
        ("F", "H", 1.25),
        ("f", "x", 1.0),
        ("Fdotaccent", "H", 1.25),
        ("figuredash.case", "figuredash.case", 1.0),
        ("figuredash", "figuredash", 1.0),
        ("figurespace", "figurespace", 1.4),
        ("firsttonechinese", "firsttonechinese", 1.0),
        ("fita-cy.sc", "h.sc", 1.1),
        ("Fita-cy", "H", 1.25),
        ("fita-cy", "x", 1.0),
        ("five.dnom", "five.dnom", 0.8),
        ("five.lf", "one", 1.2),
        ("five.osf", "zero.osf", 1.2),
        ("five.sc", "five.sc", 0.8),
        ("five.tf", "one", 1.2),
        ("five.tosf", "one", 1.2),
        ("five", "one", 1.2),
        ("florin", "florin", 1.6),
        ("four.dnom", "four.dnom", 0.8),
        ("four.lf", "one", 1.2),
        ("four.osf", "zero.osf", 1.2),
        ("four.sc", "four.sc", 0.8),
        ("four.tf", "one", 1.2),
        ("four.tosf", "one", 1.2),
        ("four", "one", 1.2),
        ("fourperemspace", "fourperemspace", 1.0),
        ("fourthtonechinese", "fourthtonechinese", 1.0),
        # ("fraction", "fraction", 1.3),  # Skipped in original code.
        ("franc", "franc", 1.6),
        ("g.sc", "h.sc", 1.1),
        ("G", "H", 1.25),
        ("g", "x", 1.0),
        ("ge-cy.loclBGR", "x", 1.0),
        ("ge-cy.sc", "h.sc", 1.1),
        ("Ge-cy", "H", 1.25),
        ("ge-cy", "x", 1.0),
        ("gedescender-cy", "x", 1.0),
        ("germandbls.sc", "h.sc", 1.1),
        ("Germandbls", "H", 1.25),
        ("germandbls", "x", 1.0),
        ("gestrokehook-cy.sc", "h.sc", 1.1),
        ("Gestrokehook-cy", "H", 1.25),
        ("ghemiddlehook-cy.sc", "h.sc", 1.1),
        ("Ghemiddlehook-cy", "H", 1.25),
        ("ghemiddlehook-cy", "x", 1.0),
        ("ghestroke-cy.loclBSH", "x", 1.0),
        ("ghestroke-cy.sc", "h.sc", 1.1),
        ("Ghestroke-cy", "H", 1.25),
        ("ghestroke-cy", "x", 1.0),
        ("gheupturn-cy.sc", "h.sc", 1.1),
        ("Gheupturn-cy", "H", 1.25),
        ("gheupturn-cy", "x", 1.0),
        ("gje-cy", "x", 1.0),
        ("grave", "grave", 1.0),
        ("gravecomb.case", "gravecomb.case", 1.0),
        ("gravecomb", "gravecomb", 1.0),
        ("greater.case", "greater.case", 1.5),
        ("greater", "greater", 1.5),
        ("greaterequal.case", "greaterequal.case", 1.5),
        ("greaterequal", "greaterequal", 1.5),
        ("gstroke", "x", 1.0),
        ("guarani", "guarani", 1.6),
        ("guillemetleft.case", "guillemetleft.case", 1.2),
        ("guillemetleft", "guillemetleft", 1.2),
        ("guillemetright.case", "guillemetright.case", 1.2),
        ("guillemetright", "guillemetright", 1.2),
        ("guilsinglleft.case", "guilsinglleft.case", 1.2),
        ("guilsinglleft", "guilsinglleft", 1.2),
        ("guilsinglright.case", "guilsinglright.case", 1.2),
        ("guilsinglright", "guilsinglright", 1.2),
        ("h.sc", "h.sc", 1.1),
        ("H", "H", 1.25),
        ("h", "x", 1.0),
        ("haabkhasian-cy.sc", "h.sc", 1.1),
        ("Haabkhasian-cy", "H", 1.25),
        ("haabkhasian-cy", "x", 1.0),
        ("hahook-cy.sc", "h.sc", 1.1),
        ("Hahook-cy", "H", 1.25),
        ("hahook-cy", "x", 1.0),
        ("hairspace", "hairspace", 1.5),
        ("hardsign-cy.sc", "h.sc", 1.1),
        ("Hardsign-cy", "H", 1.25),
        ("hardsign-cy", "x", 1.0),
        ("hbar.sc", "h.sc", 1.1),
        ("Hbar", "H", 1.25),
        ("hook_cy.part", "hook_cy.part", 1.0),
        ("hookabovecomb.case", "hookabovecomb.case", 1.0),
        ("hookabovecomb", "hookabovecomb", 1.0),
        ("horizontalbar", "horizontalbar", 1.0),
        ("horncomb.case", "horncomb.case", 1.0),
        ("horncomb", "horncomb", 1.0),
        ("hryvnia", "hryvnia", 1.6),
        ("hungarumlaut", "hungarumlaut", 1.0),
        ("hungarumlautcomb.case", "hungarumlautcomb.case", 1.0),
        ("hungarumlautcomb", "hungarumlautcomb", 1.0),
        ("hyphen.case", "hyphen.case", 1.0),
        ("hyphen", "hyphen", 1.0),
        ("hyphentwo", "hyphentwo", 1.0),
        ("i.sc", "h.sc", 1.1),
        ("I.uc", "H", 1.25),
        ("I", "H", 1.25),
        ("ia-cy.sc", "h.sc", 1.1),
        ("Ia-cy", "H", 1.25),
        ("ia-cy", "x", 1.0),
        ("idotless", "x", 1.0),
        ("Ii-cy.loclBGR", "H", 1.25),
        ("ii-cy.sc", "h.sc", 1.1),
        ("Ii-cy", "H", 1.25),
        ("IJ", "H", 1.25),
        ("infinity.case", "infinity.case", 1.5),
        ("infinity", "infinity", 1.5),
        ("integral", "integral", 1.5),
        ("iu-cy.loclBGR", "x", 1.0),
        ("iu-cy.sc", "h.sc", 1.1),
        ("Iu-cy", "H", 1.25),
        ("iu-cy", "x", 1.0),
        ("izhitsa-cy.sc", "h.sc", 1.1),
        ("Izhitsa-cy", "H", 1.25),
        ("izhitsa-cy", "x", 1.0),
        ("j.sc", "h.sc", 1.1),
        ("J", "H", 1.25),
        ("jdotless", "x", 1.0),
        ("je-cy.sc", "h.sc", 1.1),
        ("k.sc", "h.sc", 1.1),
        ("K", "H", 1.25),
        ("k", "x", 1.0),
        ("ka-cy.sc", "h.sc", 1.1),
        ("Ka-cy", "H", 1.25),
        ("ka-cy", "x", 1.0),
        ("kabashkir-cy.sc", "h.sc", 1.1),
        ("Kabashkir-cy", "H", 1.25),
        ("kabashkir-cy", "x", 1.0),
        ("kahook-cy.sc", "h.sc", 1.1),
        ("Kahook-cy", "H", 1.25),
        ("kahook-cy", "x", 1.0),
        ("kaverticalstroke-cy.sc", "h.sc", 1.1),
        ("Kaverticalstroke-cy", "H", 1.25),
        ("kaverticalstroke-cy", "x", 1.0),
        ("kgreenlandic", "x", 1.0),
        ("kip", "kip", 1.6),
        ("komide-cy.sc", "h.sc", 1.1),
        ("Komide-cy", "H", 1.25),
        ("komidje-cy.sc", "h.sc", 1.1),
        ("Komidje-cy", "H", 1.25),
        ("komidje-cy", "x", 1.0),
        ("komidzje-cy.sc", "h.sc", 1.1),
        ("Komidzje-cy", "H", 1.25),
        ("komidzje-cy", "x", 1.0),
        ("komilje-cy.sc", "h.sc", 1.1),
        ("Komilje-cy", "H", 1.25),
        ("komilje-cy", "x", 1.0),
        ("kominje-cy.sc", "h.sc", 1.1),
        ("Kominje-cy", "H", 1.25),
        ("kominje-cy", "x", 1.0),
        ("komisje-cy.sc", "h.sc", 1.1),
        ("Komisje-cy", "H", 1.25),
        ("komisje-cy", "x", 1.0),
        ("komitje-cy.sc", "h.sc", 1.1),
        ("Komitje-cy", "H", 1.25),
        ("komitje-cy", "x", 1.0),
        ("komizje-cy.sc", "h.sc", 1.1),
        ("Komizje-cy", "H", 1.25),
        ("komizje-cy", "x", 1.0),
        ("l.sc", "h.sc", 1.1),
        ("L", "H", 1.25),
        ("l", "x", 1.0),
        ("lcaron", "x", 1.0),
        ("ldot.sc", "h.sc", 1.1),
        ("leftanglebracket-math", "leftanglebracket-math", 1.2),
        ("leftArrow", "leftArrow", 1.5),
        ("leftRightArrow", "leftRightArrow", 1.5),
        ("lefttooth_part.", "lefttooth_part.", 1.0),
        ("less.case", "less.case", 1.5),
        ("less", "less", 1.5),
        ("lessequal.case", "lessequal.case", 1.5),
        ("lessequal", "lessequal", 1.5),
        ("lira", "lira", 1.6),
        ("liraTurkish", "liraTurkish", 1.6),
        ("literSign", "literSign", 1.5),
        ("LJ", "H", 1.25),
        ("lje-cy.sc", "h.sc", 1.1),
        ("Lje-cy", "H", 1.25),
        ("lje-cy", "x", 1.0),
        ("logicalnot.case", "logicalnot.case", 1.5),
        ("logicalnot", "logicalnot", 1.5),
        ("longbar_part.", "longbar_part.", 1.0),
        ("lozenge", "lozenge", 1.5),
        ("lslash", "x", 1.0),
        ("m.sc", "h.sc", 1.1),
        ("M", "H", 1.25),
        ("m", "x", 1.0),
        ("macron", "macron", 1.0),
        ("macronbelowcomb.case", "macronbelowcomb.case", 1.0),
        ("macronbelowcomb", "macronbelowcomb", 1.0),
        ("macroncomb.case", "macroncomb.case", 1.0),
        ("macroncomb", "macroncomb", 1.0),
        ("manat", "manat", 1.6),
        ("mediumBlackSquare", "mediumBlackSquare", 1.5),
        ("micro", "micro", 1.5),
        ("minus.case", "minus.case", 1.5),
        ("minus", "minus", 1.5),
        ("minute", "minute", 1.5),
        ("mu", "x", 1.0),
        ("multiply.case", "multiply.case", 1.5),
        ("multiply", "multiply", 1.5),
        ("n.sc", "h.sc", 1.1),
        ("N", "H", 1.25),
        ("n", "x", 1.0),
        ("naira", "naira", 1.6),
        ("narrownbspace", "narrownbspace", 1.4),
        ("nbspace", "nbspace", 1.0),
        ("nine.dnom", "nine.dnom", 0.8),
        ("nine.lf", "one", 1.2),
        ("nine.osf", "zero.osf", 1.2),
        ("nine.sc", "nine.sc", 0.8),
        ("nine.tf", "one", 1.2),
        ("nine.tosf", "one", 1.2),
        ("nine", "one", 1.2),
        ("nje-cy.sc", "h.sc", 1.1),
        ("Nje-cy", "H", 1.25),
        ("nje-cy", "x", 1.0),
        ("northEastArrow", "northEastArrow", 1.5),
        ("northWestArrow", "northWestArrow", 1.5),
        ("notequal.case", "notequal.case", 1.5),
        ("notequal", "notequal", 1.5),
        ("numbersign", "numbersign", 1.4),
        ("numero", "numero", 1.5),
        ("o.sc", "h.sc", 1.1),
        ("O", "H", 1.25),
        ("o", "x", 1.0),
        ("obarred-cy.sc", "h.sc", 1.1),
        ("Obarred-cy", "H", 1.25),
        ("obarred-cy", "x", 1.0),
        ("odotbelow", "x", 1.0),
        ("oe.sc", "h.sc", 1.1),
        ("OE", "H", 1.25),
        ("oe", "x", 1.0),
        ("ogonek", "ogonek", 1.0),
        ("ogonekcomb.case", "ogonekcomb.case", 1.0),
        ("ogonekcomb", "ogonekcomb", 1.0),
        ("ohornacute.sc", "h.sc", 1.1),
        ("ohorndotbelow.sc", "h.sc", 1.1),
        ("ohorngrave.sc", "h.sc", 1.1),
        ("ohornhookabove.sc", "h.sc", 1.1),
        ("ohorntilde.sc", "h.sc", 1.1),
        ("omega.sc", "h.sc", 1.1),
        ("Omega", "H", 1.25),
        ("one.dnom", "one.dnom", 0.8),
        ("one.lf", "one", 1.2),
        ("one.osf", "zero.osf", 1.2),
        ("one.sc", "one.sc", 0.8),
        ("one.tf", "one", 1.2),
        ("one.tosf", "one", 1.2),
        ("one", "one", 1.2),
        ("ordfeminine", "ordfeminine", 1.0),
        ("ordmasculine", "ordmasculine", 1.0),
        ("p.sc", "h.sc", 1.1),
        ("P", "H", 1.25),
        ("p", "x", 1.0),
        ("palochka-cy.sc", "h.sc", 1.1),
        ("paragraph", "paragraph", 1.5),
        ("parenleft.case", "parenleft.case", 1.2),
        ("parenleft.sc", "parenleft.sc", 1.0),
        ("parenleft", "parenleft", 1.2),
        ("parenright.case", "parenright.case", 1.2),
        ("parenright.sc", "parenright.sc", 1.0),
        ("parenright", "parenright", 1.2),
        ("partialdiff", "partialdiff", 1.5),
        ("pe-cy.loclBGR", "x", 1.0),
        ("pe-cy.sc", "h.sc", 1.1),
        ("Pe-cy", "H", 1.25),
        ("pe-cy", "x", 1.0),
        ("pemiddlehook-cy.sc", "h.sc", 1.1),
        ("Pemiddlehook-cy", "H", 1.25),
        ("pemiddlehook-cy", "x", 1.0),
        ("percent", "percent", 1.5),
        ("period", "period", 1.4),
        ("periodcentered.case", "periodcentered.case", 1.4),
        ("periodcentered.loclCAT.case", "periodcentered.loclCAT.case", 1.4),
        ("periodcentered.loclCAT", "periodcentered.loclCAT", 1.4),
        ("periodcentered", "periodcentered", 1.4),
        ("perthousand", "perthousand", 1.5),
        ("peseta", "peseta", 1.6),
        ("peso", "peso", 1.6),
        ("pi.sc", "h.sc", 1.1),
        ("pi", "x", 1.0),
        ("plus.case", "plus.case", 1.5),
        ("plus", "plus", 1.5),
        ("plusminus.case", "plusminus.case", 1.5),
        ("plusminus", "plusminus", 1.5),
        ("primemod", "primemod", 1.0),
        ("product", "product", 1.5),
        ("ptick_part.", "ptick_part.", 1.0),
        ("published", "published", 1.5),
        ("punctuationspace", "punctuationspace", 1.25),
        ("q", "x", 1.0),
        ("qtail.case", "qtail.case", 1.0),
        ("qtail.sc", "qtail.sc", 1.0),
        ("question.sc", "question.sc", 1.0),
        ("question.ss01.sc", "question.ss01.sc", 1.0),
        ("question.ss01", "question.ss01", 1.4),
        ("question", "question", 1.4),
        ("questiondown.case", "questiondown.case", 1.4),
        ("questiondown.sc", "questiondown.sc", 1.0),
        ("questiondown.ss01.case", "questiondown.ss01.case", 1.4),
        ("questiondown.ss01.sc", "questiondown.ss01.sc", 1.0),
        ("questiondown.ss01", "questiondown.ss01", 1.4),
        ("questiondown", "questiondown", 1.4),
        ("quotedbl", "quotedbl", 1.2),
        ("quotedblbase.ss01", "quotedblbase.ss01", 1.2),
        ("quotedblbase", "quotedblbase", 1.2),
        ("quotedblleft.ss01", "quotedblleft.ss01", 1.2),
        ("quotedblleft", "quotedblleft", 1.2),
        ("quotedblright.ss01", "quotedblright.ss01", 1.2),
        ("quotedblright", "quotedblright", 1.2),
        ("quoteleft.ss01", "quoteleft.ss01", 1.2),
        ("quoteleft", "quoteleft", 1.2),
        ("quotereversed.ss01", "quotereversed.ss01", 1.2),
        ("quotereversed", "quotereversed", 1.2),
        ("quoteright.ss01", "quoteright.ss01", 1.2),
        ("quoteright", "quoteright", 1.2),
        ("quotesinglbase.ss01", "quotesinglbase.ss01", 1.2),
        ("quotesinglbase", "quotesinglbase", 1.2),
        ("quotesingle", "quotesingle", 1.2),
        ("r.sc", "h.sc", 1.1),
        ("R", "H", 1.25),
        ("r", "x", 1.0),
        ("radical", "radical", 1.5),
        ("ratio.case", "ratio.case", 1.5),
        ("ratio", "ratio", 1.5),
        ("registered", "registered", 1.5),
        ("reversedze-cy.sc", "h.sc", 1.1),
        ("Reversedze-cy", "H", 1.25),
        ("reversedze-cy", "x", 1.0),
        ("rightanglebracket-math", "rightanglebracket-math", 1.2),
        ("rightArrow", "rightArrow", 1.5),
        ("righttooth_part.", "righttooth_part.", 1.0),
        ("ring", "ring", 1.0),
        ("ringcomb.case", "ringcomb.case", 1.0),
        ("ringcomb", "ringcomb", 1.0),
        ("ringhalfleft", "ringhalfleft", 1.0),
        ("ringhalfright", "ringhalfright", 1.0),
        ("rtail_part.", "rtail_part.", 1.0),
        ("ruble", "ruble", 1.6),
        ("rupeeIndian", "rupeeIndian", 1.6),
        ("s.sc", "h.sc", 1.1),
        ("S", "H", 1.25),
        ("s", "x", 1.0),
        ("Schwa-cy", "H", 1.25),
        ("schwa.sc", "h.sc", 1.1),
        ("Schwa", "H", 1.25),
        ("schwa", "x", 1.0),
        ("second", "second", 1.5),
        ("secondtonechinese", "secondtonechinese", 1.0),
        ("section", "section", 1.5),
        ("semicolon", "semicolon", 1.4),
        ("semisoftsign-cy.sc", "h.sc", 1.1),
        ("Semisoftsign-cy", "H", 1.25),
        ("semisoftsign-cy", "x", 1.0),
        ("servicemark", "servicemark", 1.5),
        ("seven.dnom", "seven.dnom", 0.8),
        ("seven.lf", "one", 1.2),
        ("seven.osf", "zero.osf", 1.2),
        ("seven.sc", "seven.sc", 0.8),
        ("seven.tf", "one", 1.2),
        ("seven.tosf", "one", 1.2),
        ("seven", "one", 1.2),
        ("sha-cy.loclBGR", "x", 1.0),
        ("sha-cy.sc", "h.sc", 1.1),
        ("Sha-cy", "H", 1.25),
        ("sha-cy", "x", 1.0),
        ("sheqel", "sheqel", 1.6),
        ("shha-cy.sc", "h.sc", 1.1),
        ("Shha-cy", "H", 1.25),
        ("shha-cy", "x", 1.0),
        ("Shhadescender-cy", "H", 1.25),
        ("six.dnom", "six.dnom", 0.8),
        ("six.lf", "one", 1.2),
        ("six.osf", "zero.osf", 1.2),
        ("six.sc", "six.sc", 0.8),
        ("six.tf", "one", 1.2),
        ("six.tosf", "one", 1.2),
        ("six", "one", 1.2),
        ("sixperemspace", "sixperemspace", 1.6),
        ("slash.case", "slash.case", 1.0),
        ("slash", "slash", 1.0),
        ("slashlongcomb", "slashlongcomb", 1.0),
        ("slashocomb.case", "slashocomb.case", 1.0),
        ("slashocomb.sc", "slashocomb.sc", 1.0),
        ("slashocomb", "slashocomb", 1.0),
        ("slashshortcomb", "slashshortcomb", 1.0),
        ("softhyphen.case", "softhyphen.case", 1.0),
        ("softhyphen", "softhyphen", 1.0),
        ("softsign-cy.sc", "h.sc", 1.1),
        ("Softsign-cy", "H", 1.25),
        ("softsign-cy", "x", 1.0),
        ("southEastArrow", "southEastArrow", 1.5),
        ("southWestArrow", "southWestArrow", 1.5),
        ("space", "space", 1.1),
        ("sterling", "sterling", 1.6),
        ("strokelongcomb", "strokelongcomb", 1.0),
        ("strokeshortcomb", "strokeshortcomb", 1.0),
        ("summation", "summation", 1.5),
        ("t.sc", "h.sc", 1.1),
        ("T", "H", 1.25),
        ("t", "x", 1.0),
        ("tcaron", "x", 1.0),
        ("te-cy.loclBGR", "x", 1.0),
        ("te-cy", "x", 1.0),
        ("tenge", "tenge", 1.6),
        ("tetse-cy.sc", "h.sc", 1.1),
        ("Tetse-cy", "H", 1.25),
        ("thinspace", "thinspace", 1.0),
        ("thorn.sc", "h.sc", 1.1),
        ("Thorn", "H", 1.25),
        ("thorn", "x", 1.0),
        ("three.dnom", "three.dnom", 0.8),
        ("three.lf", "one", 1.2),
        ("three.osf", "zero.osf", 1.2),
        ("three.sc", "three.sc", 0.8),
        ("three.tf", "one", 1.2),
        ("three.tosf", "one", 1.2),
        ("three", "one", 1.2),
        ("threeperemspace", "threeperemspace", 1.1),
        ("tilde", "tilde", 1.0),
        ("tildecomb.case", "tildecomb.case", 1.0),
        ("tildecomb", "tildecomb", 1.0),
        ("trademark", "trademark", 1.5),
        ("tse-cy.sc", "h.sc", 1.1),
        ("Tse-cy", "H", 1.25),
        ("tshe-cy.sc", "h.sc", 1.1),
        ("Tshe-cy", "H", 1.25),
        ("tugrik", "tugrik", 1.6),
        ("two.dnom", "two.dnom", 0.8),
        ("two.lf", "one", 1.2),
        ("two.osf", "zero.osf", 1.2),
        ("two.sc", "two.sc", 0.8),
        ("two.tf", "one", 1.2),
        ("two.tosf", "one", 1.2),
        ("two", "one", 1.2),
        ("u-cy.sc", "h.sc", 1.1),
        ("U-cy", "H", 1.25),
        ("u.sc", "h.sc", 1.1),
        ("U", "H", 1.25),
        ("u", "x", 1.0),
        ("uhorn.sc", "h.sc", 1.1),
        ("Uhorn", "H", 1.25),
        ("uhorn", "x", 1.0),
        ("uhornacute.sc", "h.sc", 1.1),
        ("uhornacute", "x", 1.0),
        ("uhorndotbelow", "x", 1.0),
        ("uhorngrave", "x", 1.0),
        ("uhornhookabove", "x", 1.0),
        ("uhorntilde", "x", 1.0),
        ("underscore", "underscore", 1.0),
        ("upArrow", "upArrow", 1.5),
        ("upDownArrow", "upDownArrow", 1.5),
        ("ustrait-cy", "x", 1.0),
        ("v.sc", "h.sc", 1.1),
        ("V", "H", 1.25),
        ("v", "x", 1.0),
        ("ve-cy.loclBGR", "x", 1.0),
        ("ve-cy", "x", 1.0),
        ("vertbarlc_part.", "vertbarlc_part.", 1.0),
        ("verticallinelowmod", "verticallinelowmod", 1.0),
        ("verticallinemod", "verticallinemod", 1.0),
        ("w.sc", "h.sc", 1.1),
        ("W", "H", 1.25),
        ("w", "x", 1.0),
        ("we-cy.sc", "h.sc", 1.1),
        ("We-cy", "H", 1.25),
        ("won", "won", 1.6),
        ("x.sc", "h.sc", 1.1),
        ("X", "H", 1.25),
        ("x", "x", 1.0),
        ("y.sc", "h.sc", 1.1),
        ("Y", "H", 1.25),
        ("y", "x", 1.0),
        ("yat-cy.sc", "h.sc", 1.1),
        ("Yat-cy", "H", 1.25),
        ("yat-cy", "x", 1.0),
        ("yen", "yen", 1.6),
        ("yeru-cy", "x", 1.0),
        ("yusbig-cy.sc", "h.sc", 1.1),
        ("Yusbig-cy", "H", 1.25),
        ("yusbig-cy", "x", 1.0),
        ("z.sc", "h.sc", 1.1),
        ("Z", "H", 1.25),
        ("z", "x", 1.0),
        ("ze-cy.loclBGR", "x", 1.0),
        ("ze-cy.sc", "h.sc", 1.1),
        ("Ze-cy", "H", 1.25),
        ("ze-cy", "x", 1.0),
        ("zero.dnom", "zero.dnom", 0.8),
        ("zero.lf.zero", "one", 1.2),
        ("zero.lf", "one", 1.2),
        ("zero.osf.zero", "zero.osf", 1.2),
        ("zero.osf", "zero.osf", 1.2),
        ("zero.sc.zero", "zero.sc.zero", 0.8),
        ("zero.sc", "zero.sc", 0.8),
        ("zero.tf.zero", "one", 1.2),
        ("zero.tf", "one", 1.2),
        ("zero.tosf.zero", "one", 1.2),
        ("zero.tosf", "one", 1.2),
        ("zero.zero", "one", 1.2),
        ("zero", "one", 1.2),
        ("zerowidthjoiner", "zerowidthjoiner", 1.4),
        ("zerowidthnonjoiner", "zerowidthnonjoiner", 1.1),
        ("zerowidthspace", "zerowidthspace", 1.25),
        ("zhe-cy.loclBGR", "x", 1.0),
        ("zhe-cy.sc", "h.sc", 1.1),
        ("Zhe-cy", "H", 1.25),
        ("zhe-cy", "x", 1.0),
    ):
        glyph_orig = ufo_orig[glyph]
        if glyph_orig.components:
            continue  # composites in Merriweather are complicated: change other glyphs, ...
            # would have to space them first and then all dependents.
        glyph_ref_orig = ufo_orig[glyph_ref]

        o.width = glyph_orig.width
        o.newWidth = 0.0
        o.factor = factor
        o.spaceMain(glyph_orig, glyph_ref_orig, ufo_orig)

        glyph_rspc = ufo_rspc[glyph]
        try:
            left_should = glyph_rspc.getLeftMargin(ufo_rspc)
            if left_should is None:
                continue  # skip emquad, etc.
            left_is = glyph_orig.getLeftMargin(ufo_orig)
            assert left_is is not None
            assert round(left_is) == pytest.approx(round(left_should), abs=1), glyph

            right_should = glyph_rspc.getRightMargin(ufo_rspc)
            assert right_should is not None
            right_is = glyph_orig.getRightMargin(ufo_orig)
            assert right_is is not None
            assert round(right_is) == pytest.approx(round(right_should), abs=1), glyph
        except:
            output_ufo = ufoLib2.Font()
            output_ufo.info.unitsPerEm = ufo_orig.info.unitsPerEm
            output_ufo.layers.defaultLayer.insertGlyph(glyph_orig, name="respaced")
            output_ufo.layers.defaultLayer.insertGlyph(glyph_rspc, name="comparison")
            output_ufo.save("/tmp/test.ufo", overwrite=True)
            raise
