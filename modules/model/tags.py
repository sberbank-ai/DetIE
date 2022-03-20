# coding: utf-8

from collections import defaultdict

UD_DEPREL_RAW = """
        acl: clausal modifier of noun (adnominal clause)
        acl:relcl: relative clause modifier
        advcl: adverbial clause modifier
        advmod: adverbial modifier
        advmod:emph: emphasizing word, intensifier
        advmod:lmod: locative adverbial modifier
        amod: adjectival modifier
        appos: appositional modifier
        aux: auxiliary
        aux:pass: passive auxiliary
        case: case marking
        cc: coordinating conjunction
        cc:preconj: preconjunct
        ccomp: clausal complement
        clf: classifier
        compound: compound
        compound:lvc: light verb construction
        compound:prt: phrasal verb particle
        compound:redup: reduplicated compounds
        compound:svc: serial verb compounds
        conj: conjunct
        cop: copula
        csubj: clausal subject
        csubj:pass: clausal passive subject
        dep: unspecified dependency
        det: determiner
        det:numgov: pronominal quantifier governing the case of the noun
        det:nummod: pronominal quantifier agreeing in case with the noun
        det:poss: possessive determiner
        discourse: discourse element
        dislocated: dislocated elements
        expl: expletive
        expl:impers: impersonal expletive
        expl:pass: reflexive pronoun used in reflexive passive
        expl:pv: reflexive clitic with an inherently reflexive verb
        fixed: fixed multiword expression
        flat: flat multiword expression
        flat:foreign: foreign words
        flat:name: names
        goeswith: goes with
        iobj: indirect object
        list: list
        mark: marker
        nmod: nominal modifier
        nmod:poss: possessive nominal modifier
        nmod:tmod: temporal modifier
        nsubj: nominal subject
        nsubj:pass: passive nominal subject
        nummod: numeric modifier
        nummod:gov: numeric modifier governing the case of the noun
        obj: object
        obl: oblique nominal
        obl:agent: agent modifier
        obl:arg: oblique argument
        obl:lmod: locative modifier
        obl:tmod: temporal modifier
        orphan: orphan
        parataxis: parataxis
        punct: punctuation
        reparandum: overridden disfluency
        root: root
        vocative: vocative
        xcomp: open clausal complement"""

UD_DEPREL = [line.strip().split(": ")[0] for line in UD_DEPREL_RAW.split("\n") if line.strip()]
UD_DEPREL_IDS = defaultdict(lambda: 0, {dr: i for i, dr in enumerate(UD_DEPREL, start=1)})
INV_UD_DEPREL_IDS = defaultdict(lambda: "???", {v: k for k, v in UD_DEPREL_IDS.items()})

# https://universaldependencies.org/u/pos/
UPOS_TAGS_ALL = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

UPOS_TAGS_IDS = {tag: i for i, tag in enumerate(sorted(list(UPOS_TAGS_ALL.keys())), start=1)}
UPOS_TAGS_IDS = defaultdict(lambda: 0, UPOS_TAGS_IDS)
INV_UPOS_TAGS_IDS = defaultdict(lambda: "???", {v: k for k, v in UPOS_TAGS_IDS.items()})
