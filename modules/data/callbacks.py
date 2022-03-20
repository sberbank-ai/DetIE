# -*- coding: utf-8 -*-

from typing import List

import stanza

from modules.scripts.data.gpt2_perplexity import PretrainedLMScorer


class DefaultCallback:
    def __call__(self, triplet: List[str]):
        for item in triplet:
            if not item:
                return False
            if "category:" in item.lower():
                return False

        return True


class LMScoreCallback:
    scores = []

    def __init__(self, threshold: float, **scorer_kwargs):
        self.scorer = PretrainedLMScorer(**scorer_kwargs)
        self.threshold = threshold

    def __call__(self, triplet: List[str]):
        score = self.scorer.eval(" ".join(triplet))
        if score > self.threshold:
            return False
        self.scores.append(score)
        print(" ".join(triplet))
        return True


class StanzaCallback:
    def __init__(self, lang: str = "en"):
        self.lang = lang
        stanza.download(lang)
        self.tagger = stanza.Pipeline(lang=lang, processors="tokenize,mwt,pos")
        # if lang == 'en':
        #     self.tagger = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos')
        # elif lang == 'ru':
        #     self.tagger = stanza.Pipeline(lang=lang, processors='tokenize,pos')
        # else:
        #     raise ValueError("Unknown lang: " + lang)

    def __call__(self, triplet: List[str]):
        if self.lang == "ru":
            # TODO: remove this dirty hack
            return True
        doc = self.tagger(triplet[1])
        return "VBZ" in (word.xpos for sentence in doc.sentences for word in sentence.words)
