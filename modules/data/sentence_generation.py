# -*- coding: utf-8 -*-

import random
from collections import defaultdict
from typing import Callable, Dict, List

import stanza
from nltk.tokenize.treebank import TreebankWordDetokenizer
from stanza import Pipeline
from tqdm import tqdm

from ..model.apply import split_adp_right
from .templates import TemplateCreator
from .wikidata import SemanticTriplet, load_triplets


class TripletGenerator:
    def __init__(self, triplets_dir: str, lang: str = "en", preprocess_triplet: bool = False):

        self.triplets: List[SemanticTriplet] = load_triplets(triplets_dir)
        self.lang = lang
        self.preprocess_triplet = preprocess_triplet
        self.detokenizer = TreebankWordDetokenizer()

        if self.preprocess_triplet:
            stanza.download(self.lang)
            self.pipeline = Pipeline(self.lang, processors="tokenize,mwt,pos")

    def sample_triplets(self, callbacks: List[Callable] = tuple()) -> List[List[str]]:

        while True:
            triplet = random.choice(self.triplets).sample()

            while not all(callback(triplet) for callback in callbacks):
                triplet = random.choice(self.triplets).sample()

            source, relation, target = triplet

            if self.preprocess_triplet:
                doc = self.pipeline(relation)
                relation, adp = split_adp_right(doc, self.detokenizer)
                target = self.detokenizer.detokenize([adp, target])

            yield source, relation, target

    def generate_texts(self, n_iter: int, callbacks: List[Callable] = tuple()) -> Dict[str, List[str]]:

        tc = TemplateCreator(self.lang)
        out_dict = defaultdict(list)

        for _ in tqdm(range(n_iter)):
            template, relations, indices = tc.fill_one(self.sample_triplets(callbacks))
            out_dict["texts"].append(template)
            out_dict["relations"].append(relations)
            out_dict["indices"].append(indices)
        return out_dict
