# -*- coding: utf-8 -*-

import random
from itertools import islice
from random import choice, choices, randint
from typing import Iterable, List, Tuple

conjunctions = ("and", "or", "but", "as well as", "meanwhile")
ru_conjunctions = ("и", "или", ", но", ", а также", "в тоже время")

fillers = ("",)
ru_fillers = ("", "это", "является")


class TemplateCreator:
    def __init__(self, lang: str = "en"):

        self.fillers = fillers

        if lang == "en":
            conj = conjunctions
            self.fillers = fillers
        elif lang == "ru":
            conj = ru_conjunctions
            self.fillers = ru_fillers
        else:
            conj = (",",)

        self.templates = {
            lambda: "{}.": 0.1,
            lambda: "{} " + choice(conj) + " {}.": 0.2,
            lambda: ", ".join(["{}" for _ in range(randint(3, 6))]) + ".": 0.35,
            lambda: ". ".join(["{}" for _ in range(randint(2, 10))]) + ".": 0.35,
        }

        self.template_funcs = list(self.templates.keys())
        self.template_probas = list(self.templates.values())

    def _sample_template(self) -> str:
        return choices(self.template_funcs, self.template_probas)[0].__call__()

    def _fill_template(self, item, use_filler, base_pos) -> Tuple[str, List[int]]:

        result = ""

        if use_filler:
            filler = random.choice(self.fillers)

            if filler:
                result += filler + " "

        start = base_pos + len(result)
        result += item + " "
        stop = base_pos + len(result) - 1

        return result, [start, stop]

    def fill_one(self, triplets: Iterable) -> (List[str], List[List[str]]):
        template = self._sample_template()
        split = template.split("{}")

        indices, curr_pos = [], 0
        triplets = [*islice(triplets, len(split) - 1)]
        filled_template = ""

        for i, triplet in enumerate(triplets):

            indices.append([])
            curr_pos += len(split[i])
            filled_template += split[i]

            for j, item in enumerate(triplet):
                source, limits = self._fill_template(item, j == 1, curr_pos)  # if this is predicate
                filled_template += source
                indices[-1].append([limits])
                assert filled_template[indices[-1][-1][0][0] : indices[-1][-1][0][1]] == item
                curr_pos = limits[-1] + 1

            curr_pos -= 1
            filled_template = filled_template[:-1]

        filled_template += split[-1]

        return filled_template, triplets, indices
