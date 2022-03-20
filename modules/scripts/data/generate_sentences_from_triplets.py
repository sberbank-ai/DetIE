# -*- coding: utf-8 -*-

import json
import logging
import random

import hydra
from omegaconf import DictConfig

from config.hydra import cleanup_hydra
from modules.data.callbacks import DefaultCallback, StanzaCallback
from modules.data.sentence_generation import TripletGenerator
from modules.data.wikidata import filter_triplets_by_relation


@cleanup_hydra
@hydra.main("../../../config", "config.yaml")
def main(cfg: DictConfig):
    cfg = cfg.wikidata
    random.seed(cfg)

    if cfg.preprocessing.refilter:
        filter_triplets_by_relation(
            cfg.crawling.triplets_dir, cfg.preprocessing.triplets_filtered_dir, StanzaCallback(cfg.lang)
        )

    triplet_generator = TripletGenerator(cfg.preprocessing.triplets_filtered_dir, cfg.lang)

    cfg = cfg.generation
    log = logging.getLogger(__name__)

    callbacks = [
        DefaultCallback(),
        # StanzaCallback(),
        # LMScoreCallback(cfg.lm_threshold, stride=cfg.lm_stride),
    ]
    texts = triplet_generator.generate_texts(cfg.n_texts, callbacks)

    # print(sum(callbacks[-1].scores) / len(callbacks[-1].scores))

    with open(cfg.sentences_path, "w") as fp:
        json.dump(texts, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
