# -*- coding: utf-8 -*-

import logging

import hydra
from omegaconf import DictConfig

from config.hydra import cleanup_hydra
from modules.data.wikidata import download_triplets


@cleanup_hydra
@hydra.main("../../../config", "config.yaml")
def main(cfg: DictConfig):
    lang = cfg.wikidata.lang
    cfg = cfg.wikidata.crawling
    log = logging.getLogger(__name__)

    download_triplets(
        out_dir=cfg.triplets_dir,
        min_property_id=cfg.min_property_id,
        max_property_id=cfg.max_property_id,
        n_threads=cfg.n_threads,
        sparql_limit=cfg.sparql_limit,
        log=log,
        lang=lang,
    )


if __name__ == "__main__":
    main()
