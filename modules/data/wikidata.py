# -*- coding: utf-8 -*-

import dataclasses
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Union

from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.linked_data_interface import (LdiResponseNotOk,
                                             get_entity_dict_from_api)
from qwikidata.sparql import return_sparql_query_results
from tqdm import tqdm


@dataclass
class SemanticTriplet:
    property_id: str
    source: List[str] = field(default_factory=list)
    relation: List[str] = field(default_factory=list)
    target: List[str] = field(default_factory=list)

    def sample(self) -> List[str]:
        return [random.choice(entities) for entities in (self.source, self.relation, self.target)]


def fetch_triplets_by_property(
    property_id: str,
    add_entity_aliases: bool = True,
    add_relation_aliases: bool = True,
    limit: Union[int, str] = "",
    lang: str = "en",
) -> List[SemanticTriplet]:
    prop = WikidataProperty(get_entity_dict_from_api(property_id))
    if limit != "":
        limit = f"LIMIT {limit}"

    sparql_query = (
        """
            SELECT ?source ?target 
            WHERE 
            {
              ?source
            """
        + f" wdt:{property_id}"
        + """ ?target.
                  SERVICE wikibase:label { bd:serviceParam wikibase:language """
        f""""{lang}". """
        + """}
            }
            ORDER BY STRLEN(STR(?source)) STRLEN(STR(?target))
            """
        + f"{limit}"
    )

    query_results = return_sparql_query_results(sparql_query)

    triplets = []
    for query_item in query_results["results"]["bindings"]:
        triplet = SemanticTriplet(property_id)

        for triplet_attr in ("source", "target"):
            # Parse item from url
            url = query_item[triplet_attr]["value"]
            match_obj = re.match(".*/(Q\d+)$", url)
            if match_obj is not None:
                item_id = match_obj.group(1)
            else:
                break

            item = WikidataItem(get_entity_dict_from_api(item_id))

            # Get attribute to fill
            item_aliases = getattr(triplet, triplet_attr)
            item_aliases.append(item.get_label(lang))

            if add_entity_aliases:
                item_aliases.extend(item.get_aliases(lang))
        else:
            triplet.relation.append(prop.get_label(lang))

            if add_relation_aliases:
                triplet.relation.extend(prop.get_aliases(lang))

            triplets.append(triplet)

    return triplets


def download_triplets(
    out_dir: str,
    min_property_id: int = 1,
    max_property_id: int = 1000,
    n_threads: int = 10,
    sparql_limit: int = 1000,
    log: logging.Logger = logging.getLogger(__name__),
    lang: str = "en",
) -> None:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    properties = [f"P{property_id}" for property_id in range(min_property_id, max_property_id)]

    skipped = 0
    progress_bar = tqdm(total=len(properties), position=0)

    def crawl(property_id):
        nonlocal skipped
        filepath = os.path.join(out_dir, f"{property_id}.json")

        try:
            if not os.path.exists(filepath):
                try:
                    prop = WikidataProperty(get_entity_dict_from_api(property_id))
                except LdiResponseNotOk as exc:
                    log.info(exc.__str__())
                    skipped += 1
                    progress_bar.update(1)
                    return
                prop.get_aliases()
                triplets = fetch_triplets_by_property(property_id, limit=sparql_limit, lang=lang)

                with open(filepath, "w") as fp:
                    json.dump([dataclasses.asdict(triplet) for triplet in triplets], fp)
            else:
                skipped += 1
        except Exception as exc:
            log.info(exc.__str__())
        finally:
            progress_bar.update(1)
            progress_bar.set_description_str(f"skipped {skipped} out of")

    log.info(f"Starting with {n_threads} threads...")
    executor = ThreadPoolExecutor(max_workers=n_threads)
    for property_id in properties:
        executor.submit(crawl, property_id)
    executor.shutdown(wait=True)


def load_triplets(triplets_dir: str, callback: Callable = lambda: True) -> List[SemanticTriplet]:
    triplets = []
    for filepath in tqdm(list(Path(triplets_dir).glob("*.json")), "Loading triplets"):
        with open(filepath) as fp:
            triplets_json = json.load(fp)
        # if not triplets_json:
        #     continue

        # peek_triplet = SemanticTriplet(**triplets_json[0])
        # if not peek_triplet.relation:
        #     continue
        #
        # if any([callback(['', relation, '']) for relation in peek_triplet.relation]):
        for triplet in triplets_json:
            triplet = SemanticTriplet(**triplet)
            triplets.append(triplet)

    return triplets


def filter_triplets_by_relation(in_dir: str, out_dir: str, callback: Callable = lambda: True) -> List[SemanticTriplet]:

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    triplets = []

    for filepath in tqdm(list(Path(in_dir).glob("*.json")), "Loading triplets"):
        with open(filepath) as fp:
            triplets_json = json.load(fp)
        if not triplets_json:
            continue

        peek_triplet = SemanticTriplet(**triplets_json[0])

        if not peek_triplet.relation:
            continue

        relation_aliases = list(filter(lambda relation: callback(["", relation, ""]), peek_triplet.relation))

        if relation_aliases:
            for item in triplets_json:
                item["relation"] = relation_aliases
            with open(os.path.join(out_dir, filepath.name), "w") as fp:
                json.dump(triplets_json, fp)

    return triplets


# def filter_triplets(triplets: List[SemanticTriplet]) -> List[SemanticTriplet]:
#     callback = StanzaCallback()
#
#     def is_valid_relation(relation: str):
#         return callback(['', relation, ''])
#
#     filtered = []
#
#     for triplet in tqdm(triplets, desc='Filtering'):
#         triplet.relation = list(filter(is_valid_relation, triplet.relation))
#         if triplet.relation:
#             filtered.append(triplet)
#
#     return filtered
