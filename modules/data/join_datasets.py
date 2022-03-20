# -*- coding: utf-8 -*-

import json
import sys

from tqdm import tqdm


def main():

    _, *in_files, out_file = sys.argv

    out_json = {
        "indices": [],
        "relations": [],
        "texts": [],
    }

    for in_file in tqdm(in_files):

        with open(in_file) as f:
            in_json = json.load(f)

        for key in out_json.keys():
            out_json[key].extend(in_json[key])

    with open(out_file, "w") as f:
        json.dump(out_json, f)


if __name__ == "__main__":
    main()
