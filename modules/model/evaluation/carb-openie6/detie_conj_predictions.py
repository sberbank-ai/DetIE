# coding: utf-8
from collections import defaultdict

import hydra
from detie_predict import prepare_detie_ollie_format

from config.hydra_ext import cleanup_hydra
from modules.model import models


def get_conj_map(fp="systems_output/carb_test.txt.conj"):
    sentences = defaultdict(lambda: [])

    with open(fp, "r+", encoding="utf-8") as rf:
        first_sentence = True
        sentence = None

        for line in rf:

            line = line.strip()
            if not line:
                first_sentence = True
            elif first_sentence:
                first_sentence = False
                sentence = line
                sentences[sentence].append(sentence)
            else:
                sentences[sentence].append(line)

    ext2sentence = {v: k for k in sentences for v in sentences[k]}

    return sentences, ext2sentence


@cleanup_hydra
@hydra.main("../../../../config", "config.yaml")
def main(cfg):

    VERSION = 243
    cfg.model.best_version = VERSION
    cfg.model.best_ckpt_path = "../../../../" + cfg.model.best_ckpt_path
    cfg.model.best_hparams_path = "../../../../" + cfg.model.best_hparams_path

    for split in ["test"]:
        test_set = f"data/carb_sentences.txt"
        conj_data = f"data/carb_test-openie6.txt.conj"
        sentence2reduced, reduced2sentece = get_conj_map(conj_data)
        save_path = f"systems_output/detie{cfg.model.best_version}conj_output.txt"

        try:
            results_data = prepare_detie_ollie_format(conj_data, save_path, cfg, save_file=False)
        except RuntimeError as rte:
            print(rte)
            print(dir(models))

            for model_name in dir(models):
                if "Triplet" not in model_name:
                    continue
                try:
                    cfg.model.name = model_name
                    results_data = prepare_detie_ollie_format(conj_data, save_path, cfg, save_file=False)
                except Exception as e:
                    print(e)
                    print(f"This '{model_name}' is the wrong model name, moving on with {VERSION}")
                    # raise e

        results_data["text"] = results_data["text"].map(lambda x: reduced2sentece.get(x, x))
        results_data.drop(results_data.index, inplace=False).to_csv(save_path, index=None, sep="\t")

        for line in open(test_set, "r+", encoding="utf-8"):
            line = line.strip()
            if line:
                extractions = results_data[results_data["text"] == line].drop_duplicates(subset=["arg1", "rel", "arg2"])
                # print(results_data[results_data["text"]==line])
                # print(results_data[results_data["text"]==line].drop_duplicates(subset=["arg1", "rel", "arg2"]))
                # print("[%s]" % line)
                # print(reduced2sentece[line])
                # print(sentence2reduced[line])
                if extractions.shape[0] > 0:
                    extractions.to_csv(save_path, mode="a", header=False, index=None, sep="\t")


if __name__ == "__main__":
    main()
