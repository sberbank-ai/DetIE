# coding: utf-8
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from config.hydra import cleanup_hydra
from modules.model import models


class DetIETripletExtractor:
    def __init__(self, cfg=None, model_name=None, best_ckpt_path=None, best_hparams_path=None, most_common=False):
        # super().__new__(cls)
        self.most_common = most_common
        if cfg is not None:
            self.model = getattr(models, cfg.model.name).load_from_checkpoint(
                checkpoint_path=cfg.model.best_ckpt_path,
                hparams_file=cfg.model.best_hparams_path,
                scheduler_cfg=cfg.scheduler,
            )
        else:
            self.model = getattr(models, model_name).load_from_checkpoint(
                checkpoint_path=best_ckpt_path,
                hparams_file=best_hparams_path,
                scheduler_cfg=DictConfig({"name": "ExponentialLR", "gamma": 1}),
            )

    def __call__(self, text: str):
        triplets = self.model.predict([text], most_common=self.most_common)[0]
        return [triplet for rel_id, triplet in triplets]


def prepare_detie_ollie_format(sentences_raw_file_path, save_file_path, cfg, save_file=True, most_common=False):
    print("Loading triplet extractor from checkpoint...")
    try:
        mte = DetIETripletExtractor(cfg, most_common=most_common)
    except Exception as e:
        print(e, "moving on...")
        mte = DetIETripletExtractor(
            model_name=cfg.model.name,
            best_ckpt_path=cfg.model.best_ckpt_path,
            best_hparams_path=cfg.model.best_hparams_path,
            most_common=most_common,
        )

    print(
        mte(
            "To say of what is that it is not , or of what is not that it is , is false , "
            "while to say of what is that it is , and of what is not that it is not , is true ."
        )
    )

    # quit()

    with open(sentences_raw_file_path, "r+", encoding="utf-8") as rf:
        raw_sentences = list(set([line.strip() for line in rf if line.strip()]))

    # confidence	arg1	rel	arg2	enabler	attribution	text	pattern	dependencies
    future_dataframe = {
        "confidence": 1.0,  # we don't do confidence
        "arg1": [],
        "rel": [],
        "arg2": [],
        "enabler": None,  # we don't do that
        "attribution": None,  # we don't do that
        "text": [],
        "pattern": None,  # we don't do that
        "dependencies": None,  # we don't do that
    }

    for raw_sentence in tqdm(raw_sentences):
        oie_spans = mte(raw_sentence)

        for s, r, o in oie_spans:
            future_dataframe["arg1"].append(s)
            future_dataframe["arg2"].append(o)
            future_dataframe["rel"].append(r)
            future_dataframe["text"].append(raw_sentence)

    result_dataframe = pd.DataFrame(future_dataframe)

    if save_file:
        result_dataframe.to_csv(save_file_path, index=False, sep="\t")

    return result_dataframe


# @cleanup_hydra
@hydra.main("../../../../config", "config.yaml")
def main(cfg):
    VERSION = 276
    cfg.model.best_version = VERSION
    cfg.model.best_ckpt_path = "../../../../" + cfg.model.best_ckpt_path
    cfg.model.best_hparams_path = "../../../../" + cfg.model.best_hparams_path

    for most_common in [True, False]:
        for split in ["test"]:
            # test_set = f"../lsoie-carb_formatted/data/{split}.txt"
            # save_path = f"../lsoie-carb_formatted/system_outputs/{split}/bertie{cfg.model.best_version}mc{most_common}_output.txt"
            test_set = f"data/{split}.txt"
            save_path = f"system_outputs/{split}/detie{cfg.model.best_version}_output.txt"

            try:
                prepare_detie_ollie_format(test_set, save_path, cfg, most_common=most_common)
            except RuntimeError as rte:
                print(rte)
                print(dir(models))

                for model_name in dir(models):
                    if "Triplet" not in model_name:
                        continue
                    try:
                        cfg.model.name = model_name
                        prepare_detie_ollie_format(test_set, save_path, cfg, most_common=most_common)
                    except Exception as e:
                        print(e)
                        print(f"This '{model_name}' is the wrong model name, moving on with {VERSION}")
                        # raise e


if __name__ == "__main__":
    main()
