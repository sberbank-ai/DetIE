# coding: utf-8

import hydra
import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, random_split

from config.hydra_ext import cleanup_hydra
from modules.model import models
from modules.model.dataloaders import CollateFn, WikiDataset


@cleanup_hydra
@hydra.main("../../config", "config.yaml")
def main(cfg):
    transformers.logging.set_verbosity_error()

    # trainable model
    model_class = getattr(models, cfg.model.name)
    model = model_class(cfg.model, cfg.opt, cfg.scheduler)

    # reading `sentences.json` or the like

    def make_dataset(path: str):
        return WikiDataset(path, cfg.model.tokenizer, use_syntax_features=cfg.model.use_syntax_features)

    dataset = make_dataset(cfg.wikidata.lsoie_train_path)
    test_dataset = make_dataset(cfg.wikidata.lsoie_test_path)

    train_collate_fn = CollateFn(
        model.tokenizer,
        use_syntax_features=cfg.model.use_syntax_features,
        word_dropout=cfg.model.word_dropout,
        multiple_spans=cfg.wikidata.multiple_spans,
    )

    eval_collate_fn = CollateFn(
        model.tokenizer,
        use_syntax_features=cfg.model.use_syntax_features,
        word_dropout=0,
        multiple_spans=cfg.wikidata.multiple_spans,
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg.model.batch_size, collate_fn=eval_collate_fn, shuffle=False)

    if cfg.model.validate_on_test:
        train = dataset
        train_loader = DataLoader(train, batch_size=cfg.model.batch_size, collate_fn=train_collate_fn, shuffle=True)
        val_loader = test_loader
    else:
        val_size = int(len(dataset) * cfg.model.val_fraction)
        train, val = random_split(dataset, [len(dataset) - val_size, val_size])
        train_loader = DataLoader(train, batch_size=cfg.model.batch_size, collate_fn=train_collate_fn, shuffle=True)
        val_loader = DataLoader(val, batch_size=cfg.model.batch_size, collate_fn=eval_collate_fn, shuffle=False)

    profiler = "advanced" if cfg.model.profile else None
    logger = loggers.TensorBoardLogger("./results/logs/")

    checkpoint_callback = ModelCheckpoint(
        filename="best",
        save_top_k=1,
        verbose=True,
        monitor=model_class.get_metric_name("f1_score", "val", "epoch"),
        mode="max",
    )

    def make_trainer(n_epochs: int):
        return pl.Trainer(
            max_epochs=n_epochs,
            logger=logger,
            gpus=cfg.model.gpus * torch.cuda.is_available(),
            profiler=profiler,
            log_every_n_steps=cfg.model.log_every_n_steps,
            flush_logs_every_n_steps=cfg.model.log_every_n_steps,
            callbacks=[checkpoint_callback],
        )

    trainer = make_trainer(min(cfg.model.max_epochs, cfg.model.syntetic_data_after_epoch))
    trainer.fit(model, train_loader, val_loader)

    # Adding syntetic data
    if cfg.model.max_epochs > cfg.model.syntetic_data_after_epoch:
        syntetic_dataset = make_dataset(cfg.wikidata.generation.sentences_path)
        train_loader = DataLoader(
            ConcatDataset([train, syntetic_dataset]),
            batch_size=cfg.model.batch_size,
            collate_fn=train_collate_fn,
            shuffle=True,
        )

        trainer = make_trainer(cfg.model.max_epochs - cfg.model.syntetic_data_after_epoch)
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
