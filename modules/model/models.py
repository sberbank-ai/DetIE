# coding: utf-8
import io
from collections import OrderedDict
from typing import List, Optional

import pytorch_lightning as pl
import stanza
import torch
from lapsolver import solve_dense
from nltk.tokenize.treebank import TreebankWordDetokenizer
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertModel

from .apply import postprocess_adp, pprint_triplets, prediction2triples
from .dataloaders import SyntaxFeatures, make_syntax_features
from .feature_preparation import syntax_based_features_for_bpe
from .tags import UD_DEPREL, UPOS_TAGS_ALL


class TransposeLayer(nn.Module):
    """A helper class for transposing the tensor"""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, input: torch.Tensor):
        return input.transpose(*self.args)


class TripletsExtractor(pl.LightningModule):
    """Base class for all single-shot models extracting multiple triplets"""

    def __init__(self, model_cfg: DictConfig, opt_cfg: DictConfig, scheduler_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_cfg, self.opt_cfg = model_cfg, opt_cfg

        # Init side tools (stanza & detokenizer)
        self.lang = model_cfg.lang
        self.postprocess_adp = model_cfg.postprocess_adp
        self.use_syntax_features = model_cfg.use_syntax_features
        self.init_tools()

        self.seed = model_cfg.seed
        self.example_texts = list(model_cfg.viz_sentences)

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer, cache_dir=model_cfg.cache_dir)
        self.pretrained_encoder = AutoModelForMaskedLM.from_pretrained(
            model_cfg.pretrained_encoder, cache_dir=model_cfg.cache_dir
        ).base_model
        # self.pretrained_encoder = BertModel.from_pretrained(
        #     model_cfg.pretrained_encoder, cache_dir=model_cfg.cache_dir).base_model
        # self.tokenizer = BertTokenizerFast.from_pretrained(model_cfg.tokenizer,
        #                                                    cache_dir=model_cfg.cache_dir)

        in_size = self.pretrained_encoder.config.hidden_size
        hid_size = in_size
        out_dim = model_cfg.num_detections * model_cfg.n_classes

        if model_cfg.use_syntax_features:
            in_size = in_size + 2 * model_cfg.stanza_emb_size
            self.pos_emb = nn.Embedding(len(UPOS_TAGS_ALL) + 1, model_cfg.stanza_emb_size)
            self.deprel_emb = nn.Embedding(len(UD_DEPREL) + 1, model_cfg.stanza_emb_size)

        self.logits = nn.Sequential(
            nn.Linear(in_size, hid_size),
            # TransposeLayer(1, 2),
            # nn.Conv1d(model_cfg.pretrained_emb_size, hid_size, kernel_size=5, padding=2),
            # TransposeLayer(1, 2),
            # nn.ReLU(),
            nn.LayerNorm(hid_size),
            TransposeLayer(0, 1),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hid_size, model_cfg.n_classes, hid_size), num_layers=model_cfg.num_layers
            ),
            TransposeLayer(0, 1),
            nn.ReLU(),
            nn.Linear(hid_size, out_dim),
        )

        self.compute_crossentropy_with_logits = nn.CrossEntropyLoss(
            weight=torch.tensor(list(model_cfg.class_weights), dtype=torch.float), reduction="none"
        )

        name, opt_cfg = opt_cfg.name, dict(opt_cfg)
        del opt_cfg["name"]
        self.opt = getattr(torch.optim, name)(self.parameters(), **opt_cfg)

        name, scheduler_cfg = scheduler_cfg.name, dict(scheduler_cfg)
        del scheduler_cfg["name"]
        self.scheduler = getattr(torch.optim.lr_scheduler, name)(self.opt, **scheduler_cfg)

        self.metrics = {}
        for stage in ("train", "val", "test"):
            self.metrics[self.get_metric_name("f1_score", stage)] = pl.metrics.classification.f_beta.F1(
                model_cfg.n_classes, average="macro"
            )
            self.metrics[
                self.get_metric_name("precision", stage)
            ] = pl.metrics.classification.precision_recall.Precision(model_cfg.n_classes, average="macro")
            self.metrics[self.get_metric_name("recall", stage)] = pl.metrics.classification.precision_recall.Recall(
                model_cfg.n_classes, average="macro"
            )

    def init_tools(self, cache_dir: str = None):

        if self.postprocess_adp:
            self.detokenizer = TreebankWordDetokenizer()

        if self.postprocess_adp or self.use_syntax_features:
            if not cache_dir:
                cache_dir = self.model_cfg.cache_dir

            stanza.download(self.lang, model_dir=cache_dir)
            self.stanza_pipeline = stanza.Pipeline(lang=self.lang, processors="tokenize,mwt,pos", dir=cache_dir)

    @staticmethod
    def get_metric_name(name, stage, tag: str = ""):
        if not tag:
            return f"{stage}_{name}"
        return f"{stage}_{name}_{tag}"

    def _get_stage_metrics(self, stage):
        return {key: value for key, value in self.metrics.items() if key.startswith(stage)}

    def forward(self, encoder_inputs, syntax_features: Optional[SyntaxFeatures] = None):
        # BERT embeddings
        encoder_outputs = self.pretrained_encoder(**encoder_inputs)  # [batch_size, seq_len, encoder_hid_size]

        # last hidden state goes into the main head
        main_head_input = encoder_outputs.last_hidden_state

        if self.use_syntax_features:
            pos_emb = self.pos_emb(syntax_features.pos_tags)
            deprel_emb = self.deprel_emb(syntax_features.deprel_tags)
            main_head_input = torch.cat([main_head_input, pos_emb, deprel_emb], -1)

        # Filter everything that is below threshold
        logits = self.logits(main_head_input)  # [batch_size, seq_len, n_classes * num_detections]

        batch_size, seq_len, *_ = logits.shape
        logits = logits.view(
            *logits.shape[:-1], -1, self.model_cfg.n_classes
        )  # [batch_size, seq_len, num_detections, n_classes]

        # detection_conf = torch.sigmoid(logits[:, 0, :, :1])  # [batch_size, seq_len, num_detections]
        # mask = torch.any(detection_probas > self.model_cfg.min_detection_thresh, dim=1, keepdim=True)
        # # [batch_size, num_detections, n_classes]
        # mask = torch.all(mask, -1, keepdim=True)  # [batch_size, num_detections]

        # logits = torch.masked_select(logits, mask).reshape(batch_size, seq_len, -1, 2)
        # # [batch_size, seq_len, num_filtered_detections, n_classes]

        return logits

    def predict(self, texts: List[str], calc_confidence=False, **kwargs):
        if self.training:
            self.eval()

        if self.model_cfg.join_is:
            texts = [text + " [is] [of] [from]" for text in texts]

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            # added June 3rd for running on Russian
            truncation=True,
            max_length=self.model_cfg.hid_size,
        ).to(self.device)

        syntax_features = None
        if self.use_syntax_features:
            batch = [
                list(
                    map(
                        lambda items: [item if item is not None else 0 for item in items],
                        syntax_based_features_for_bpe(text, self.stanza_pipeline, self.tokenizer),
                    )
                )
                for text in texts
            ]
            syntax_features = make_syntax_features(*zip(*batch), device=self.device)

        offsets_mapping = tokenized["offset_mapping"].cpu().numpy()
        del tokenized["offset_mapping"]  # otherwise -- error
        special_tokens = set(self.tokenizer.special_tokens_map.values())

        with torch.no_grad():
            prediction = self.forward(tokenized, syntax_features)
            triplets = prediction2triples(
                prediction=prediction,
                texts=texts,
                offsets_mapping=offsets_mapping,
                tokenized=tokenized,
                # calc_confidence=calc_confidence,
                # tokenizer=self.tokenizer,
                # most_common=False
                # **kwargs
            )

        if self.postprocess_adp:
            postprocess_adp(triplets, self.stanza_pipeline, self.detokenizer)

        return triplets

    @staticmethod
    def _match_logits_with_labels(
        logits: torch.Tensor, labels: torch.Tensor, matching: str = "iou", disable_bg: bool = True, eps: float = 1e-8
    ):
        """
            Using the magic of ops research to assign the best labeling

        :param logits: predictions scores
        :param labels: true labels
        :return:
        """
        # normalizing predictions
        detached_probas = torch.softmax(logits, dim=-1).detach()

        effective_labels = labels
        if disable_bg:
            # removing background for IoU computation
            detached_probas = detached_probas[:, :, :, 1:]
            effective_labels = effective_labels[:, :, :, 1:]

        # m -- how many detectors we decided to have
        # n -- how many actual relations we are supposed to extract
        intersection = torch.einsum("ijmk,ijnk->imn", detached_probas, effective_labels.to(torch.float))
        sum_probas = torch.sum(detached_probas, dim=(1, -1)).unsqueeze(-1)
        sum_labels = torch.sum(effective_labels, dim=(1, -1)).unsqueeze(-2)
        if matching == "iou":
            # Inclusion–exclusion principle
            union = sum_probas + sum_labels - intersection
        elif matching == "dice":
            union = sum_probas + sum_labels
        elif matching == "dice_squared":
            sum_probas = torch.sum(detached_probas**2, dim=(1, -1)).unsqueeze(-1)
            union = sum_probas + sum_labels
        else:
            raise NotImplementedError

        # IoU scores for every pair of detection-vs-actual_relation for every element of the batch
        matching_metrics = intersection / (union + eps)  # [batch_size, num_filtered_detections, n_relations]

        matched_labels, labels = torch.zeros_like(logits, dtype=torch.int), labels.to(torch.int)
        # for every element of the batch
        batch_iou = []
        for i, iou_scores in enumerate(matching_metrics):
            pred_idx, y_idx = solve_dense((-iou_scores).cpu().numpy())
            matched_labels[i, :, pred_idx] = labels[i, :, y_idx]
            batch_iou.append(iou_scores[pred_idx, y_idx].mean())

        return logits, matched_labels, sum(batch_iou) / len(batch_iou)

    def compute_loss(self, batch, calc_metrics=True, stage=None):
        (
            encoder_inputs,
            labels_one_hot,
            syntax_features,
        ) = batch  # y.shape == [batch_size, seq_len, n_relations, n_classes]
        logits = self.forward(encoder_inputs, syntax_features)

        matched_logits, matched_labels_one_hot, avg_iou = self._match_logits_with_labels(
            logits, labels_one_hot, matching=self.model_cfg.matching, disable_bg=self.model_cfg.disable_bg
        )
        if self.model_cfg.focal_gamma != 0:
            matched_probas = torch.softmax(matched_logits, -1).max(-1)[0].view(-1)
        matched_logits = matched_logits.view(-1, matched_logits.size(-1))
        matched_labels = matched_labels_one_hot.argmax(-1).view(-1)

        loss = self.compute_crossentropy_with_logits(matched_logits, matched_labels)
        if self.model_cfg.focal_gamma != 0:
            loss *= (1 - matched_probas) ** self.model_cfg.focal_gamma
        loss = loss.mean()

        metrics = OrderedDict()
        if calc_metrics:
            assert stage

            metrics[self.get_metric_name("loss", stage)] = loss.item()
            metrics[self.get_metric_name("avg_iou", stage)] = avg_iou.item()

            matched_preds = matched_logits.argmax(-1)
            # metrics['f1_score'] = f1_score(matched_labels, matched_logits, average='macro')
            # metrics['precision_score'] = precision_score(matched_labels, matched_logits, average='macro')
            # metrics['recall_score'] = recall_score(matched_labels, matched_logits, average='macro')

            for metric_name, metric_fn in self._get_stage_metrics(stage).items():
                metrics[metric_name] = metric_fn(matched_preds.cpu(), matched_labels.cpu())

            # Class-wise metrics
            matched_labels, matched_preds = map(lambda x: x.cpu().numpy(), (matched_labels, matched_preds))
            for class_name, f1, precision, recall in zip(
                ("none", "source", "relation", "target"),
                f1_score(matched_labels, matched_preds, average=None),
                precision_score(matched_labels, matched_preds, average=None),
                recall_score(matched_labels, matched_preds, average=None),
            ):
                # labels_class_mask = matched_labels == class_id
                # total_predicted_class = (matched_logits == class_id).sum()
                # total_predicted_labels = (matched_logits == class_id).sum()
                #
                # intersection = (matched_labels == matched_logits).sum()
                # # metrics[f'{class_name}_iou'] =  / (matched_logits)

                metrics[self.get_metric_name(f"{class_name}_f1_score", stage)] = f1
                metrics[self.get_metric_name(f"{class_name}_precision_score", stage)] = precision
                metrics[self.get_metric_name(f"{class_name}_recall_score", stage)] = recall

        return logits, loss, metrics

    @staticmethod
    def _set_req_grad(module: nn.Module, value: bool):
        for param in module.parameters():
            param.requires_grad = value

    def on_epoch_start(self):
        if self.current_epoch == self.model_cfg.unfreeze_epoch:
            n_layers = len(self.pretrained_encoder.encoder.layer)
            print(
                f"Unfreezing layers starting after {n_layers - self.model_cfg.unfreeze_layers_from_top} "
                f"of {n_layers} on epoch {self.current_epoch}"
            )
            for layer in self.pretrained_encoder.encoder.layer[n_layers - self.model_cfg.unfreeze_layers_from_top :]:
                self._set_req_grad(layer, True)

    def on_fit_start(self):
        pl.seed_everything(self.seed)
        self._set_req_grad(self.pretrained_encoder, False)

    def _log_metrics(self, metrics: dict, postfix: str = "_batch"):
        for key, value in metrics.items():
            self.log(key + postfix, value)

    def _log_epoch_metrics(self, stage, postfix: str = "_epoch"):
        for metric_name, metric_fn in self._get_stage_metrics(stage).items():
            self.log(metric_name + postfix, metric_fn.compute(), on_step=False, on_epoch=True)

    def log_example_prediction(self):
        triplets = self.predict(self.example_texts)
        buf = io.StringIO()
        pprint_triplets(self.example_texts, triplets, end="  \n", file=buf)
        self.logger.experiment.add_text("predictions", buf.getvalue())

    def training_step(self, batch, batch_idx):
        _, loss, metrics = self.compute_loss(batch, stage="train")
        self.scheduler.step()
        self._log_metrics(metrics)
        return loss

    def on_train_epoch_end(self, *args):
        self._log_epoch_metrics("train")

    def validation_step(self, batch, batch_idx):
        _, _, metrics = self.compute_loss(batch, stage="val")
        self._log_metrics(metrics)

    def on_validation_epoch_end(self):
        self.log_example_prediction()
        self._log_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        _, _, metrics = self.compute_loss(batch, stage="test")
        self._log_metrics(metrics)

    def on_test_epoch_end(self):
        self.log_example_prediction()
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        return self.opt


class TripletsExtractorBERTOnly(TripletsExtractor):
    def __init__(self, model_cfg: DictConfig, opt_cfg: DictConfig, scheduler_cfg: DictConfig):
        super().__init__(model_cfg, opt_cfg, scheduler_cfg)
        self.logits = nn.Linear(
            self.pretrained_encoder.config.hidden_size, model_cfg.num_detections * model_cfg.n_classes
        )
