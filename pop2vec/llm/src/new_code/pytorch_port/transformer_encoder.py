"""
TransformerEncoder – vanilla-PyTorch replacement for the LightningModule.
Now includes Lightning-style hook methods (training_step, validation_step,
train_epoch_start, on_train_epoch_end, on_validation_epoch_end) so that
diffing against the original Lightning code is almost line-for-line.
"""
from __future__ import annotations
import logging, pickle, torch, torch.nn as nn, torch.nn.functional as F
import torchmetrics
from pop2vec.llm.src.transformer.transformer import (
    CLS_DecoderS, Transformer, MaskedLanguageModel
)
from pop2vec.llm.src.transformer.transformer_utils import *

logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """Transformer with Masked Language Model"""

    # ──────────────────────────────────────────────────────────────────────
    # init / buffers / metrics
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self, hparams: dict[str, any]):
        super().__init__()
        self.hparams = hparams
        torch.manual_seed(self.hparams["seed"])

        # 1. ENCODER
        self.transformer = Transformer(self.hparams)

        # 2. DECODER BLOCK (MLM + CLS)
        self.task = self.hparams["training_task"]
        logger.info("Training task: %s", self.task)
        if "mlm" not in self.task:
            raise NotImplementedError("Only 'mlm' tasks are supported.")

        # constants (registered as buffers like in Lightning)
        self.register_buffer("cls_w", torch.tensor(0.2))
        self.register_buffer("mlm_w", torch.tensor(0.8))
        self.register_buffer("cls_a", torch.tensor([1 / 0.9, 1 / 0.1, 1 / 0.1]))

        self.num_outputs = self.hparams["vocab_size"]
        self.mlm_decoder = MaskedLanguageModel(
            self.hparams, self.transformer.embedding, act="tanh"
        )
        self.cls_decoder = CLS_DecoderS(self.hparams)

        # loss fns
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.cls_loss_fn = nn.CrossEntropyLoss(
            weight=self.cls_a, label_smoothing=0.1
        )

        # metrics
        top_k = 5 if self.num_outputs == self.hparams["vocab_size"] else 1
        metric_kw = dict(
            threshold=0.2,
            num_classes=self.num_outputs,
            average="macro",
            ignore_index=0,
            top_k=top_k,
            task="multiclass",
        )
        self.train_accuracy = torchmetrics.Accuracy(**metric_kw)
        self.train_precision = torchmetrics.Precision(**metric_kw)
        self.train_recall = torchmetrics.Recall(**metric_kw)
        self.train_f1 = torchmetrics.F1Score(**metric_kw)

        cls_kw = dict(threshold=0.5, num_classes=3, average="macro", task="multiclass")
        self.train_cls_acc = torchmetrics.Accuracy(**cls_kw)
        self.train_cls_f1 = torchmetrics.F1Score(**cls_kw)

        # same metric set for validation
        self.val_accuracy = torchmetrics.Accuracy(**metric_kw)
        self.val_precision = torchmetrics.Precision(**metric_kw)
        self.val_recall = torchmetrics.Recall(**metric_kw)
        self.val_f1 = torchmetrics.F1Score(**metric_kw)
        self.val_cls_acc = torchmetrics.Accuracy(**cls_kw)
        self.val_cls_f1 = torchmetrics.F1Score(**cls_kw)

        # accumulators for logging on_*_epoch_end
        self.reset_accumulators()

    # ──────────────────────────────────────────────────────────────────────
    # utility
    # ──────────────────────────────────────────────────────────────────────
    def reset_accumulators(self):
        self.total_train_loss = 0.0
        self.total_train_mlm = 0.0
        self.total_train_cls = 0.0
        self.total_val_loss = 0.0
        self.total_val_mlm = 0.0
        self.total_val_cls = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────────────────────────────
    def forward(self, batch):
        """Forward pass"""
        ## 1. ENCODER INPUT
        predicted = self.transformer(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long()
        )
        ## 2. MASKED LANGUAGE MODEL
        mlm_pred = self.mlm_decoder(predicted, batch)
        ## 3. CLS TASK
        cls_pred  = self.cls_decoder(predicted[:,0])
        return mlm_pred, cls_pred

    # ──────────────────────────────────────────────────────────────────────
    # Lightning-style hooks
    # ──────────────────────────────────────────────────────────────────────
    def training_step(self, batch):
        """Equivalent of LightningModule.training_step"""
        ## 1. ENCODER-DECODER
        mlm_preds, cls_preds = self(batch)
        ## 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        cls_targs = batch["target_cls"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        cls_loss = self.cls_loss(cls_preds, target = cls_targs)
        loss = self.cls_w * cls_loss + self.mlm_w * mlm_loss

        # update metrics
        with torch.no_grad():
            mlm_sm = F.softmax(mlm_preds, dim=-1).permute(0, 2, 1)
            cls_sm = F.softmax(cls_preds, dim=-1)
            self.train_accuracy.update(mlm_sm, mlm_targs)
            self.train_precision.update(mlm_sm, mlm_targs)
            self.train_recall.update(mlm_sm, mlm_targs)
            self.train_f1.update(mlm_sm, mlm_targs)
            self.train_cls_acc.update(cls_sm, cls_targs)
            self.train_cls_f1.update(cls_sm, cls_targs)

        # accumulators for pretty epoch-end logging
        self.total_train_loss += loss.item()
        self.total_train_mlm += mlm_loss.item()
        self.total_train_cls += cls_loss.item()
        return loss, mlm_loss, cls_loss

    def train_epoch_start(self, epoch: int):
        """Mimic Lightning's on_train_epoch_start (used for reseeding)."""
        torch.manual_seed(self.hparams["seed"] + epoch)

    def on_train_epoch_end(self):
        """Print epoch aggregates (trainer writes CSV)."""
        logger.info(
            "Total training loss/MLM/CLS this epoch: %.4f / %.4f / %.4f",
            self.total_train_loss, self.total_train_mlm, self.total_train_cls,
        )
        self.total_train_loss = self.total_train_mlm = self.total_train_cls = 0.0

        if self.hparams.attention_type == "performer":
            self.transformer.redraw_projection_matrix(-1)

    def validation_step(self, batch):
        """Equivalent of LightningModule.validation_step"""
        ## 1. ENCODER-DECODER
        mlm_preds, cls_preds = self(batch)
        ## 2. LOSS
        mlm_targs = batch["target_tokens"].long()
        cls_targs = batch["target_cls"].long()
        mlm_loss = self.mlm_loss(mlm_preds.permute(0, 2, 1), target=mlm_targs)
        cls_loss = self.cls_loss(cls_preds, target = cls_targs)

        loss = self.cls_w * cls_loss + self.mlm_w * mlm_loss

        # update metrics
        with torch.no_grad():
            mlm_sm = F.softmax(mlm_preds, dim=-1).permute(0, 2, 1)
            cls_sm = F.softmax(cls_preds, dim=-1)
            self.val_accuracy.update(mlm_sm, mlm_targs)
            self.val_precision.update(mlm_sm, mlm_targs)
            self.val_recall.update(mlm_sm, mlm_targs)
            self.val_f1.update(mlm_sm, mlm_targs)
            self.val_cls_acc.update(cls_sm, cls_targs)
            self.val_cls_f1.update(cls_sm, cls_targs)

        self.total_val_loss += loss.item()
        self.total_val_mlm += mlm_loss.item()
        self.total_val_cls += cls_loss.item()
        return loss, mlm_loss, cls_loss

    def on_validation_epoch_end(self):
        logger.info(
            "Total validation loss/MLM/CLS this epoch: %.4f / %.4f / %.4f",
            self.total_val_loss, self.total_val_mlm, self.total_val_cls,
        )
        self.total_val_loss = self.total_val_mlm = self.total_val_cls = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # optimiser / scheduler fabric
    # ──────────────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        """Reproduces Lightning’s AdamW + OneCycleLR schedule."""

        no_decay = [
            "bias",
            "norm",
            "age",
            "abspos",
            "token",
            "decoder.g"
        ]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams['weight_decay'],
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams["learning_rate"],
            betas=(self.hparams["beta1"], self.hparams["beta2"]),
            eps=self.hparams["epsilon"],
        )

        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=self.hparams["learning_rate"],
            epochs=self.hparams["epochs"],
            steps_per_epoch=self.hparams["steps_per_epoch"],
            pct_start=0.05,
            three_phase=False,
            max_momentum=self.hparams["beta1"],
            div_factor=30,
        )
        return optimizer, scheduler

    # ──────────────────────────────────────────────────────────────────────
    # static helper (unchanged)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def load_lookup(path: str):
        """Load token/index look-up dictionaries saved as a pickle."""
        with open(path, "rb") as f:
            indx2token, token2indx = pickle.load(f)
        return indx2token, token2indx