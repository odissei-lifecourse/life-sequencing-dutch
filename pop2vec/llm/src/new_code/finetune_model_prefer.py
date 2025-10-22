# finetune_module.py
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import functional as MF

from pop2vec.llm.src.transformer.transformer import  AttentionDecoder, CLS_Decoder, AttentionDecoderP, Deep_Decoder, Transformer


logger = logging.getLogger(__name__)


class TransformerFT(pl.LightningModule):
    """
    Fine‑tuning module that adds a lightweight classification head on top of a
    frozen / partially‑frozen Transformer encoder.

    * Works with the new `FineTuneLazyDataset` (expects ``input_ids``,
      ``padding_mask``, ``target`` [+ optional ``sequence_id``]).
    * Handles **classification** (binary / multi‑class) and **regression**.
    * Optional: layer‑wise LR decay, class‑imbalance weighting,
      pooled‑attention vs pure CLS token, etc.
    """

    # ------------------------------------------------------------------ #
    # constructor & helpers                                               #
    # ------------------------------------------------------------------ #
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()

        # P.L. 2.x: save_hyperparameters accepts a dict directly
        self.save_hyperparameters(hparams)

        # 1. BACKBONE ENCODER ------------------------------------------------
        self._init_encoder()

        # 2. TASK‑SPECIFIC HEAD ---------------------------------------------
        self._init_decoder()

        # 3. LOSS ------------------------------------------------------------
        self._init_loss()

        # 4. METRICS ---------------------------------------------------------
        self._init_metrics()

        # misc helpers
        self.train_preds: List[torch.Tensor] = []
        self.train_targs: List[torch.Tensor] = []
        self.last_logged_step: int = -1

        # For finding the best threshold in binary classifications 
        self.register_buffer("best_threshold", torch.tensor(0.50))   # default
        self._val_probs: list[torch.Tensor] = []
        self._val_targs: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # encoder / decoder                                                   #
    # ------------------------------------------------------------------ #
    def _init_encoder(self) -> None:
        """Instantiate encoder and (optionally) load a pre‑trained checkpoint."""
        self.transformer = Transformer(self.hparams)
        logger.info(
            f"Embedding sample after creating transformer with random weights: "
            f"token 1,0 --> {self.transformer.embedding.token.weight[1, 0].detach()}, "
            f"token 198,0 --> {self.transformer.embedding.token.weight[198, 0].detach()}",
        )

        ckpt = self.hparams["pretrained_model_path"]
        if ckpt != 'RANDOM' and ckpt != None:
            logger.info("Loading encoder weights from %s", ckpt)
            state = torch.load(ckpt, map_location="cpu")["state_dict"]
            # strip "transformer." prefix from keys
            state = {k.replace("transformer.", "", 1): v for k, v in state.items()
                     if k.startswith("transformer.")}
            missing, unexpected = self.transformer.load_state_dict(state, strict=False)
            if missing:
                logger.warning("Missing keys while loading encoder: %s", missing)
            if unexpected:
                logger.warning("Unexpected keys in checkpoint: %s", unexpected)
            logger.info(
                f"Embedding sample after loading transformer from checkpoint: "
                f"token 1,0 --> {self.transformer.embedding.token.weight[1, 0].detach()}, "
                f"token 198,0 --> {self.transformer.embedding.token.weight[198, 0].detach()}",
            )
        elif ckpt == 'RANDOM':
            logger.info("Starting from random‑initialised encoder")
        else:
            raise ValueError(
                f"pretrained_model_path in hparams cannot be None. Got {self.hparams['pretrained_model_path']}"
            )
        # freeze (optional) positional embeddings etc.
        if self.hparams["freeze_positions"]:
            for n, p in self.transformer.embedding.named_parameters():
                if any(k in n for k in ("age", "abspos", "segment")):
                    p.requires_grad_(False)

    def _init_decoder(self) -> None:
        """Attach a cheap head on top of the encoder output."""
        self.num_outputs = self.hparams["num_targets"]

        if self.hparams["pooled"]:
            logger.info("Using **pooled attention** representation")
            self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
            self.encoder_forward = self.transformer.forward_finetuning
        else:
            logger.info("Using **CLS token** representation")
            self.decoder = CLS_Decoder(self.hparams, num_outputs=self.num_outputs)
            self.encoder_forward = self.transformer.forward_finetuning_cls

    # ------------------------------------------------------------------ #
    # loss & metrics                                                     #
    # ------------------------------------------------------------------ #
    def _init_loss(self) -> None:
        loss_type = self.hparams["loss_type"].lower()
        task_type = self.hparams["task_type"]

        if task_type == "classification":
            weight = self.hparams["loss_weights"]
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(weight) if weight is not None else None
            )
        elif task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported (task, loss) combo: {task_type}/{loss_type}")

    def _init_metrics(self) -> None:
        """Prepare on‑epoch metrics (Lightning handles device transfer)."""
        task = self.hparams["task_type"]
        if task == "classification":
            num_classes = self.hparams["num_targets"]
            threshold = 0.5 if num_classes == 2 else None
            self.train_acc = torchmetrics.Accuracy(
                task="binary" if num_classes == 2 else "multiclass",
                num_classes=num_classes, threshold=threshold, average="macro"
            )
            self.train_f1 = torchmetrics.F1Score(
                task="binary" if num_classes == 2 else "multiclass",
                num_classes=num_classes, threshold=threshold, average="macro"
            )
            self.val_acc = self.train_acc.clone()
            self.val_f1 = self.train_f1.clone()
            self.test_acc = self.train_acc.clone()
            self.test_f1  = self.train_f1.clone()
        else:  # regression
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.val_mae = self.train_mae.clone()
            self.test_mae = self.train_mae.clone()

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Plain forward (encoder → decoder)."""
        hidden = self.encoder_forward(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long(),
        )
        if self.hparams["pooled"]:
            out = self.decoder(hidden, mask=batch["padding_mask"].long())
        else:
            out = self.decoder(hidden)
        return out  # logits (classification) or predictions (regression)

    # ------------------------------------------------------------------ #
    # training / validation loops                                        #
    # ------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------ #
    # helpers for logging                                                #
    # ------------------------------------------------------------------ #
    def _log_binary_metrics(
        self,
        stage: str,
        probs: torch.Tensor,          # P(class = 1) – shape (B,)
        target: torch.Tensor,         # shape (B,)
    ) -> None:
        """Binary‑classification extras for the validation stage."""
        # default 0.5 threshold
        pred05 = (probs > 0.5).long()

        self.log(f"{stage}_acc-50_epoch",
                 getattr(self, f"{stage}_acc")(pred05, target),
                 on_step=False, on_epoch=True, sync_dist=True)


        
        # sweep thresholds 0.10 ...0.90
        for t in range(10, 100, 10):
            thr     = t / 100.0
            pred_t  = (probs > thr).long()
            self.log(f"{stage}_pos-preds-{t}_epoch",
                     pred_t.float().mean(),
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_binary-f1-{t}_epoch",
                     torchmetrics.functional.f1_score(
                         pred_t, target.long(), task="binary"
                     ),
                     on_step=False, on_epoch=True, sync_dist=True)

            if stage == 'train':
                self.log(f"{stage}_pos-preds-{t}_step",
                         pred_t.float().mean(),
                         on_step=True, on_epoch=False, sync_dist=False)
                self.log(f"{stage}_binary-f1_{t}_step",
                         torchmetrics.functional.f1_score(
                             pred_t, target.long(), task="binary"
                         ),
                         on_step=True, on_epoch=False, sync_dist=False)


    def _log_multiclass_metrics(
        self,
        stage: str,
        logits: torch.Tensor,         # shape (B, C)
        target: torch.Tensor,         # shape (B,)
    ) -> None:
        """Multi‑class accuracy/ F1."""
        preds = logits.argmax(dim=1)
        self.log(f"{stage}_acc_epoch",
                 getattr(self, f"{stage}_acc")(preds, target),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc_step",
                 getattr(self, f"{stage}_acc")(preds, target),
                 on_step=True, on_epoch=False, sync_dist=False)
        self.log(f"{stage}_f1_epoch",
                 getattr(self, f"{stage}_f1")(preds, target),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_f1_step",
                 getattr(self, f"{stage}_f1")(preds, target),
                 on_step=True, on_epoch=False, sync_dist=False)

    # ------------------------------------------------------------------ #
    # shared training / val / test step                                  #
    # ------------------------------------------------------------------ #
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        logits  = self(batch)
        target  = batch["target"]
        loss    = self.criterion(logits, target if target.ndim == 1 else target.squeeze())

        # basic loss logging
        self.log(
            f"{stage}_loss_epoch", loss,
             on_step=False, 
             on_epoch=True,
             prog_bar=True,
             sync_dist=True
        )
        if stage == "train":
            self.log(
                f"{stage}_loss_step", loss,
                 on_step=True, 
                 on_epoch=False,
                 prog_bar=True,
                 sync_dist=False
            )    
            self.log(
                "lr-emb", 
                self._opt.param_groups[self.embedding_group_idx]["lr"], 
                on_step=True, 
                on_epoch=True,
            )    
            self.log(
                "lr-decoder", 
                self._opt.param_groups[self.decoder_group_idx]["lr"], 
                on_step=True, 
                on_epoch=True,
            )

        # delegate metric logging to compact helpers
        if (stage == 'val' or ((self.global_step + 1) % (self.trainer.log_every_n_steps) == 0)):
            
            if self.hparams["task_type"] == "classification":
                if self.hparams["num_targets"] == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]   # P(class=1)
                    self._log_binary_metrics(stage, probs, target)
                    if stage == 'val':
                        self._val_probs.append(probs.detach())
                        self._val_targs.append(target.detach())
                else:
                    self._log_multiclass_metrics(stage, logits, target)
            else:  # regression
                metric = getattr(self, f"{stage}_mae")
                self.log(f"{stage}_mae_epoch", metric(logits, target),
                         on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                if stage == 'train':
                    self.log(f"{stage}_mae_step", metric(logits, target),
                             on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)

        return loss


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._val_probs.clear()
        self._val_targs.clear()

    @staticmethod
    def _best_f1_threshold_torch(p: torch.Tensor, y: torch.Tensor):
        """Return (thr*, f1*) using the exact sweep."""
        # sort descending by probability
        p_sorted, idx = torch.sort(p, descending=True)
        y_sorted = y[idx]

        tp = torch.cumsum(y_sorted == 1, 0)
        fp = torch.cumsum(y_sorted == 0, 0)
        fn = tp[-1] - tp

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)

        best_idx = torch.argmax(f1)
        return p_sorted[best_idx], f1[best_idx]


    def on_validation_epoch_end(self) -> None:
        # --------------- gather everything across GPUs ---------------
        if not self._val_probs:       # safety check
            return

        if not (self.hparams["task_type"] == 'classification' and self.hparams['num_targets'] == 2):
            return

        p = torch.cat(self._val_probs, 0)
        y = torch.cat(self._val_targs, 0)
        # gathers p and y from all gpus to all gpus
        p = self.all_gather(p).flatten()
        y = self.all_gather(y).flatten()

        # --------------- exact threshold search ----------------------
        # Exact same calculation is done all gpus
        thr_star, f1_star = self._best_f1_threshold_torch(p.cpu(), y.cpu())

        # keep the threshold as a buffer (same dtype/device on every rank)
        self.best_threshold.fill_(thr_star.to(self.best_threshold.device))

        # --------------- log so the checkpoint callback can monitor --
        # only gpu 0 does logging. I am lazy and scared to change code that 
        # is nowhere near a bottleneck
        self.log("val_binary-f1-best_epoch",
                 f1_star,
                 prog_bar=True, sync_dist=False)

        self.log("val_binary-best-thr_epoch",
                 thr_star,
                 prog_bar=True, sync_dist=False)


    # ------------------------------------------------------------------ #
    # optimiser                                                          #
    # ------------------------------------------------------------------ #
    def debug_optimizer(
        self, lr, wt_decay_enc, wt_decay_dec, layer_decay, beta1, beta2, eps, optim_type
    ):
        logger.debug("==========================")
        logger.debug(f"learning rate = {lr}")
        logger.debug(f"weight_decay_enc = {wt_decay_enc}")
        logger.debug(f"weight_decay_dec = {wt_decay_dec}")
        logger.debug(f"layer_decay = {layer_decay}")
        logger.debug(f"beta1 = {beta1}")
        logger.debug(f"beta2 = {beta2}")
        logger.debug(f"eps = {eps}")
        logger.debug(f"optim_type = {optim_type}")
    
        logger.debug("PARAMETERS IN OPTIMIZER")
        logger.debug("\tGROUP 1:")
        for n, p in self.named_parameters():
            if "embedding" in n:
                logger.debug(f"{'Learnable' if p.requires_grad else 'Frozen'}")
                logger.debug("\t\t %s", n)
        logger.debug("\tGROUP 2:")
        for i in range(0, self.hparams.n_encoders):
            for n, p in self.named_parameters():
                if "encoders.%s." % i in n:
                    logger.debug(f"{'Learnable' if p.requires_grad else 'Frozen'}")
                    logger.debug("\t\t %s", n)
        logger.debug("\tGROUP 3:")
        for n, p in self.named_parameters():
            if "decoder" in n:
                logger.debug(f"{'Learnable' if p.requires_grad else 'Frozen'}")
                logger.debug("\t\t %s", n)
   
    def configure_optimizers(self):
        """AdamW with optional layer‑wise LR decay + scheduler."""
        lr            = self.hparams["learning_rate"]
        wt_decay_enc  = self.hparams["weight_decay_enc"]
        wt_decay_dec  = self.hparams.get("weight_decay_dec", wt_decay_enc)
        layer_decay   = self.hparams["layer_lr_decay"]
        beta1         = self.hparams["beta1"]
        beta2         = self.hparams["beta2"]
        eps           = self.hparams["epsilon"]
        optim_type    = self.hparams["optimizer_type"]


        self.debug_optimizer(
            lr, wt_decay_enc, wt_decay_dec, layer_decay, beta1, beta2, eps, optim_type
        )
        

        # build parameter groups ------------------------------------------------
        groups: List[Dict[str, Any]] = []

        # 1) embeddings
        groups.append(
            dict(
                params=[p for n, p in self.named_parameters() if "embedding" in n],
                lr=lr * (layer_decay ** self.hparams["n_encoders"]),
                weight_decay=0.0,
            )
        )
        self.embedding_group_idx = len(groups)-1
        # 2) each encoder block
        decay_exempt = {'bias', 'norm'}
        
        for i in range(self.hparams["n_encoders"]):
            scale = layer_decay ** (self.hparams["n_encoders"] - (i+1))
            lr_i = lr*scale
            enc_with_decay = [
                p for n, p in self.named_parameters()
                if f"encoders.{i}." in n
                and not any(t in n for t in decay_exempt)
            ]
            enc_no_decay = [
                p for n, p in self.named_parameters()
                if f"encoders.{i}." in n
                and any(t in n for t in decay_exempt)
            ]
            if enc_with_decay:
                groups.append(
                    dict(params=enc_with_decay, lr=lr_i, weight_decay=wt_decay_enc)
                )
            if enc_no_decay:
                groups.append(
                    dict(params=enc_no_decay,  lr=lr_i, weight_decay=0.0)
                )

        # 3) decoder / head (this gets 2x base lr)
        dec_params = [p for n, p in self.named_parameters() if "decoder" in n]
        groups.append(dict(params=dec_params, lr=2*lr, weight_decay=wt_decay_dec))
        self.decoder_group_idx = len(groups)-1

        # optimiser ------------------------------------------------------------
        optim_type = self.hparams["optimizer_type"].lower()
        
        # ---- optimiser ----
        if optim_type == "adamw":
            opt = torch.optim.AdamW(groups, betas=(beta1, beta2), eps=eps)
        elif optim_type == "radam":
            opt = torch.optim.RAdam(groups, betas=(beta1, beta2), eps=eps)
        elif optim_type == "adamax":
            opt = torch.optim.Adamax(groups, betas=(beta1, beta2), eps=eps)
        elif optim_type == "sgd":
            opt = torch.optim.SGD(groups, momentum=0.0)     
        else:
            raise ValueError(f"Unsupported optimizer_type: {optim_type}")

        # scheduler (one‑cycle or exponential) ---------------------------------
        sched_cfg = self.hparams["lr_scheduler"].lower()
        self._opt = opt
        if sched_cfg == "onecycle":
            total_steps = self.trainer.estimated_stepping_batches
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt, 
                max_lr=[g['lr'] for g in opt.param_groups], 
                total_steps=total_steps,
                pct_start=0.1, 
                anneal_strategy="linear", 
                three_phase=False
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sch, "interval": "step"}}
        elif sched_cfg == "exp":
            sch = torch.optim.lr_scheduler.ExponentialLR(
                opt, gamma=self.hparams.get("lr_gamma", 0.98)
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
        else:
            return opt  # no scheduler

    # ------------------------------------------------------------------ #
    # utilities                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_lookup(path: str | Path):
        """Helper kept for compatibility with previous code."""
        with open(path, "rb") as f:
            idx2tok, tok2idx = pickle.load(f)
        return idx2tok, tok2idx
