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
from torchmetrics import R2Score, AUROC, MeanSquaredError, MatthewsCorrCoef



from pop2vec.llm.src.transformer.transformer import  AttentionDecoder, CLS_Decoder, AttentionDecoderP, Deep_Decoder, Transformer


logger = logging.getLogger(__name__)


class TransformerFT(pl.LightningModule):
    """
    Fine‑tuning module that adds a lightweight classification head on top of a
    frozen / partially‑frozen Transformer encoder.

    * Works with the new 'FineTuneLazyDataset' (expects ''input_ids'',
      ''padding_mask'', ''target'' [+ optional ''sequence_id'']).
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
        if hparams['task_type'] == 'numeric':
            self.mu = hparams['mu']
            self.sigma = hparams['sigma']

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
        ttype = self.hparams["task_type"].lower()          # {binary,numeric,categorical}

        if ttype == "numeric":                             # numeric  →MSE
            self.criterion = nn.MSELoss()

        elif ttype in {"binary", "categorical"}:           # 
            weight = self.hparams.get("loss_weights")
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(weight) if weight is not None else None
            )
        else:
            raise ValueError(f"Unsupported task_type: {ttype}")

    def _init_metrics(self) -> None:
        ttype = self.hparams["task_type"].lower()

        if ttype == "binary":
            self.train_acc = torchmetrics.Accuracy(task="binary", threshold=0.5)
            self.train_f1  = torchmetrics.F1Score(task="binary", threshold=0.5)
            self.train_mcc = MatthewsCorrCoef(task="binary")
            self.train_auc = AUROC(task="binary")                   

            # clones
            self.val_acc,  self.test_acc  = self.train_acc.clone(),  self.train_acc.clone()
            self.val_f1,   self.test_f1   = self.train_f1.clone(),   self.train_f1.clone()
            self.val_mcc,  self.test_mcc  = self.train_mcc.clone(),  self.train_mcc.clone()
            self.val_auc,  self.test_auc  = self.train_auc.clone(),  self.train_auc.clone()  # ← NEW
        
        elif ttype == "categorical":
            k = self.hparams["num_targets"]
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=k, average="macro")
            self.train_f1  = torchmetrics.F1Score (task="multiclass", num_classes=k, average="macro")
            self.train_auc = AUROC           (task="multiclass", num_classes=k, average="macro")
            self.train_mcc = MatthewsCorrCoef(task="multiclass", num_classes=k)

            self.val_acc,  self.test_acc  = self.train_acc.clone(),  self.train_acc.clone()
            self.val_f1,   self.test_f1   = self.train_f1.clone(),   self.train_f1.clone()
            self.val_auc,  self.test_auc  = self.train_auc.clone(),  self.train_auc.clone()
            self.val_mcc,  self.test_mcc  = self.train_mcc.clone(),  self.train_mcc.clone()

        elif ttype == "numeric":
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.train_mse = MeanSquaredError()
            self.train_r2  = R2Score()

            self.val_mae,  self.test_mae = self.train_mae.clone(), self.train_mae.clone()
            self.val_mse,  self.test_mse = self.train_mse.clone(), self.train_mse.clone()
            self.val_r2,   self.test_r2  = self.train_r2.clone(),  self.train_r2.clone()

        else:
            raise ValueError(f"Unsupported task_type: {ttype}")

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, torch.Tensor], invert=False) -> torch.Tensor:
        """Plain forward (encoder → decoder)."""
        hidden = self.encoder_forward(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long(),
        )
        if self.hparams["pooled"]:
            out = self.decoder(hidden, mask=batch["padding_mask"].long())
        else:
            out = self.decoder(hidden)
        if invert:
            out = out * self.sigma + self.mu
        return out  # logits (classification) or predictions (regression)

    # ------------------------------------------------------------------ #
    # training / validation loops                                        #
    # ------------------------------------------------------------------ #
    
    # ------------------------------------------------------------------ #
    # helpers for logging                                                #
    # ------------------------------------------------------------------ #
    def _log_binary_metrics(self, stage, probs, target, thr_star):
        preds = (probs >= thr_star).long()

        # Accuracy, F1, MCC at 0.5
        self.log(f"{stage}_acc_epoch", getattr(self, f"{stage}_acc")(preds, target),
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_mcc_epoch", getattr(self, f"{stage}_mcc")(preds, target),
                 on_step=False, on_epoch=True, sync_dist=True)

        # ── AUC (threshold‑independent) ───────────────────────────────────
        self.log(f"{stage}_auc_epoch", getattr(self, f"{stage}_auc")(probs, target),
             on_step=False, on_epoch=True,
             prog_bar=(stage == "val"),   # show on val bar
             sync_dist=True)


    def _log_multiclass_metrics(
        self,
        stage: str,
        logits: torch.Tensor,   # (B, K)
        target: torch.Tensor,   # (B,)
    ) -> None:
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)

        for metric in ['acc', 'f1', 'mcc']:
            getattr(self, f"{stage}_{metric}").update(preds, target)
            self.log(
                f"{stage}_{metric}_epoch",
                getattr(self, f"{stage}_{metric}"),
                on_step=False, 
                on_epoch=True, 
                sync_dist=True
            )
            
        getattr(self, f"{stage}_auc").update(probs, target)
        self.log(
                f"{stage}_auc_epoch",
                getattr(self, f"{stage}_auc"),
                on_step=False, 
                on_epoch=True,
                prog_bar=(stage=='val'), 
                sync_dist=True
        )
        # ── ACC / F1 / MCC ────────────────────────────────────────────────
        # self.log(f"{stage}_acc_epoch",
        #          getattr(self, f"{stage}_acc")(preds, target),
        #          on_step=False, on_epoch=True, sync_dist=True)
        # self.log(f"{stage}_f1_epoch",
        #          getattr(self, f"{stage}_f1")(preds, target),
        #          on_step=False, on_epoch=True, sync_dist=True)
        # self.log(f"{stage}_mcc_epoch",
        #          getattr(self, f"{stage}_mcc")(preds, target),
        #          on_step=False, on_epoch=True, sync_dist=True)

        # # ── macro‑AUC (using the cloned Metric instance) ──────────────────
        # self.log(f"{stage}_auc_epoch",
        #          getattr(self, f"{stage}_auc")(probs, target),
        #          on_step=False, on_epoch=True,
        #          prog_bar=(stage == "val"),   # keep AUC on Val progress‑bar
        #          sync_dist=True)

    def _log_regression_metrics(self, stage, logits, target):
        for metric in ['mae', 'mse', 'r2']:
            getattr(self, f"{stage}_{metric}").update(logits, target)
            self.log(
                f"{stage}_{metric}_epoch",
                getattr(self, f"{stage}_{metric}"),
                on_step=False, 
                on_epoch=True,
                prog_bar=(stage == "val" and metric == 'r2'), 
                sync_dist=True
            )
        


        # mae = getattr(self, f"{stage}_mae")(logits.squeeze(), target)
        # mse = getattr(self, f"{stage}_mse")(logits.squeeze(), target)
        # r2  = getattr(self, f"{stage}_r2")(logits.squeeze(), target)

        # self.log(f"{stage}_mae_epoch", mae, on_step=False, on_epoch=True, sync_dist=True)
        # self.log(f"{stage}_mse_epoch", mse, on_step=False, on_epoch=True, sync_dist=True)
        # self.log(
        #     f"{stage}_r2_epoch",  
        #     r2,  
        #     on_step=False, 
        #     on_epoch=True,
        #     prog_bar=(stage == "val"), 
        #     sync_dist=True
        # )


    # ------------------------------------------------------------------ #
    # shared training / val / test step                                  #
    # ------------------------------------------------------------------ #
    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        logits = self(batch)
        target  = batch["target"]
        if self.hparams['task_type'] == 'numeric':
            target = (target - self.mu)/self.sigma 
        if self.hparams["task_type"] in ['binary', 'categorical']:
            loss = self.criterion(logits, target if target.ndim == 1 else target.squeeze(-1))
        else: # numeric
            # logging.info(f"logits shape, {logits.size()}, targets shape, {target.size()}")
            logits = logits.squeeze(-1)
            # logging.info(f"after squeezing, logits shape = {logits.size()}")
            loss = self.criterion(logits, target)    
        if target.ndim != 1:
            target = target.squeeze(-1)
        # basic loss logging
        self.log(
             f"{stage}_loss_epoch", loss,
             on_step=False, 
             on_epoch=True,
             prog_bar=True,
             sync_dist=True
        )
        self.log(
            f"{stage}_target_mean_epoch",
            target.float().mean(),
            on_step=False, 
            on_epoch=True,
            sync_dist=True,                
        )
        if stage == "train":
            self.log(
                "lr-emb", 
                self._opt.param_groups[self.embedding_group_idx]["lr"], 
                on_step=True, 
                on_epoch=False,
                sync_dist=False
            )    
            self.log(
                "lr-decoder", 
                self._opt.param_groups[self.decoder_group_idx]["lr"], 
                on_step=True, 
                on_epoch=False,
                sync_dist=False
            )
            self.log(
                "train_target_mean_step",
                target.float().mean(),
                on_step=True, 
                on_epoch=False,
                sync_dist=False,                
            )
        if self.hparams["task_type"] == "binary" and stage == 'val':
            probs = torch.softmax(logits, dim=1)[:, 1]   # P(class=1)
            self._val_probs.append(probs.detach())
            self._val_targs.append(target.detach())
        elif self.hparams["task_type"] == "categorical":
            self._log_multiclass_metrics(stage, logits, target)
        elif self.hparams["task_type"] == "numeric":
            if stage != 'train':
                logits = logits * self.sigma + self.mu
                target = target * self.sigma + self.mu
            self._log_regression_metrics(stage, logits, target)
        
        return loss

    def predict_step(self, batch, batch_idx):
        preds = self(
            batch, 
            invert=self.hparams['task_type']=='regression',
        )
        return {
            "preds": preds.detach(),           # (B, …)
            "RINPERSOON": batch["sequence_id"],   # (B,)
        }

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

        if not (self.hparams["task_type"] == 'binary' and self.hparams['num_targets'] == 2):
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
        self.log("val_f1_epoch",
                 f1_star,
                 sync_dist=False)

        self.log("val_best-thr_epoch",
                 thr_star,
                 sync_dist=False)

        self._log_binary_metrics('val', p, y, thr_star)

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
