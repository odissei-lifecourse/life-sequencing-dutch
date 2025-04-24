
from cProfile import label
from dataclasses import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics 
import pickle
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
#from focal_loss.focal_loss import FocalLoss
# from imblearn.metrics import macro_averaged_mean_absolute_error
from scipy.stats import median_abs_deviation
from sklearn.metrics import f1_score, cohen_kappa_score
import logging

# from coral_pytorch.losses import corn_loss
from pathlib import Path
import os


"""Custom code"""
from pop2vec.llm.src.transformer.transformer_utils import *
from pop2vec.llm.src.transformer.transformer import  AttentionDecoder, CLS_Decoder, AttentionDecoderP, Deep_Decoder, Transformer
from pop2vec.llm.src.transformer.metrics import  CorrectedBAcc, CorrectedF1, CorrectedMCC, AUL

HOME_PATH = str(Path.home())
log = logging.getLogger(__name__)
REG_LOSS = ["mae", "mse", "smooth"]
CLS_LOSS = ["entropy", "focal", "ordinal", "corn", "cdw", "nll_loss"]

class Transformer_CLS(pl.LightningModule):
    """Transformer with Classification Layer"""

    def __init__(self, hparams):
        super(Transformer_CLS, self).__init__()
        self.hparams.update(hparams)
        # 1. Transformer: Load pretrained encoders
        self.init_encoder()
        # 2. Decoder
        self.init_decoder()
        # 3. Loss
        self.init_loss()
        # 4. Metric
        self.init_metrics()
        self.init_collector()
        # 5. Embeddings
        self.embedding_grad()
        # x. Logging params
        self.train_step_targets = []
        self.train_step_predictions = []
        self.last_update = 0
        self.last_global_step = 0
        self.save_hyperparameters(self.hparams)

    @property
    def num_outputs(self):
        return self.hparams.num_targets

    def init_encoder(self):
        self.transformer = Transformer(self.hparams)
        log.info(f"Embedding sample before load: {self.transformer.embedding.token.weight[1, 0].detach()}, {self.transformer.embedding.token.weight[198, 0].detach()}")
        if "none" in self.hparams.pretrained_model_path:
            log.info("No pretrained model")
        else:
            log.info("Pretrained Model Path:\n\t%s" % self.hparams.pretrained_model_path)
            full_state_dict = torch.load(
                self.hparams.pretrained_model_path, map_location=self.device
            )['state_dict']
            self.transformer.load_state_dict(
                {
                    k.replace("transformer.", "", 1): v 
                    for k, v in full_state_dict.items()
                    if k.startswith("transformer.")
                },
                strict=False
            )
        log.info(f"Embedding sample after load: {self.transformer.embedding.token.weight[1, 0].detach()}, {self.transformer.embedding.token.weight[198, 0].detach()}")
    def init_decoder(self):
        if self.hparams.pooled: 
            log.info("Model with the POOLED representation")
            self.decoder = AttentionDecoder(self.hparams, num_outputs=self.num_outputs)
            self.encoder_f = self.transformer.forward_finetuning
        else: 
            log.info("Model with the CLS representation")
            self.decoder = CLS_Decoder(self.hparams)
            self.encoder_f = self.transformer.forward_finetuning_cls

    def init_loss(self):
    
        if self.hparams.loss_type == "robust":
            raise NotImplementedError("Deprecated: use asymmetric loss instead")
        elif self.hparams.loss_type == "asymmetric":
            self.loss = AsymmetricCrossEntropyLoss(pos_weight=self.hparams.pos_weight)
        elif self.hparams.loss_type == "asymmetric_dynamic":
            raise NotImplementedError
        elif self.hparams.loss_type == "entropy":
            if 'loss_weights' in self.hparams:
                self.loss = nn.CrossEntropyLoss(
                    weight=torch.tensor(self.hparams.loss_weights)
                )
            else:
                self.loss = nn.CrossEntropyLoss()
        else:
            raise NotImplemented

    def embedding_grad(self):
        ### Remove parameters from the computational graph
        if self.hparams.freeze_positions:
            for param in self.transformer.embedding.age.parameters():
                param.requires_grad = False
            for param in self.transformer.embedding.abspos.parameters():
                param.requires_grad = False
            for param in self.transformer.embedding.segment.parameters():
                param.requires_grad = False

    def forward(self, batch):
        """Forward pass"""
        ## 1. ENCODER INPUT
        predicted = self.encoder_f(
            x=batch["input_ids"].long(),
            padding_mask=batch["padding_mask"].long()
        )
        ## 2. CLS Predictions 
        if self.hparams.pooled: 
            predicted = self.decoder(predicted, mask = batch["padding_mask"].long())
        else:
            predicted = self.decoder(predicted)

        return predicted 

    
    def forward_with_embeddings(self, embeddings, meta = None):
        """Same as forward, but takes pre-calculated embeddings instead of tokens"""
        ## 2. CLS Predictions 
        if self.hparams.pooled:
            predicted = self.transformer.forward_finetuning_with_embeddings(
            x=embeddings,
            padding_mask=meta["padding_mask"].long()
            )
            predicted = self.decoder(predicted, meta["padding_mask"].long())
        
        else:
            predicted = self.transformer.forward_finetuning_with_embeddings_cls(
            x=embeddings,
            padding_mask=meta["padding_mask"].long()
            )
            predicted = self.decoder(predicted)
        return predicted         

    def init_metrics(self):
        """Initialise variables to store metrics"""
        ### TRAIN
        self.train_accuracy = torchmetrics.Accuracy(task=self.hparams.task, threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        self.train_precision = torchmetrics.Precision(task=self.hparams.task, threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        self.train_recall = torchmetrics.Recall(task=self.hparams.task, threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        self.train_f1 = torchmetrics.F1Score(task=self.hparams.task, threshold=0.5, num_classes=self.hparams.num_targets, average="macro")
        self.train_binary_f1 = torchmetrics.classification.BinaryF1Score()
        ##### VALIDATION
        # self.val_cr_bacc = CorrectedBAcc(task=self.hparams.task, alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        # self.val_cr_f1 = CorrectedF1(task=self.hparams.task, alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        # self.val_cr_mcc = CorrectedMCC(task=self.hparams.task, alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        # self.val_aul = AUL(task=self.hparams.task)
        self.val_binary_f1 = torchmetrics.classification.BinaryF1Score()

        ##### TEST
        # self.test_cr_bacc = CorrectedBAcc(task=self.hparams.task, alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        # self.test_cr_f1 = CorrectedF1(task=self.hparams.task, alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        # self.test_cr_mcc = CorrectedMCC(task=self.hparams.task, alpha = self.hparams.asym_alpha, beta= self.hparams.asym_beta, threshold = 0.5, average="micro")
        # self.test_aul = AUL()

    def init_collector(self):
        """Collect predictions and targets"""
        self.test_trg = torchmetrics.CatMetric()
        self.test_prb = torchmetrics.CatMetric()
        self.test_id  = torchmetrics.CatMetric()
        self.val_trg = torchmetrics.CatMetric()
        self.val_prb = torchmetrics.CatMetric()
        self.val_id  = torchmetrics.CatMetric()

    def on_train_epoch_start(self, *args):
        """"""
        seed_everything(self.hparams.seed + self.trainer.current_epoch)
    #    log.info("Redraw Projection Matrices")

    def on_train_epoch_end(self, *args):
        if self.hparams.attention_type == "performer":
            log.info("Redraw Projection Matrices")
            self.transformer.redraw_projection_matrix(-1)


    def transform_targets(self, targets, seq, stage: str):
        """Transform Tensor of targets based on the type of loss"""
        if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
            targets = F.one_hot(targets.long(), num_classes = self.hparams.num_targets) ## for custom cross entropy we need to encode targets into one hot representation
        elif self.hparams.loss_type == "entropy":
           targets = targets.long()
        else:
            raise NotImplemented
        return targets

    def training_step(self, batch, batch_idx):
        """Training Iteration"""
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"],  seq = batch["input_ids"], stage="train")

        loss = self.loss(predicted, targets)
        ## 3. METRICS
        self.train_step_predictions.append(predicted.detach())
        self.train_step_targets.append(targets.detach())

        if ((self.global_step + 1) % (self.trainer.log_every_n_steps) == 0) and (
            self.last_update != self.global_step
        ):
            self.last_update = self.global_step
            self.log_metrics(
                predictions=torch.cat(self.train_step_predictions),
                targets=torch.cat(self.train_step_targets),
                loss=loss.detach(),
                stage="train",
                on_step=True,
                on_epoch=True,
            )
            del self.train_step_predictions
            del self.train_step_targets
            self.train_step_targets = []
            self.train_step_predictions = []

        ## 5. RETURN LOSS
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation Step"""

        ## 1. ENCODER-DECODER
        predicted = self(batch)
        predicted = predicted
        ## 2. LOSS
        targets = self.transform_targets(batch["target"], seq = batch["input_ids"], stage="val")
        loss = self.loss(predicted, targets)

        ## 3. METRICS
        self.log_metrics(
            predictions=predicted.detach(),
            targets=targets.detach(),
            loss=loss.detach(),
            stage="val",
            on_step=False,
            on_epoch=True,
            sid = batch["sequence_id"]
        )

        return None

    def test_step(self, batch, batch_idx):
        """Test and collect stats"""
        ## 1. ENCODER-DECODER
        predicted = self(batch)
        ## 2. LOSS
        targets = self.transform_targets(batch["target"],  seq = batch["input_ids"], stage="test")
        loss = self.loss(predicted, targets)
        ## 3. METRICS
        self.log_metrics(predictions = predicted.detach(), targets = targets.detach(), loss = loss.detach(), sid = batch["sequence_id"], stage = "test", on_step = False, on_epoch = True)
        return None

    def on_after_backward(self) -> None:
        #### Freeze Embedding Layer, except for the CLS token
        if self.hparams.parametrize_emb:
             w = self.transformer.embedding.token.parametrizations.weight.original
        else:
            w = self.transformer.embedding.token.weight
        if self.hparams.freeze_embeddings:
            w.grad[0] = 0
            w.grad[2:8] = 0
            w.grad[10:] = 0
        return super().on_after_backward()


    def debug_optimizer(self):
        print("==========================")
        print("PARAMETERS IN OPTIMIZER")
        print("\tGROUP 1:")
        for n, p in self.named_parameters():
            if "embedding" in n:
                print(f"{'Learnable' if p.requires_grad else 'Frozen'}")
                print("\t\t", n)
        print("\tGROUP 2:")
        for i in range(0, self.hparams.n_encoders):
            for n, p in self.named_parameters():
                if "encoders.%s." % i in n:
                    print(f"{'Learnable' if p.requires_grad else 'Frozen'}")
                    print("\t\t", n)
        print("\tGROUP 3:")
        for n, p in self.named_parameters():
            if "decoder" in n:
                print(f"{'Learnable' if p.requires_grad else 'Frozen'}")
                print("\t\t", n)
   
    def configure_optimizers(self):
        """"""
        self.debug_optimizer()
        no_decay = [ 
           "bias",
           "norm"
        ]
        optimizer_grouped_parameters = list()
        lr = self.hparams.learning_rate * (self.hparams.layer_lr_decay**self.hparams.n_encoders)
        optimizer_grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if ("embedding" in n) 
                ],
                "weight_decay": 0.0,
                "lr": lr,
            }
        )
        for i in range(0, self.hparams.n_encoders):
            lr = self.hparams.learning_rate * (
                (self.hparams.layer_lr_decay) ** (self.hparams.n_encoders - i)
            )  # lwoer layers should have lower learning rate

            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if ("encoders.%s." % i in n) and not (nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.hparams.weight_decay,
                    "lr": lr,
                }
            )
            optimizer_grouped_parameters.append(
                {
                    "params": [
                        p
                        for n, p in self.named_parameters()
                        if ("encoders.%s." % i in n) and  (nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                }
            )
        optimizer_grouped_parameters.append(
            {
                "params": [p for n, p in self.named_parameters() if "decoder" in n],
                "weight_decay": self.hparams.weight_decay_dc,
                "lr": self.hparams.learning_rate,
            }
        )

        if self.hparams.optimizer_type == "radam":
            optimizer = torch.optim.RAdam(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        elif self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        elif self.hparams.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                momentum=0.0,
                dampening=0.00)
        elif self.hparams.optimizer_type == "asgd":
            optimizer = torch.optim.ASGD(
                optimizer_grouped_parameters,
                t0=3000)
        elif self.hparams.optimizer_type == "adamax":
            optimizer = torch.optim.Adamax(
                optimizer_grouped_parameters,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=self.hparams.epsilon,
            )
        # no sgd , adsgd, nadam, rmsprop
        else:
            raise NotImplementedError

        if self.hparams.stage in ["search", "finetuning"]:
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=self.hparams.lr_gamma
                ), 
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            }
        }


    def log_metrics(self, predictions, targets, loss, stage, on_step: bool = True, on_epoch: bool = True, sid = None):             
        """Compute on step/epoch metrics"""
        assert stage in ["train", "val", "test"]
        scores = F.softmax(predictions, dim=1)
        if self.hparams.task == 'binary':
            predictions = (scores[:,1] > 0.5).long()
        if stage == "train":
            self.log("train/loss", loss, on_step=on_step, on_epoch = on_epoch)
            if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                self.log("train/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                self.log("train/pos_predictions", sum(scores[:,1]>0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)
            else:
                self.log("train/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)

            self.log("train/accuracy", self.train_accuracy(predictions, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/recall", self.train_recall(predictions, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/precision", self.train_precision(predictions, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/f1", self.train_f1(predictions, targets), on_step=on_step, on_epoch = on_epoch)
            self.log("train/binary-f1", self.train_binary_f1(predictions, targets), on_step=on_step, on_epoch=on_epoch)
            self.log("train/pos_predictions", sum(predictions)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)      
        elif stage == "val":
            self.log("val_loss", loss, on_step=on_step, on_epoch = on_epoch)
            
            if self.hparams.task == 'binary':
                if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                    self.log("val/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                    self.log("val/pos_predictions", sum(scores[:,1]>0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)
                else:
                    self.log("val/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)   
                
                for t in range(10, 100, 10):
                    threshold = t/100
                    predictions = (scores[:,1] > threshold).long()
                    self.log(f"val/pos_predictions_{t}", sum(predictions)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)
                    self.log(f"val-binary-f1_{t}", self.val_binary_f1(predictions, targets), on_step=on_step, on_epoch=on_epoch)

            # self.val_cr_f1.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            # self.val_cr_bacc.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            # self.val_cr_mcc.update(scores[:,1], targets[:,1])  # this should not happen in self.log 
            # self.val_aul.update(scores[:,1], targets[:,1]) # this should not happen in self.log 

            # self.log("val/f1_corrected", self.val_cr_f1, on_step = False, on_epoch = True)
            # self.log("val/bacc_corrected", self.val_cr_bacc, on_step = False, on_epoch = True)
            # self.log("val/mcc_corrected", self.val_cr_mcc, on_step = False, on_epoch = True)
            # self.log("val/aul", self.val_aul, on_step = False, on_epoch = True)
            # self.val_trg.update(targets[:,1])
            # self.val_prb.update(scores[:,1])
            # self.val_id.update(sid)

        elif stage == "test":
            self.log("test/loss", loss, on_step=on_step, on_epoch = on_epoch)
            if self.hparams.loss_type in ["robust", "asymmetric", "asymmetric_dynamic"]:
                self.log("test/pos_samples", torch.sum(targets[:,1])/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)
                self.log("test/pos_predictions", sum(scores[:,1]>0.5)/targets.shape[0], on_step=on_step, on_epoch = on_epoch)

            else:
                self.log("test/pos_samples", torch.sum(targets)/targets.shape[0],  on_step=on_step, on_epoch = on_epoch)   

            # self.test_cr_f1.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            # self.test_cr_bacc.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            # self.test_cr_mcc.update(scores[:,1], targets[:,1])  # this should not happen in self.log 
            # self.test_aul.update(scores[:,1], targets[:,1]) # this should not happen in self.log 
            # self.test_trg.update(targets[:,1])
            # self.test_prb.update(scores[:,1])
            # self.test_id.update(sid)

            # self.log("test/f1_corrected", self.test_cr_f1, on_step=False, on_epoch = True)
            # self.log("test/bacc_corrected", self.test_cr_bacc, on_step=False, on_epoch = True)
            # self.log("test/mcc_corrected", self.test_cr_mcc, on_step=False, on_epoch = True)
            # self.log("test/aul", self.test_aul, on_step = False, on_epoch = True)
    

    @staticmethod
    def load_lookup(path):
        """Load Token-to-Index and Index-to-Token dictionary"""
        with open(path, "rb") as f:
            indx2token, token2indx = pickle.load(f)
        return indx2token, token2indx


