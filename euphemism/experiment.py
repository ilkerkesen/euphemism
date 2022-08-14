import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics.functional import f1_score

from .model import DebertaBaseline


class Experiment(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DebertaBaseline(config.get('model', {}))
        self.save_hyperparameters(config)

    def forward(self, inputs, labels=None):
        return self.model(inputs=inputs, labels=labels)

    def training_step(self, batch, batch_index):
        output = self(inputs=batch['inputs'], labels=batch['labels'])
        return {
            "loss": output.loss,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_index):
        output = self(inputs=batch['inputs'])
        return {
            'gold': batch['labels'],
            'pred': output.logits.argmax(dim=1),
        }

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(inputs=batch['inputs'])
        pred = output.logits.argmax(dim=1).tolist()
        indexes = batch['indexes'].tolist()
        return {
            'predictions': pred,
            'indexes': indexes,
        }
    
    def validation_epoch_end(self, outputs):
        pred = torch.cat([x['pred'] for x in outputs])
        gold = torch.cat([x['gold'] for x in outputs])
        self.log('f1', f1_score(pred, gold), prog_bar=True)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=5e-5)