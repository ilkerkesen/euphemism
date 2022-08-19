import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from torchmetrics.functional import f1_score

from .model import TransformerBaseline


class Experiment(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TransformerBaseline(config.get('model', {}))
        self.save_hyperparameters(config)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_index):
        output = self(batch)
        return {
            "loss": output.loss,
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_index):
        labels = batch.pop('labels')
        output = self(batch)
        return {
            'gold': labels,
            'pred': output.logits.argmax(dim=1),
        }

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch)
        prob = output.logits.softmax(dim=1)
        prob = prob[:, 1].tolist()
        indexes = batch['indexes'].tolist()
        return {
            'predictions': prob,
            'indexes': indexes,
        }
    
    def validation_epoch_end(self, outputs):
        pred = torch.cat([x['pred'] for x in outputs])
        gold = torch.cat([x['gold'] for x in outputs])
        self.log('f1', f1_score(pred, gold), prog_bar=True)

    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.config['lr'])