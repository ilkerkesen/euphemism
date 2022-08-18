from ast import AugStore
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class TransformerBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            config.get('text_encoder'))

    def forward(self, batch):
        return self.transformer(**batch['inputs'], labels=batch.get('labels'))
