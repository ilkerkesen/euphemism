import torch
import torch.nn as nn
from transformers import DebertaForSequenceClassification


class DebertaBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = DebertaForSequenceClassification.from_pretrained(
            config.get('text_encoder'))

    def forward(self, inputs, labels=None):
        return self.transformer(**inputs, labels=labels)
