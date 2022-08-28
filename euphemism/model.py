from ast import AugStore
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification
from .custom_deberta import CustomDebertaV2ForSequenceClassification as CustomDeberta

D_IMAGE_FEATURES = 1024


class TransformerBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            config.get('text_encoder'))
        # self.transformer = CustomDeberta.from_pretrained(
            # config.get('text_encoder'))

    def forward(self, batch):
        return self.transformer(**batch['inputs'], labels=batch.get('labels'))


class GroundedBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_encoder = CustomDeberta.from_pretrained(
            config.get('text_encoder'))
        self.projector = nn.Linear(
            in_features=D_IMAGE_FEATURES,
            out_features=self.text_encoder.config.hidden_size,
            bias=False,
        )

        freeze_text_encoder = self.config.get('freeze_text_encoder', False)
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def forward(self, batch):
        assert batch.get('image_features') is not None
        image_features = self.projector(batch['image_features'])
        return self.text_encoder(
            **batch['inputs'],
            visual_features=image_features.unsqueeze(1),
            labels=batch.get('labels'),
        )


class HallucinationBaseline(GroundedBaseline):
    def forward(self, batch):
        assert batch.get('term_features') is not None
        assert batch.get('defn_features') is not None
        term_features = self.projector(batch['term_features']).unsqueeze(1)
        defn_features = self.projector(batch['defn_features']).unsqueeze(1)
        visual_features = torch.cat([term_features, defn_features], dim=1)
        return self.text_encoder(
            **batch['inputs'],
            visual_features=visual_features,
            labels=batch.get('labels'),
        )