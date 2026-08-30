import torch
import torch.nn.functional as F

from factor_data import FACTOR_GROUPS
from models_xin import ParaCLAP


class FactorCLAP(ParaCLAP):
    def __init__(
        self,
        speech_name,
        text_name,
        embedding_dim=768,
        train_audio_encoder=True,
    ):
        super().__init__(
            speech_name=speech_name,
            text_name=text_name,
            embedding_dim=embedding_dim,
            train_audio_encoder=train_audio_encoder,
        )
        self.factor_heads = torch.nn.ModuleDict(
            {
                name: torch.nn.Linear(self.audio_branch.hidden_size, len(columns))
                for name, columns in FACTOR_GROUPS.items()
            }
        )

    def forward(self, audio, caption, tags=None, audio_attention_mask=None):
        del tags
        if self.train_audio_encoder and self.training:
            raw_audio = self.audio_branch(audio, attention_mask=audio_attention_mask)
        else:
            with torch.no_grad():
                raw_audio = self.audio_branch(audio, attention_mask=audio_attention_mask)
        caption_features = F.normalize(
            self.text_projection(self.text_branch(caption)), dim=-1
        )
        audio_features = F.normalize(self.audio_projection(raw_audio), dim=-1)
        factor_predictions = {
            name: torch.sigmoid(head(raw_audio))
            for name, head in self.factor_heads.items()
        }
        return caption_features, audio_features, factor_predictions, self.logit_scale.exp()
