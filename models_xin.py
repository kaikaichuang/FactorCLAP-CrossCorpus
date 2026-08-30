import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    Wav2Vec2Model,
)

class Projection(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_in, d_out, bias=False)
        self.linear2 = torch.nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = torch.nn.LayerNorm(d_out)
        self.drop = torch.nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class SpeechEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.base = Wav2Vec2Model.from_pretrained(self.model_name)
        self.hidden_size = self.base.config.hidden_size

    def forward(self, x, attention_mask=None):
        x = self.base(x, attention_mask=attention_mask)["last_hidden_state"]
        if attention_mask is None:
            return x.mean(1)

        feature_mask = self.base._get_feature_vector_attention_mask(
            x.shape[1], attention_mask
        ).unsqueeze(-1)
        x = (x * feature_mask).sum(1) / feature_mask.sum(1).clamp_min(1)
        return x


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)

    def forward(self, x):
        out = self.base(**x)[0]
        out = out[:, 0, :]  # get CLS token output
        return out


class ParaCLAP(torch.nn.Module):
    def __init__(
        self,
        speech_name: str,
        text_name: str,
        embedding_dim: int = 1024,
        train_audio_encoder: bool = False,
    ):
        super().__init__()

        self.audio_branch = SpeechEncoder(model_name=speech_name)
        self.text_branch = TextEncoder(model_name=text_name)
        self.audio_projection = Projection(self.audio_branch.hidden_size, embedding_dim)
        self.text_projection = Projection(self.text_branch.base.config.hidden_size, embedding_dim)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.train_audio_encoder = train_audio_encoder
        if not self.train_audio_encoder:
            self.audio_branch.requires_grad_(False)
            self.audio_branch.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.train_audio_encoder:
            self.audio_branch.eval()
        return self

    def forward(self, audio, caption, tags=None, audio_attention_mask=None):
        del tags
        if self.train_audio_encoder and self.training:
            speech_emb = self.audio_branch(
                audio,
                attention_mask=audio_attention_mask,
            )
        else:
            with torch.no_grad():
                speech_emb = self.audio_branch(
                    audio,
                    attention_mask=audio_attention_mask,
                )
        caption_emb = self.text_branch(caption)
        speech_emb = F.normalize(self.audio_projection(speech_emb), dim=-1)
        caption_emb = F.normalize(self.text_projection(caption_emb), dim=-1)
        return (
            caption_emb,
            speech_emb,
            None,
            None,
            self.logit_scale.exp(),
        )


class SmoothCLAP(torch.nn.Module):
    def __init__(
        self,
        speech_name: str,
        text_name: str,
        local_speech_name: str,
        embedding_dim: int = 1024,
        train_audio_encoder: bool = False,
    ):
        super().__init__()

        self.audio_branch = SpeechEncoder(model_name=speech_name)
        self.local_audio_branch = SpeechEncoder(model_name=local_speech_name)
        self.text_branch = TextEncoder(model_name=text_name)
        self.audio_projection = Projection(self.audio_branch.hidden_size, embedding_dim)
        self.text_projection = Projection(self.text_branch.base.config.hidden_size, embedding_dim)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.train_audio_encoder = train_audio_encoder
        self._configure_audio_branches()

    def _configure_audio_branches(self):
        self.local_audio_branch.requires_grad_(False)
        self.local_audio_branch.eval()
        if not self.train_audio_encoder:
            self.audio_branch.requires_grad_(False)
            self.audio_branch.eval()

    def train(self, mode=True):
        super().train(mode)
        self.local_audio_branch.eval()
        if not self.train_audio_encoder:
            self.audio_branch.eval()
        return self

    def forward(self, audio, caption, tags, audio_attention_mask=None):
        with torch.no_grad():
            local_audio_emb = self.local_audio_branch(
                audio,
                attention_mask=audio_attention_mask,
            )
        if self.train_audio_encoder and self.training:
            speech_emb = self.audio_branch(
                audio,
                attention_mask=audio_attention_mask,
            )
        else:
            with torch.no_grad():
                speech_emb = self.audio_branch(
                    audio,
                    attention_mask=audio_attention_mask,
                )
        caption_emb = self.text_branch(caption)
        tag_emb = self.text_branch(tags)

        speech_emb = F.normalize(self.audio_projection(speech_emb), dim=-1)
        caption_emb = F.normalize(self.text_projection(caption_emb), dim=-1)
        local_audio_emb = F.normalize(local_audio_emb, dim=-1)
        tag_emb = F.normalize(tag_emb, dim=-1)

        return (
            caption_emb,
            speech_emb,
            local_audio_emb,
            tag_emb,
            self.logit_scale.exp(),
        )
