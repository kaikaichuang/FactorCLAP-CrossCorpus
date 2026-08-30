import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCLAPLoss(nn.Module):
    def __init__(
        self,
        gamma=0.5,
        beta=0.1,
        audio_temperature=0.07,
        text_temperature=0.07,
        detach_targets=False,
        epsilon=1e-8,
    ):
        super().__init__()
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be greater than 0 and at most 1")
        if audio_temperature <= 0 or text_temperature <= 0:
            raise ValueError("target temperatures must be positive")
        self.gamma = gamma
        self.beta = beta
        self.audio_temperature = audio_temperature
        self.text_temperature = text_temperature
        self.detach_targets = detach_targets
        self.epsilon = epsilon

    def build_soft_targets(self, local_audio_features, tag_features):
        local_audio_features = F.normalize(local_audio_features.float(), dim=-1)
        tag_features = F.normalize(tag_features.float(), dim=-1)
        audio_targets = F.softmax(
            local_audio_features @ local_audio_features.T / self.audio_temperature,
            dim=-1,
        )
        text_targets = F.softmax(
            tag_features @ tag_features.T / self.text_temperature,
            dim=-1,
        )
        relational = (
            (1.0 - self.gamma) * audio_targets + self.gamma * text_targets
        )
        identity = torch.eye(
            relational.shape[0],
            device=relational.device,
            dtype=relational.dtype,
        )
        targets = (1.0 - self.beta) * identity + self.beta * relational
        targets = targets.clamp_min(self.epsilon)
        targets = targets / targets.sum(dim=-1, keepdim=True)
        return targets.detach() if self.detach_targets else targets

    @staticmethod
    def symmetric_kl(log_predictions, targets):
        log_targets = targets.log()
        predictions = log_predictions.exp()
        forward = (targets * (log_targets - log_predictions)).sum(dim=-1)
        reverse = (predictions * (log_predictions - log_targets)).sum(dim=-1)
        return forward, reverse

    def forward(
        self,
        text_features,
        audio_features,
        local_audio_features,
        tag_features,
        logit_scale,
    ):
        device_type = audio_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            text_features = F.normalize(text_features.float(), dim=-1)
            audio_features = F.normalize(audio_features.float(), dim=-1)
            logits = logit_scale.float() * audio_features @ text_features.T
            audio_to_text = F.log_softmax(logits, dim=-1)
            text_to_audio = F.log_softmax(logits.T, dim=-1)
            targets = self.build_soft_targets(local_audio_features, tag_features)
            a_forward, a_reverse = self.symmetric_kl(audio_to_text, targets)
            t_forward, t_reverse = self.symmetric_kl(text_to_audio, targets)
            return (
                a_forward.mean()
                + a_reverse.mean()
                + t_forward.mean()
                + t_reverse.mean()
            ) / 2.0
