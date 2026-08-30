import torch
import torch.nn.functional as F

from factor_data import FACTOR_GROUPS


def class_aware_clap_loss(text_features, audio_features, labels, logit_scale):
    logits = logit_scale.float() * audio_features.float() @ text_features.float().T
    positive = torch.tensor(
        [[left == right for right in labels] for left in labels],
        device=logits.device,
        dtype=logits.dtype,
    )
    audio_targets = positive / positive.sum(dim=1, keepdim=True)
    text_targets = positive.T / positive.T.sum(dim=1, keepdim=True)
    audio_to_text = -(audio_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    text_to_audio = -(text_targets * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return (audio_to_text + text_to_audio) / 2.0


def grouped_factor_loss(predictions, targets):
    losses = {}
    offset = 0
    for name, columns in FACTOR_GROUPS.items():
        width = len(columns)
        losses[name] = F.smooth_l1_loss(
            predictions[name].float(), targets[:, offset : offset + width].float()
        )
        offset += width
    return sum(losses.values()) / len(losses), losses
