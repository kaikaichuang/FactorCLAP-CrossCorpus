import torch.nn.functional as F


def compute_similarity(logit_scale, audio_embeddings, text_embeddings):
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    return logit_scale * audio_embeddings @ text_embeddings.T
