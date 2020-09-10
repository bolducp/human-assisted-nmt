import torch
from torch.distributions.bernoulli import Bernoulli


def calculate_predictions_entropy(
    predictions: torch.Tensor,
):
    bernoulli_distribution = Bernoulli(probs=predictions)
    entropies = bernoulli_distribution.entropy()
    return sum(entropies)