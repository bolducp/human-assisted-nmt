import torch
from torch.distributions.bernoulli import Bernoulli


def calculate_predictions_entropy(
    predictions: torch.Tensor,
):
    entropies = calculate_entropy(predictions)
    return sum(entropies)


def calculate_entropy(
    predictions: torch.Tensor,
):
    bernoulli_distribution = Bernoulli(probs=predictions)
    return bernoulli_distribution.entropy()