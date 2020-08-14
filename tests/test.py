import torch
from hnmt.feedback_requester.train import loss_function


def test_lost_function_1():
    nmt_output = torch.Tensor([0, 0.5, 1, 0])
    chrf_scores = torch.Tensor([4, 10, 8, 10])

    assert loss_function(nmt_output, chrf_scores) == torch.Tensor([-3.5]), 'incorrect loss return value'