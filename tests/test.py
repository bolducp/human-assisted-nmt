import torch
import random
from hnmt.feedback_requester.train import loss_function
from hnmt.feedback_requester.data import collate_pad_fn, collate_pad_with_gold_text
from hnmt.feedback_requester.experiments.graphs import calculate_time_step_averages

def test_lost_function_1():
    nmt_output = torch.Tensor([0, 0.5, 1, 0])
    chrf_scores = torch.Tensor([4, 10, 8, 10])

    assert loss_function(nmt_output, chrf_scores) == torch.Tensor([-3.5]), 'incorrect loss return value'


def test_collate_pad_1():
    data = [(torch.rand((51, 1586)), 0.5),
            (torch.rand((20, 1586)), 0.4),
            (torch.rand((32, 1586)), 0.9)]

    collated = collate_pad_fn(data)

    assert collated[0][0].shape == torch.Size([103, 1586])
    assert torch.all(collated[1].eq(torch.tensor([0.5000, 0.4000, 0.9000])))


def test_collate_pad_with_gold_text_1():
    data = [(torch.rand((15, 1586)), "i hope i can tell you good news next time.", "I wish I had good news when I can see it this time."),
            (torch.rand((10, 1586)), "then what do you suggest, mr. strand?", "So where is Strand?"),
            (torch.rand((8, 1586)), "can you not just call her?", "What if you ask by phone?")]

    collated = collate_pad_with_gold_text(data)

    assert collated[0][0].shape == torch.Size([33, 1586])

    assert collated[1] == ['i hope i can tell you good news next time.',
                            'then what do you suggest, mr. strand?',
                            'can you not just call her?']

    assert collated[2] == ['I wish I had good news when I can see it this time.',
                            'So where is Strand?',
                            'What if you ask by phone?']


def test_calculate_time_step_averages_1():
    effort_scores = [10, 20, 45, 10, 50, 30, 20, 10, 93, 90]
    averages_per_3 = calculate_time_step_averages(effort_scores, 3)
    averages_per_2 = calculate_time_step_averages(effort_scores, 2)

    assert averages_per_3 == [25.0, 27.5, 32.0]
    assert averages_per_2 == [15.0, 21.25, 27.5, 24.375, 37.8]
