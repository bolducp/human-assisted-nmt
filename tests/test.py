import torch
import random
from hnmt.feedback_requester.train import loss_function
from hnmt.feedback_requester.data import collate_pad_fn, collate_pad_with_gold_text
from hnmt.feedback_requester.experiments.graphs import calculate_time_step_averages
from hnmt.feedback_requester.experiments.prepare_documents import divided_jesc_docs_nmt_output

def test_lost_function_1():
    nmt_output = torch.Tensor([0, 0.5, 1, 0])
    chrf_scores = torch.Tensor([4, 10, 8, 10])

    assert loss_function(nmt_output, chrf_scores) == torch.Tensor([6.25]), 'incorrect loss return value'


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

    assert averages_per_3 == [25.0, 30.0, 41.0]
    assert averages_per_2 == [15.0, 27.5, 40.0, 15.0, 91.5]


def test_divided_jesc_docs_nmt_output():
    lines = ["you seem to be mistaken.	公序良俗に反するレベルだと 思われることのほうが問題だ。",
             "what a surprise.	驚いたわ",
             "stirrings?	動揺 ?",
             "i think it's an inspired idea.	それは見事なアイデアだと思います",
             "please. sorry, kiddo.	お願い",
             "i have more invested in this case than any of you assholes.	そこのアホより 多くを費やした",
             "come on, guys.	さあ、行くわよ!"]

    MODEL_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/nmt/corpus/enja_spm_models/spm.ja.nopretok.model'
    final_doc_outputs = divided_jesc_docs_nmt_output(MODEL_PATH, lines, 3)

    assert len(final_doc_outputs) == 3
    assert type(final_doc_outputs[0][0][0]) == torch.Tensor
    assert type(final_doc_outputs[0][0][1]) == str
    assert final_doc_outputs[0][0][2] == 'you seem to be mistaken.'


test_lost_function_1()
test_collate_pad_1()
test_collate_pad_with_gold_text_1()
test_calculate_time_step_averages_1()
test_divided_jesc_docs_nmt_output()