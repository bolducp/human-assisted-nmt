from typing import List, Tuple
import torch
import torch.nn as nn
import sacrebleu
from hnmt.feedback_requester.model import LSTMClassifier
from hnmt.feedback_requester.learned_sampling_AL.model import LearnedALSamplingLSTMClassifier
from torch.distributions.bernoulli import Bernoulli

POST_FEEDBACK_STRUCT = Tuple[torch.Tensor, int, str, str] # [(model_pred, was_asked, hypo_str, final_str)]
sys_obj_criterion = nn.MSELoss()

def calculate_post_edited_loss(
    post_interactive: List[POST_FEEDBACK_STRUCT],
    post_edited: List[str]
):
    loss = torch.tensor(0)

    for i in range(len(post_interactive)):
        model_pred = post_interactive[i][0]
        nmt_hypo = post_interactive[i][2]
        final_sent = post_edited[i]

        if was_asked(post_interactive[i]):
            chrf_score = torch.tensor(sacrebleu.sentence_chrf(nmt_hypo, [final_sent]).score)
            loss = loss + (2.25 * model_pred * chrf_score) + ((1 - model_pred) * (1 - chrf_score))
        elif was_post_edited(post_interactive[i], post_edited[i]):
            chrf_score = torch.tensor(sacrebleu.sentence_chrf(nmt_hypo, [final_sent]).score)
            loss = loss + (0.25 * (1 - model_pred) * (1 - chrf_score))
        else:
            loss = loss + model_pred
    return loss


def calculate_sent_post_edited_loss(
    sent: POST_FEEDBACK_STRUCT,
    final_sent: str
):
    model_pred = sent[0]
    nmt_hypo = sent[2]

    if was_asked(sent):
        chrf_score = torch.tensor(sacrebleu.sentence_chrf(nmt_hypo, [final_sent]).score)
        return (2.25 * model_pred * chrf_score) + ((1 - model_pred) * (1 - chrf_score))
    elif was_post_edited(sent, final_sent):
        chrf_score = torch.tensor(sacrebleu.sentence_chrf(nmt_hypo, [final_sent]).score)
        return (0.25 * (1 - model_pred) * (1 - chrf_score))
    else:
        return model_pred


def was_asked(
    post_interactive: POST_FEEDBACK_STRUCT
) -> bool:
    return post_interactive[1] == 1


def was_post_edited(
    post_interactive: POST_FEEDBACK_STRUCT,
    post_edited: str
) -> bool:
    return post_interactive[3] != post_edited


def update_model(
        model: LSTMClassifier,
        optimizer,
        post_interactive: List[POST_FEEDBACK_STRUCT],
        post_edited: List[str]
    ) -> float:
    model.train()
    optimizer.zero_grad()
    loss = calculate_post_edited_loss(post_interactive, post_edited)
    print("Document loss", loss)
    loss.backward()
    optimizer.step()
    return loss.item()


def update_learned_al_model(
        model: LearnedALSamplingLSTMClassifier,
        optimizer,
        post_interactive: List[POST_FEEDBACK_STRUCT],
        post_edited: List[str],
        sys_obj_predictions: torch.Tensor
    ) -> float:
    model.train()
    optimizer.zero_grad()

    loss_1 = 0
    loss_2 = 0

    for x, final_sent, sys_pred in zip(post_interactive, post_edited, sys_obj_predictions):
        sent_loss = calculate_sent_post_edited_loss(x, final_sent)
        loss_1 += sent_loss

        sent_loss.backward(retain_graph=True)
        mean = sum((torch.mean(torch.abs(param.grad))
                    for param in model.parameters()
                    if param.requires_grad and param.grad is not None))

        sent_loss_2 = sys_obj_criterion(sys_pred, mean)
        loss_2 += sent_loss_2

    loss_2.backward()
    optimizer.step()
    loss = loss_1 + loss_2
    return loss.item()
