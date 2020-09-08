from typing import List, Tuple
import torch
import sacrebleu
from hnmt.feedback_requester.model import LSTMClassifier
from torch.distributions.bernoulli import Bernoulli

POST_FEEDBACK_STRUCT = Tuple[torch.Tensor, int, str, str] # [(model_pred, was_asked, hypo_str, final_str)]


def calculate_post_edited_loss(
    post_interactive: List[POST_FEEDBACK_STRUCT],
    post_edited: List[str],
    active_learning: bool = False,
):
    user_obj_loss = calculate_user_objective(post_interactive, post_edited)
    predictions = torch.tensor([x[0] for x in post_interactive])
    system_obj_loss = calculate_system_objective(predictions)

    return (0.5 * user_obj_loss) + (2.0 * system_obj_loss)


def calculate_system_objective(
    predictions: torch.Tensor,
):
    bernoulli_distribution = Bernoulli(probs=predictions)
    entropies = bernoulli_distribution.entropy()
    return sum(entropies)


def calculate_user_objective(
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
        post_edited: List[str],
        active_learning: bool
    ) -> float:
    model.train()
    optimizer.zero_grad()
    loss = calculate_post_edited_loss(post_interactive, post_edited, active_learning)
    print("Document loss", loss)
    loss.backward()
    optimizer.step()
    return loss.item()
