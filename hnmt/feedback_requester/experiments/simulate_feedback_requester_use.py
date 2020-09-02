import os
import sys
import pickle
from typing import List, Tuple, Union
from difflib import ndiff
import torch
from torch.utils.data import DataLoader
import sentencepiece as spm
import sacrebleu
from hnmt.utils import calculate_effort
from hnmt.nmt.main import get_document_nmt_output
from hnmt.feedback_requester.model import LSTMClassifier
from hnmt.feedback_requester.data import collate_pad_with_gold_text
from hnmt.feedback_requester.update import POST_FEEDBACK_STRUCT, calculate_post_edited_loss, update_model


def main(
    threshold: float,
    model_path: str,
    docs_path: str,
    online_learning: bool = False,
    policy: int = 1,
):
    model = LSTMClassifier(1586, 1586)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    effort_scores = []
    bleu_scores = []
    chrf_scores = []
    orig_bleu = []
    orig_chrf = []
    precent_sents_requested = []

    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    for document in documents:
        dataloader = DataLoader(document, batch_size=32, shuffle=False, num_workers=0,
                                collate_fn=collate_pad_with_gold_text, pin_memory=True)
        document_effort = 0
        gold_translations = [x[2] for x in document]
        post_interactive = []
        total_requested = 0

        for batch in dataloader:
            predictions = model(batch[0]).squeeze()

            for i, prediction in enumerate(predictions):
                nmt_hypo = batch[1][i]
                gold_translation = batch[2][i]

                if prediction >= threshold:
                    total_requested += 1
                    sent_effort_score, final_sent = do_policy_feedback_and_post_edit(nmt_hypo,
                                                                                    gold_translation,
                                                                                    policy)
                    document_effort += sent_effort_score
                    feedback = get_prompted_feedback(online_learning, prediction, nmt_hypo, final_sent)
                    post_interactive.append(feedback)
                else:
                    no_feedback_struct = get_unprompted_struct(online_learning, prediction, nmt_hypo)
                    post_interactive.append(no_feedback_struct)

        doc_bleu_score, doc_chrf_score = calculate_bleu_and_chrf_scores(post_interactive,
                                                                        online_learning,
                                                                        gold_translations)
        effort_scores.append(document_effort)
        bleu_scores.append(doc_bleu_score)
        chrf_scores.append(doc_chrf_score)

        orig_out_bleu, orig_out_chrf, percent_requested = calculate_additional_stats(document,
                                                                                    gold_translations,
                                                                                    total_requested)
        orig_bleu.append(orig_out_bleu)
        orig_chrf.append(orig_out_chrf)
        precent_sents_requested.append(percent_requested)

    return {
        'ksmr': effort_scores,
        'post_feedback_bleu': bleu_scores,
        'post_feedback_chrf': chrf_scores,
        'orig_nmt_out_bleu': orig_bleu,
        'orig_nmt_out_chrf': orig_chrf,
        'percent_sent_requested': precent_sents_requested
    }


def calculate_additional_stats(
    document: List[Tuple[torch.Tensor, str, str]],
    gold_translations: List[str],
    total_requested: int
):
    nmt_out_sents = [x[1] for x in document]
    original_nmt_output_bleu = sacrebleu.corpus_bleu(nmt_out_sents, [gold_translations], lowercase=True).score
    original_nmt_output_chrf = sacrebleu.corpus_chrf(nmt_out_sents, [gold_translations]).score
    percent_requested = total_requested / len(document)

    return original_nmt_output_bleu, original_nmt_output_chrf, percent_requested


def get_prompted_feedback(
    online_learning: bool,
    prediction: torch.Tensor,
    nmt_hypo: str,
    final_sent: str
) -> Union[str, POST_FEEDBACK_STRUCT]:
    if online_learning:
        return (prediction, 1, nmt_hypo, final_sent)
    return final_sent


def get_unprompted_struct(
    online_learning: bool,
    prediction: torch.Tensor,
    nmt_hypo: str
) -> Union[str, POST_FEEDBACK_STRUCT]:
    if online_learning:
        return (prediction, 0, nmt_hypo, nmt_hypo)
    return nmt_hypo


def calculate_bleu_and_chrf_scores(
    post_interactive: Union[List[str], List[POST_FEEDBACK_STRUCT]],
    online_learning: bool,
    gold_translations: List[str]
) -> Tuple[float, float]:
    if online_learning:
        references = [x[3] for x in post_interactive]
    else:
        references = post_interactive

    bleu_score = sacrebleu.corpus_bleu(references, [gold_translations], lowercase=True).score
    chrf_score = sacrebleu.corpus_chrf(references, [gold_translations]).score

    return bleu_score, chrf_score


def do_policy_feedback_and_post_edit(
    nmt_hypo: str,
    gold_translation: str,
    policy: int
) -> Tuple[float, str]:
    """
    Return the sentence effort score and the final translation based on the policy.
    Policy #1: the translator will fully correct each sentence always (when prompted or post-editing)
    Policy #2:  if asked by the feedback-requester and the chrF score is <= 0.95:  fix/replace
    if not asked by the feedback-requester (i.e. post-editing) and the chrF score is <= 0.70: fix/replace
    """
    if policy == 1:
        sent_effort_score = calculate_effort(nmt_hypo, gold_translation)
        return sent_effort_score, gold_translation
    else:
        chrf_score = sacrebleu.sentence_chrf(nmt_hypo, [gold_translation]).score
        if chrf_score <= 0.95:
            sent_effort_score = calculate_effort(nmt_hypo, gold_translation)
            return sent_effort_score, gold_translation
        else:
            sent_effort_score = calculate_effort(nmt_hypo, nmt_hypo)
            return sent_effort_score, nmt_hypo


def policy_post_edit_for_updating(
    nmt_hypo: str,
    gold_translation: str,
    policy: int
) -> str:
    """
    Policy #1: the translator will fully correct each sentence always (when prompted or post-editing)
    Policy #2:  if not asked by the feedback-requester (i.e. post-editing) and the chrF score is <= 0.70: fix/replace
    """
    if policy == 1:
        return gold_translation
    else:
        chrf_score = sacrebleu.sentence_chrf(nmt_hypo, [gold_translation]).score
        if chrf_score <= 0.70:
            return gold_translation
        return nmt_hypo



if __name__ == "__main__":
    MODEL_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/saved_state_dicts/epoch_9.pt'
    DOCS_PATH = "/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/preprocessed_docs/docs_8k_sents.p"
    policy_1_stats = main(0.5, MODEL_PATH, DOCS_PATH)
    policy_2_stats = main(0.5, MODEL_PATH, DOCS_PATH, policy=2)

    with open("/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores_pol_1.p", 'wb') as f:
        pickle.dump(policy_1_stats, f)

    with open("/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores_pol_2.p", 'wb') as f:
        pickle.dump(policy_2_stats, f)
