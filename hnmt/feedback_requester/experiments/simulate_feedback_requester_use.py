import os
import sys
import pickle
from typing import List, Tuple, Union
from difflib import ndiff
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sentencepiece as spm
import sacrebleu
from hnmt.feedback_requester.util import calculate_entropy
from hnmt.utils import calculate_effort, normalize_effort_scores
from hnmt.nmt.main import get_document_nmt_output
from hnmt.feedback_requester.model import LSTMClassifier
from hnmt.feedback_requester.learned_sampling_AL.model import LearnedALSamplingLSTMClassifier
from hnmt.feedback_requester.data import collate_pad_with_gold_text
from hnmt.feedback_requester.update import POST_FEEDBACK_STRUCT, calculate_post_edited_loss, \
                                            update_model, update_learned_al_model


def main(
    threshold: float,
    model_path: str,
    docs_path: str,
    online_learning: bool = False,
    policy: int = 1,
    active_learning: bool = False,
    al_strategy: str = 'entropy'
):
    if al_strategy == 'learned_sampling':
        model = LearnedALSamplingLSTMClassifier(1586, 1586)
    else:
        model = LSTMClassifier(1586, 1586)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    optimizer = optim.Adam(model.parameters())

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
        post_edited = []

        for batch in dataloader:
            if al_strategy == 'learned_sampling':
                predictions, sys_obj_predictions = model(batch[0])
                predictions = predictions.squeeze()
                sys_obj_predictions = sys_obj_predictions.squeeze()
            else:
                predictions = model(batch[0]).squeeze()

            for i, prediction in enumerate(predictions):
                nmt_hypo = batch[1][i]
                gold_translation = batch[2][i]
                sys_obj_pred = sys_obj_predictions[i] if al_strategy == 'learned_sampling' else None

                request_feedback = should_request_feedback(threshold, prediction, active_learning,
                                                            al_strategy, sys_obj_pred)

                if request_feedback:
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

                if online_learning:
                    posted_edited_sent = policy_post_edit_for_updating(nmt_hypo, gold_translation, policy)
                    post_edited.append(posted_edited_sent)

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

        if online_learning:
            if al_strategy == 'learned_sampling':
                update_learned_al_model(model, optimizer, post_interactive, post_edited, sys_obj_predictions)
            else:
                update_model(model, optimizer, post_interactive, post_edited)

            current_dir = os.path.dirname(os.path.realpath(__file__))
            name = f"online_updated_policy={policy}_al={active_learning}_ALstrategy={al_strategy}.pt"
            weights_updated_path = current_dir + "/saved_state_dicts/" + name
            torch.save(model.state_dict(), weights_updated_path)
            print("\nModel weights saved at {}.\n".format(weights_updated_path))

    return {
        'ksmr': normalize_effort_scores(effort_scores),
        'post_feedback_bleu': bleu_scores,
        'post_feedback_chrf': chrf_scores,
        'orig_nmt_out_bleu': orig_bleu,
        'orig_nmt_out_chrf': orig_chrf,
        'percent_sent_requested': precent_sents_requested
    }


def should_request_feedback(
    threshold: float,
    prediction: torch.Tensor,
    active_learning: bool,
    al_strategy: str,
    sys_pred: torch.Tensor
) -> bool:

    if active_learning:
        if al_strategy == 'entropy':
            return 0.5 * calculate_entropy(prediction) + 0.5 * prediction >= threshold
        elif al_strategy == 'learned_sampling':
            return 0.5 * sys_pred + 0.5 * prediction >= threshold
        else:
            raise ValueError(f'Unsupported active learning strategy {al_strategy}')

    return prediction >= threshold


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
        if chrf_score <= 0.75:
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
        if chrf_score <= 0.60:
            return gold_translation
        return nmt_hypo



if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    MODEL_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/saved_state_dicts/baseline/epoch_4.pt'
    LEARNED_AL_MODEL_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/learned_sampling_AL/saved_state_dicts/epoch_4.pt'
    DOCS_PATH = current_dir + "/preprocessed_docs/docs_60k_sents.p"

    policy_1_stats = main(0.5, MODEL_PATH, DOCS_PATH, online_learning=False)
    with open(current_dir + "/scores_pol_1.p", 'wb') as f:
        pickle.dump(policy_1_stats, f)


    policy_2_stats = main(0.5, MODEL_PATH, DOCS_PATH, policy=2, online_learning=False)
    with open(current_dir + "/scores_pol_2.p", 'wb') as f:
        pickle.dump(policy_2_stats, f)


    policy_2_online_stats = main(0.5, MODEL_PATH, DOCS_PATH, online_learning=True,
                                 policy=2, active_learning=False)
    with open(current_dir + "/scores_pol_2_online.p", 'wb') as f:
        pickle.dump(policy_2_online_stats, f)


    policy_2_AL_stats = main(0.5, MODEL_PATH, DOCS_PATH, online_learning=True, policy=2,
                            active_learning=True, al_strategy="entropy")
    with open(current_dir + "/scores_pol_2_AL.p", 'wb') as f:
        pickle.dump(policy_2_AL_stats, f)


    policy_2_learned_sampling_AL_stats = main(0.5,
                                              LEARNED_AL_MODEL_PATH,
                                              DOCS_PATH,
                                              online_learning=True,
                                              policy=2,
                                              active_learning=True,
                                              al_strategy="learned_sampling")
    with open(current_dir + "/scores_pol_2_learned_AL.p", 'wb') as f:
        pickle.dump(policy_2_learned_sampling_AL_stats, f)
