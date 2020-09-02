import os
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Dict
import torch
import pickle

current_dir = os.path.dirname(os.path.realpath(__file__))
CATEGORY = List[Union[int, float]]

def plot_score_and_acc_over_docs(
    run_name: str,
    stats: Dict[str, Union[int, float]],
    per_docs: int = 1
) -> None:
    if not os.path.exists(current_dir + "/plots/" + run_name):
        os.makedirs(current_dir + "/plots/" + run_name)

    averages = calculate_averages(stats, per_docs)
    num_docs = [count for count in range(per_docs, len(stats['ksmr']) + 1, per_docs)]

    bleu_improvement_avg = calculate_score_improvement_averages(stats['orig_nmt_out_bleu'],
                                                                averages['post_feedback_bleu'])
    chrf_improvement_avg = calculate_score_improvement_averages(stats['orig_nmt_out_chrf'],
                                                                averages['post_feedback_chrf'])

    save_plot_image(num_docs, averages['ksmr'], 'KSMR', run_name)
    save_plot_image(num_docs, averages['orig_nmt_out_bleu'], 'Original BLEU', run_name)
    save_plot_image(num_docs, averages['orig_nmt_out_chrf'], 'Original ChrF', run_name)
    save_plot_image(num_docs, averages['post_feedback_bleu'], 'Post Feedback BLEU', run_name)
    save_plot_image(num_docs, averages['post_feedback_chrf'], 'Post Feedback ChrF', run_name)
    save_plot_image(num_docs, averages['percent_sent_requested'], 'Percent Sents Requested', run_name)
    save_plot_image(num_docs, bleu_improvement_avg, 'Bleu Improvement', run_name)
    save_plot_image(num_docs, chrf_improvement_avg, 'ChrF Improvement', run_name)

    save_plot_map_ksmr_against_score(stats['ksmr'], stats['post_feedback_bleu'], run_name, 'BLEU')
    save_plot_map_ksmr_against_score(stats['ksmr'], stats['post_feedback_chrf'], run_name, 'ChrF')


def save_plot_image(
    num_docs: List[int],
    averages: CATEGORY,
    title: str,
    folder_name: str
) -> None:
    plt.plot(num_docs, averages, 'g', label='No updates')
    plt.title('{} Averages'.format(title))
    plt.xlabel('Num Docs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(current_dir + '/plots/{}/{}.png'.format(folder_name, title))
    plt.close()


def calculate_averages(
    stats: Dict[str, Union[int, float]],
    per_docs: int,
) -> Dict[str, Union[List[int], List[float]]]:
    categories = ['ksmr', 'post_feedback_bleu', 'post_feedback_chrf', 'percent_sent_requested',
                  'orig_nmt_out_bleu', 'orig_nmt_out_chrf']
    averages = {}

    for category in categories:
        avgs = calculate_time_step_averages(stats[category], per_docs)
        averages[category] = avgs
    return averages


def calculate_time_step_averages(
    scores: CATEGORY,
    per_docs: int
) -> Union[List[int], List[float]]:
    """
    Calculate the running average at each time step
    """
    chunk_indexes = [i for i in range(per_docs, len(scores) + 1, per_docs)]
    averages = []

    for i, count in enumerate(chunk_indexes):
        starting_i = 0 if i == 0 else chunk_indexes[i - 1]
        docs = scores[starting_i: count]
        average = sum(docs) / per_docs
        averages.append(average)

    return averages


def calculate_score_improvement_averages(
    original_score_avgs: List[float],
    post_feedback_score_avgs: List[float]
) -> List[float]:
    return [post_feedback_ave - orig_avg
            for post_feedback_ave, orig_avg
            in zip(post_feedback_score_avgs, original_score_avgs)]


def save_plot_map_ksmr_against_score(
    ksmr_scores: List[int],
    eval_scores: List[float],
    run_name: str,
    title: str
):
    ksmr_values, scores = zip(*sorted(zip(ksmr_scores, eval_scores)))

    plt.plot(ksmr_values, scores, 'g', label='No updates')
    plt.title('{} Scores Across KSMR'.format(title))
    plt.xlabel('KSMR (human effort)')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(current_dir + '/plots/{}/{}_v_KSMR.png'.format(run_name, title))
    plt.close()


if __name__ == "__main__":
    file_name = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores_e9.p'

    with open(file_name, "rb") as f:
        stats = pickle.load(f)

    plot_score_and_acc_over_docs('run_0', stats)