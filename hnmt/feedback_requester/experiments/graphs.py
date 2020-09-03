import os
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Dict
import torch
import pickle

current_dir = os.path.dirname(os.path.realpath(__file__))
CATEGORY = List[Union[int, float]]
RUN_STATS = Dict[str, Union[int, float]]

def plot_score_and_acc_over_docs(
    dir_name: str,
    stats: List[Tuple[str, RUN_STATS]],
    per_docs: int = 10
) -> None:
    if not os.path.exists(current_dir + "/plots/" + dir_name):
        os.makedirs(current_dir + "/plots/" + dir_name)

    averages = calculate_averages(stats, per_docs)
    num_docs = [count for count in range(per_docs, len(stats[0][1]['ksmr']) + 1, per_docs)]

    bleu_improvement_avg = calculate_score_improvement_averages(averages['orig_nmt_out_bleu'],
                                                                averages['post_feedback_bleu'])
    chrf_improvement_avg = calculate_score_improvement_averages(averages['orig_nmt_out_chrf'],
                                                                averages['post_feedback_chrf'])

    save_plot_image(num_docs, averages['ksmr'], 'KSMR', dir_name)
    save_plot_image(num_docs, averages['orig_nmt_out_bleu'], 'Original BLEU', dir_name)
    save_plot_image(num_docs, averages['orig_nmt_out_chrf'], 'Original ChrF', dir_name)
    save_plot_image(num_docs, averages['post_feedback_bleu'], 'Post Feedback BLEU', dir_name)
    save_plot_image(num_docs, averages['post_feedback_chrf'], 'Post Feedback ChrF', dir_name)
    save_plot_image(num_docs, averages['percent_sent_requested'], 'Percent Sents Requested', dir_name)
    save_plot_image(num_docs, bleu_improvement_avg, 'Bleu Improvement', dir_name)
    save_plot_image(num_docs, chrf_improvement_avg, 'ChrF Improvement', dir_name)

    save_plot_map_ksmr_against_score_improvement(averages['ksmr'], bleu_improvement_avg, dir_name, 'BLEU')
    save_plot_map_ksmr_against_score_improvement(averages['ksmr'], chrf_improvement_avg, dir_name, 'ChrF')


def save_plot_image(
    num_docs: List[int],
    averages: List[Tuple[str, CATEGORY]],
    title: str,
    folder_name: str
) -> None:
    for run in averages:
        plt.plot(num_docs, run[1], "--", label=run[0])

    plt.title('{} Averages'.format(title))
    plt.xlabel('Num Docs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(current_dir + '/plots/{}/{}.png'.format(folder_name, title))
    plt.close()


def calculate_averages(
    stats: List[RUN_STATS],
    per_docs: int,
) -> Dict[str, Union[List[int], List[float]]]:
    categories = ['ksmr', 'post_feedback_bleu', 'post_feedback_chrf', 'percent_sent_requested',
                  'orig_nmt_out_bleu', 'orig_nmt_out_chrf']
    averages = {cat: [] for cat in categories}

    for category in categories:
        for run in stats:
            avgs = calculate_time_step_averages(run[1][category], per_docs)
            averages[category].append((run[0], avgs))
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
    original_score_avgs: List[Tuple[str, List[float]]],
    post_feedback_score_avgs: List[Tuple[str, List[float]]],
) -> List[Tuple[str, List[float]]]:
    run_improvement_avgs = []

    for i in range(len(original_score_avgs)):
        assert original_score_avgs[i][0] == post_feedback_score_avgs[i][0]

        improve_avgs = [post_feedback_ave - orig_avg
                            for post_feedback_ave, orig_avg
                            in zip(post_feedback_score_avgs[i][1], original_score_avgs[i][1])]
        run_improvement_avgs.append((original_score_avgs[i][0], improve_avgs))
    return run_improvement_avgs


def save_plot_map_ksmr_against_score_improvement(
    ksmr_scores: List[Tuple[str, List[int]]],
    eval_improvement_scores: List[Tuple[str, List[int]]],
    dir_name: str,
    title: str
):
    for i, run in enumerate(ksmr_scores):
        ksmr_values, scores = zip(*sorted(zip(run[1], eval_improvement_scores[i][1])))
        plt.plot(ksmr_values, scores, "o--", label=run[0])

    plt.title('{} Improvement Across KSMR'.format(title))
    plt.xlabel('KSMR (human effort)')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(current_dir + '/plots/{}/{} Improvement v KSMR.png'.format(dir_name, title))
    plt.close()


if __name__ == "__main__":
    files = [("Policy 1", "/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores_pol_1.p"),
            ("Policy 2", "/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores_pol_2.p")]

    run_stats = []
    for run in files:
        with open(run[1], "rb") as f:
            stats = pickle.load(f)
            run_stats.append((run[0], stats))

    plot_score_and_acc_over_docs('run_0', run_stats)