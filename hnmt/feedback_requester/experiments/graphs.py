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

    save_plot_image(num_docs, averages[0], '{}/KSMR'.format(run_name))
    save_plot_image(num_docs, averages[1], '{}/BLEU'.format(run_name))
    save_plot_image(num_docs, averages[2], '{}/ChrF'.format(run_name))


def save_plot_image(
    num_docs: List[int],
    averages: CATEGORY,
    title: str
) -> None:
    plt.plot(num_docs, averages, 'g', label='No updates')
    plt.title('{} Score Averages at Num Docs'.format(title))
    plt.xlabel('Num Docs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(current_dir + '/plots/{}.png'.format(title))
    plt.close()


def calculate_averages(
    stats: Dict[str, Union[int, float]],
    per_docs: int,
) -> List[Union[List[int], List[float]]]:
    categories = ['ksmr', 'post_feedback_bleu', 'post_feedback_chrf']
    return [calculate_time_step_averages(stats[category], per_docs)
            for category in categories]


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


if __name__ == "__main__":
    file_name = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores_e9.p'

    with open(file_name, "rb") as f:
        stats = pickle.load(f)

    plot_score_and_acc_over_docs('run_0', stats)