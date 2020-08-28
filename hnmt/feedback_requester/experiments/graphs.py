import matplotlib.pyplot as plt
from typing import List, Union, Tuple
import torch
import pickle

            # Effort scores,  BLEU scores,  chrF scores
CATEGORIES = Tuple[List[int], List[float], List[float]]
CATEGORY = List[Union[int, float]]

def plot_score_and_acc_over_docs(
    score_lists: CATEGORIES,
    per_docs: int = 1,
) -> None:
    averages = calculate_averages(score_lists, per_docs)
    num_docs = [count for count in range(per_docs, len(score_lists[0]) + 1, per_docs)]

    save_plot_image(num_docs, averages[0], 'KSMR')
    save_plot_image(num_docs, averages[1], 'BLEU')
    save_plot_image(num_docs, averages[2], 'ChrF')


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
    plt.savefig('plots/{}.png'.format(title))
    plt.close()


def calculate_averages(
    scores_lists: CATEGORIES,
    per_docs: int,
) -> List[Union[List[int], List[float]]]:
    return [calculate_time_step_averages(scores, per_docs) 
            for scores in scores_lists]


def calculate_time_step_averages(
    scores: CATEGORY,
    per_docs: int
) -> Union[List[int], List[float]]:
    """
    Calculate the running average at each time step
    """
    chunk_indexes = [i for i in range(per_docs, len(scores) + 1, per_docs)]
    averages = []

    for i in chunk_indexes:
        up_to_index = scores[0: i]
        average = sum(up_to_index) / len(up_to_index)
        averages.append(average)

    return averages

if __name__ == "__main__":
    file_name = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores.p'

    with open(file_name, "rb") as f:
        score_lists = pickle.load(f)

    plot_score_and_acc_over_docs(score_lists)