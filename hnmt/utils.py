import os
from typing import List, Tuple, Generator
from difflib import ndiff
import torch


def get_root_directory():
    return os.path.dirname(os.path.realpath(__file__))


def chunks(
    sentence_pairs: List[Tuple[str, torch.Tensor, str]], 
    n: int
)-> Generator:
    """Yield successive n-sized chunks from sentence_pairs."""
    for i in range(0, len(sentence_pairs), n):
        yield sentence_pairs[i:i + n]


def calculate_effort(
    x: str,
    y: str,
    base_effort: int = 25
) -> int:
    """
    Use the python difflib library to help calculate the KSMR variant score between two sentences
    """
    return len([i for i in ndiff(x, y) if i[0] != ' ']) + base_effort