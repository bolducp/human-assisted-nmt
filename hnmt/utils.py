import os
import torch
from typing import List, Tuple, Generator


def get_root_directory():
    return os.path.dirname(os.path.realpath(__file__))


def chunks(
    sentence_pairs: List[Tuple[str, torch.Tensor, str]], 
    n: int
)-> Generator:
    """Yield successive n-sized chunks from sentence_pairs."""
    for i in range(0, len(sentence_pairs), n):
        yield sentence_pairs[i:i + n]
