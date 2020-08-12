import pickle
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, PackedSequence

from typing import List, Tuple, Union

class NMTOutputDataset(Dataset):
    def __init__(self, data_file: str):
        self.data = pickle.load(open(data_file, "rb"))   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def collate_pad_fn(
        batch: List[Tuple[Tensor, float]]
    ) -> List[Union[PackedSequence, List[float]]]:
    """
    Pads variable-length sequences of word_piece vectors where
    'batch' is a list of tuples like: (sequence of wordpeice tensors, chrf_score)
    """
    data = [item[0] for item in batch]
    packed_data = pack_sequence(data, enforce_sorted=False)
    targets = [item[1] for item in batch]
    return [packed_data, targets]