import os
import pickle
from typing import List, Tuple, Union, Any
import nltk.translate.chrf_score
from typing import List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from transformers import BertTokenizer, BertModel
from hnmt.utils import get_root_directory

NMT_OUTPUT_TRAIN = List[Tuple[Tuple[str, torch.Tensor], str, str]]
NMT_OUTPUT_EVAL = List[Tuple[Tuple[str, torch.Tensor], str]]

root_dir = get_root_directory()

class NMTOutputDataset(Dataset):
    def __init__(self, data_file: str):
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_pad_fn(
        batch: List[Tuple[torch.Tensor, float]]
    ) -> List[Union[PackedSequence, torch.Tensor]]:
    """
    Pads variable-length sequences of word_piece vectors where
    'batch' is a list of tuples like: (sequence of wordpeice tensors, chrf_score)
    """

    data, targets = map(list, zip(*batch))
    packed_data = pack_sequence(data, enforce_sorted=False) # if I want to use dropout, might need to move this packing into the forward pass
    return [packed_data, torch.Tensor(targets)]


def prediction_collate_pad_fn(
        batch: List[Tuple[torch.Tensor, None]]
    ) -> PackedSequence:
    """
    Pads variable-length sequences of word_piece vectors where
    'batch' is a list of tuples like: (sequence of wordpeice tensors, None)
    """
    data = [seq[0] for seq in batch]
    packed_data = pack_sequence(data, enforce_sorted=False)
    return packed_data


def collate_pad_with_gold_text(
        batch: List[Tuple[torch.Tensor, str, str]]
    ) -> List[Any]:
    """
    Pads variable-length sequences of word_piece vectors where
    'batch' is a list of tuples like: (sequence of wordpeice tensors, hypo text, gold text)
    """
    data, nmt_out_text, gold_texts = map(list, zip(*batch))
    packed_data = pack_sequence(data, enforce_sorted=False)
    return [packed_data, nmt_out_text, gold_texts]


def generate_source_sent_embeddings(
        output: Union[NMT_OUTPUT_TRAIN, NMT_OUTPUT_EVAL],
    ) -> List[torch.Tensor]:
    """
    Given the nmt output list of tuples where each element: ((hypo, pos_probs), source_sent, gold_translation)
    Returns a list of all of the source sentence embeddings (per multilingual BERT model)
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)
    output_sent_embeds = []

    for sent in output:
        input_ids = torch.tensor([tokenizer.encode(sent[1][1:], add_special_tokens=True)])
        with torch.no_grad():
            model_out = model(input_ids)
            token_vecs = model_out[2][-2][0]
            output_sent_embeds.append(torch.mean(token_vecs, dim=0))

    return output_sent_embeds


def generate_word_piece_sequential_input(
        output: Union[NMT_OUTPUT_TRAIN, NMT_OUTPUT_EVAL],
        source_sent_embeds: List[torch.Tensor],
        training: bool = True,
    ) -> List[Tuple[torch.Tensor, Optional[float]]]:
    """
    Given the nmt output (a list of ((hypo, pos_probs), source_sent, gold_translation) tuples)
    and the list of source sentence embeddings as input,
    uses multilingual BERT to get contextualized word-piece embeddings for each
    predicted word (dim 768), and concats each word-piece embedding with the
    source sent embedding and the vector of (padded if necessary) probablities
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)
    processed_sent_data = []

    for i in range(len(output)):
        source_sent_embed: torch.Tensor = source_sent_embeds[i]
        pos_probs: torch.Tensor = pad_or_truncate(output[i][0][1])

        if training:
            chrf_score: Optional[float] = nltk.translate.chrf_score.sentence_chrf(output[i][2], output[i][0][0])
        else:
            chrf_score = None

        input_ids = torch.tensor([tokenizer.encode(output[i][0][0], add_special_tokens=True)])
        with torch.no_grad():
            last_four_hidden_states = model(input_ids)[2][-4:]
            last_four_stacked = torch.squeeze(torch.stack(last_four_hidden_states), dim=1)
            word_piece_vectors = torch.sum(last_four_stacked, dim=0)

        word_piece_steps: torch.Tensor = generate_word_piece_step_tensor(word_piece_vectors, pos_probs, source_sent_embed)
        processed_sent_data.append((word_piece_steps, chrf_score))

    return processed_sent_data


def generate_word_piece_sequential_input_keep_gold_text(
        output: NMT_OUTPUT_TRAIN,
        source_sent_embeds: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, str, str]]:
    """
    Given the nmt output (a list of ((hypo, pos_probs), source_sent, gold_translation) tuples)
    and the list of source sentence embeddings as input,
    uses multilingual BERT to get contextualized word-piece embeddings for each
    predicted word (dim 768), and concats each word-piece embedding with the
    source sent embedding and the vector of (padded if necessary) probablities
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states = True)
    processed_sent_data = []

    for i in range(len(output)):
        source_sent_embed: torch.Tensor = source_sent_embeds[i]
        pos_probs: torch.Tensor = pad_or_truncate(output[i][0][1])
        input_ids = torch.tensor([tokenizer.encode(output[i][0][0], add_special_tokens=True)])

        with torch.no_grad():
            last_four_hidden_states = model(input_ids)[2][-4:]
            last_four_stacked = torch.squeeze(torch.stack(last_four_hidden_states), dim=1)
            word_piece_vectors = torch.sum(last_four_stacked, dim=0)

        word_piece_steps: torch.Tensor = generate_word_piece_step_tensor(word_piece_vectors, pos_probs, source_sent_embed)
        nmt_pred: str = output[i][0][0]
        gold: str = output[i][2]

        processed_sent_data.append((word_piece_steps, nmt_pred, gold))

    return processed_sent_data


def generate_word_piece_step_tensor(
        sent_piece_embeddings: torch.Tensor,
        pos_probs: torch.Tensor,
        source_sent: torch.Tensor
    ) -> torch.Tensor:
    """
    For a set of sent_piece_embeddings for an nmt output sentence, concat each embedding
    with the source sentence embedding and positional probabalities tensor
    """
    sent_piece_dim = len(sent_piece_embeddings[0])
    source_sent_dim = len(source_sent)
    pos_probs_dim = len(pos_probs)
    word_piece_dimenson = sent_piece_dim + source_sent_dim + pos_probs_dim
    sent_piece_source_sent_dim = sent_piece_dim + source_sent_dim

    t = torch.zeros((len(sent_piece_embeddings), word_piece_dimenson))
    for i, piece in enumerate(sent_piece_embeddings):
        t[i, 0: sent_piece_dim] = piece
        t[i, sent_piece_dim: sent_piece_source_sent_dim] = source_sent
        t[i, sent_piece_source_sent_dim: sent_piece_source_sent_dim + pos_probs_dim] = pos_probs
    return t


def pad_or_truncate(pos_probs: torch.Tensor) -> torch.Tensor:
    """
    To standarize ultimate embedding dim for inputs, adjust the tensor of probablity
    scores for each word to be 50, either by adding 0 padding or truncating
    """
    probs_length = len(pos_probs)
    if probs_length < 50:
        return torch.cat([pos_probs, torch.zeros(50 - probs_length)])
    return pos_probs[:50]


def generate_and_save_source_sent_embeddings(
        nmt_out: Union[NMT_OUTPUT_TRAIN, NMT_OUTPUT_EVAL],
        save_filepath: str
    ) -> None:
    """
    Generate the source sentence embeddings (per BERT-as-a-serivce) and save
    """
    source_sent_embds = generate_source_sent_embeddings(nmt_out)
    pickle.dump(source_sent_embds, open(save_filepath, 'wb'))


# Grammar Checker Addition- this works but is super slow, so holding off on including as input for now
"""
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
grammar_error_matches = tool.check(hypo)
"""

def run_source_sent_embeddings(nmt_out_file: str, sent_embeds_file: str) -> None:
    with open(root_dir + nmt_out_file, "rb") as f:
        saved_nmt_out = pickle.load(f)
    generate_and_save_source_sent_embeddings(saved_nmt_out, sent_embeds_file)


def run_final_preprocessing(
    nmt_out_file: str,
    sent_embeds_path: str,
    save_output_path: str = None
    ) -> List[Tuple[torch.Tensor, Optional[float]]]:
    with open(root_dir + nmt_out_file, "rb") as f:
        saved_nmt_out = pickle.load(f)

    with open(sent_embeds_path, "rb") as f:
        saved_source_sent_embeddings = pickle.load(f)

    final_training_input = generate_word_piece_sequential_input(saved_nmt_out, saved_source_sent_embeddings)

    if save_output_path:
        pickle.dump(final_training_input, open(save_output_path, 'wb'))

    return final_training_input