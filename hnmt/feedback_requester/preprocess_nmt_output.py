import pickle
import torch
import nltk.translate.chrf_score
from bert_serving.client import BertClient
from typing import List, Tuple

bc = BertClient()

# Grammar Checker Addition- this works but is super slow, so holding off on including as input for now
"""
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
grammar_error_matches = tool.check(hypo)
"""


def generate_source_sent_embeddings(
        output: List[Tuple[Tuple[str, torch.Tensor], str, str]]
    ) -> List[torch.Tensor]:
    """
    Given the nmt output list of tuples where each element: ((hypo, pos_probs), source_sent, gold_translation)
    Returns a list of all of the source sentence embeddings (per multilingual BERT model)

    This requires that the bert-as-a-service server is running with a pooling_strategy in use (default is REDUCE_MEAN)
    e.g. bert-serving-start -model_dir multi_cased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=NONE
    """
    bc = BertClient()
    return [bc.encode([x[1][1:]]) for x in output]


def generate_word_piece_sequential_input(
        output: List[Tuple[Tuple[str, torch.Tensor], str, str]], 
        source_sent_embeds: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, float]]:
    """
    Given the nmt output (a list of ((hypo, pos_probs), source_sent, gold_translation) tuples)
    and the list of source sentence embeddings as input, 
    uses multilingual BERT to get contextualized word-piece embeddings for each
    predicted word (dim 768), and concats each word-piece embedding with the 
    source sent embedding and the vector of (padded if necessary) probablities 
    """
    processed_sent_data = []

    for i in range(len(output)):
        source_sent_embed: torch.Tensor = torch.from_numpy(source_sent_embeds[i][0])
        pos_probs: torch.Tensor = pad_or_truncate(output[i][0][1])
        sent_piece_embeddings: torch.Tensor = torch.from_numpy(bc.encode([output[i][0][0]])[0])
        chrf_score: float = nltk.translate.chrf_score.sentence_chrf(output[i][2], output[i][0][0])

        word_piece_steps: torch.Tensor = generate_word_piece_step_tensor(sent_piece_embeddings, pos_probs, source_sent_embed)
        processed_sent_data.append((word_piece_steps, chrf_score))

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
        nmt_out: List[Tuple[Tuple[str, torch.Tensor], str, str]],
        save_filepath: str
    ) -> None:
    """
    Generate the source sentence embeddings (per BERT-as-a-serivce) and save
    """
    source_sent_embds = generate_source_sent_embeddings(nmt_out)
    pickle.dump(source_sent_embds, open(save_filepath, 'wb'))



# Change these files to match local paths
saved_nmt_out = pickle.load(open('/Users/paigefink/human-assisted-nmt/nmt/5000_nmt_out.p', "rb"))
sent_embeds_file = '/Users/paigefink/human-assisted-nmt/nmt/5000_source_sent_embeddings.p'

def run_source_sent_embeddings() -> None:
    # First, make sure that the BERT-as-a-service server is running with pooling_strategy (default is REDUCE_MEAN)
    # e.g. "bert-serving-start -model_dir multi_cased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=NONE"
    # then generate and save the source sentence embeddings to use in the subsequent preprocessing
    generate_and_save_source_sent_embeddings(saved_nmt_out, sent_embeds_file)


def run_final_preprocessing(
    save_output_path: str = None
    ) -> List[Tuple[torch.Tensor, float]]:

    # Change the BERT-as-a-service server to run with pooling_strategy set to NONE
    # e.g. "bert-serving-start -model_dir multi_cased_L-12_H-768_A-12/ -pooling_strategy=NONE -num_worker=4 -max_seq_len=NONE"

    saved_source_sent_embeddings = pickle.load(open(sent_embeds_file, "rb"))
    final_training_input = generate_word_piece_sequential_input(saved_nmt_out, saved_source_sent_embeddings)
    
    if save_output_path:
        pickle.dump(final_training_input, open(save_output_path, 'wb'))

    return final_training_input


# must do these 2 separately, with different BERT-as-a-service server settings!
# run_source_sent_embeddings()
run_final_preprocessing('/Users/paigefink/human-assisted-nmt/nmt/5000_final_out.p')