import os
import pickle
import torch
import sentencepiece as spm
import fairseq
from typing import List, Tuple


sp = spm.SentencePieceProcessor()
sp.load("/Users/paigefink/human-assisted-nmt/nmt/corpus/enja_spm_models/spm.en.nopretok.model")

current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = current_dir + '/pretrained_model_jaen'
pretrained_nmt_model = fairseq.models.BaseFairseqModel.from_pretrained(model_dir, checkpoint_file='small.pretrain.pt')


def read_lines(
        src_translations_path: str,
        trg_translations_path: str
    ) -> List[Tuple[str, str]]:
    """
    Read in lines from two separate text files the form the tokenized paralell corpus
    Return a list of tuples contained in the sentence pair, i.e. (source_translation, target_translation)
    """
    source_lines: List[str] = open(src_translations_path, encoding='utf-8').read().strip().split('\n')
    target_lines: List[str] = open(trg_translations_path, encoding='utf-8').read().strip().split('\n')
    return [(source_lines[i], line) for i, line in enumerate(target_lines)]


def parse_nmt_output(
        nmt_output: Tuple[str, torch.Tensor]
    ) -> Tuple[str, torch.Tensor]:
    """
    Takes the nmt_model output for a sentence, decodes the prediction,
    and returns the decoded string along with probablility score for each word 
    """
    hypo: str = sp.decode_pieces(nmt_output[0].split())
    pos_probs: torch.Tensor = nmt_output[1]
    return (hypo, pos_probs)


def generate_nmt_output(
        src_translations_path: str,
        trg_translations_path: str, 
        output_save_path: str
    ) -> None:
    """
    Parses the paralell corpora, runs the pretrained nmt model on the source sentences, 
    and saves the output results (probability of each word and text translation) 
    as well as the target gold translation to a pickle file
    """
    lines = read_lines(src_translations_path, trg_translations_path)
    nmt_out_lines = [(parse_nmt_output(pretrained_nmt_model.translate(line[0])), line[0], line[1]) 
                    for line in lines]
    pickle.dump(nmt_out_lines, open(output_save_path, 'wb'))



if __name__ == "__main__":
    tok_jpn_sents_path = "/Users/paigefink/human-assisted-nmt/nmt/corpus/spm/kyoto-train.ja"
    tok_en_sents_path = "/Users/paigefink/human-assisted-nmt/nmt/corpus/kftt-data-1.0/data/tok/kyoto-train.en"
    generate_nmt_output(tok_jpn_sents_path, tok_en_sents_path, "nmt/test_nmt_out.p")