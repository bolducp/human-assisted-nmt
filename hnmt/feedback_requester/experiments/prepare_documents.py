import pickle
import torch
from typing import List, Tuple
import sentencepiece as spm
from hnmt.nmt.main import get_nmt_output
from hnmt.feedback_requester.data import generate_source_sent_embeddings, generate_word_piece_sequential_input_keep_gold_text

import time

def main(save_path: str) -> None:
    tokenizer = spm.SentencePieceProcessor(model_file='/Users/paigefink/human-assisted-nmt/hnmt/nmt/corpus/enja_spm_models/spm.ja.nopretok.model')

    with open('raw_data/jesc_train.txt') as f:
        lines = f.read().strip().split('\n')

    parallel_data = []

    for line in lines:
        eng, jpn = line.split('\t')
        tok_jpn = " ".join(tokenizer.encode("".join(jpn.split()), out_type=str))
        parallel_data.append((tok_jpn, eng))

    nmt_out = get_nmt_output(parallel_data)
    documents = list(chunks(nmt_out, 100))

    with open(save_path, 'wb') as f:
        pickle.dump(documents, f)


def finish_preprocessing(file_path, save_path):
    outs = []
    with open(file_path, "rb") as f:
        documents = pickle.load(f)

    for doc in documents:
        source_sent_embds = generate_source_sent_embeddings(doc)
        final_training_input = generate_word_piece_sequential_input_keep_gold_text(doc, source_sent_embds)
        outs.append(final_training_input)

    pickle.dump(outs, open(save_path, 'wb'))


def chunks(
    sentence_pairs: List[Tuple[str, torch.Tensor, str]], 
    n: int
)-> List[Tuple[str, torch.Tensor, str]]:
    """Yield successive n-sized chunks from sentence_pairs."""
    for i in range(0, len(sentence_pairs), n):
        yield sentence_pairs[i:i + n]


if __name__ == "__main__":
    start = time.time()
    main('/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/preprocessed_docs/docs.p')
    print("seconds:", time.time() - start)