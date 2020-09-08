import os
from typing import List, Tuple
import pickle
import sentencepiece as spm
from hnmt.utils import chunks
from hnmt.nmt.main import get_nmt_output
from hnmt.feedback_requester.data import generate_source_sent_embeddings, generate_word_piece_sequential_input_keep_gold_text

current_dir = os.path.dirname(os.path.realpath(__file__))

def main(
    model_path: str,
    output_path: str
) -> List[Tuple[torch.Tensor, str, str]]:
    tokenizer = spm.SentencePieceProcessor(model_file=model_path)
    parallel_data = []

    with open(current_dir + '/raw_data/jesc_train.txt') as f:
        lines = f.read().strip().split('\n')

    for line in lines:
        eng, jpn = line.split('\t')
        tok_jpn = " ".join(tokenizer.encode("".join(jpn.split()), out_type=str))
        parallel_data.append((tok_jpn, eng))

    final_docs = divided_jesc_docs_nmt_output(parallel_data, num_sents_per_doc=50)

    with open(output_path, 'wb') as f:
        pickle.dump(final_docs, f)


def divided_jesc_docs_nmt_output(
    parallel_data: List[Tuple[str, str]],
    num_sents_per_doc: int = 100,
) -> None:
    """
    Splits, tokenizes the JESC data lines, gets NMT output for the source inputs,
    and chunks into separate documents of 100 sentences
    """
    nmt_out = get_nmt_output(parallel_data)
    documents = list(chunks(nmt_out, num_sents_per_doc))
    final_doc_outputs = []

    for doc in documents:
        source_sent_embds = generate_source_sent_embeddings(doc)
        final_training_input = generate_word_piece_sequential_input_keep_gold_text(doc, source_sent_embds)
        final_doc_outputs.append(final_training_input)

    return final_doc_outputs


if __name__ == "__main__":
    MODEL_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/nmt/corpus/enja_spm_models/spm.ja.nopretok.model'
    SAVE_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/preprocessed_docs/docs_20k_to_40k_sents.p'
    main(MODEL_PATH, SAVE_PATH)