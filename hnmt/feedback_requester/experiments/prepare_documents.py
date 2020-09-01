from typing import List
import pickle
import sentencepiece as spm
from hnmt.utils import chunks
from hnmt.nmt.main import get_nmt_output
from hnmt.feedback_requester.data import generate_source_sent_embeddings, generate_word_piece_sequential_input_keep_gold_text


def divided_jesc_docs_nmt_output(
    model_path: str,
    lines: List[str],
    num_sents_per_doc: int = 50,
) -> None:
    """
    Splits, tokenizes the JESC data lines, gets NMT output for the source inputs,
    and chunks into separate documents of 100 sentences
    """
    tokenizer = spm.SentencePieceProcessor(model_file=model_path)
    parallel_data = []

    for line in lines:
        eng, jpn = line.split('\t')
        tok_jpn = " ".join(tokenizer.encode("".join(jpn.split()), out_type=str))
        parallel_data.append((tok_jpn, eng))

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
    with open('raw_data/jesc_train.txt') as f:
        lines = f.read().strip().split('\n')
    final_docs = divided_jesc_docs_nmt_output(MODEL_PATH, lines)

    SAVE_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/preprocessed_docs/docs_8k_sents.p'
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(final_docs, f)
