import sys
from hnmt.nmt.main import get_document_nmt_output
from hnmt.feedback_requester.data import generate_source_sent_embeddings, generate_word_piece_sequential_input
import sentencepiece as spm
from hnmt.feedback_requester.model import LSTMClassifier
import torch
import pickle
import os
import sacrebleu
from difflib import ndiff
from torch.utils.data import DataLoader
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn, prediction_collate_pad_fn, collate_pad_with_gold_text


def main(threshold: float):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    # PATH = current_dir + "/saved/epoch_9.pt"
    PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/saved/epoch_9.pt'
    model = LSTMClassifier(1586, 1586)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    doc_effort_scores = []
    doc_bleu_scores = []
    doc_chrf_scores = []


    with open("/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/preprocessed_docs/docs_final_20000.p", "rb") as f:
        documents = pickle.load(f)

    for document in documents:
        dataloader = DataLoader(document, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_pad_with_gold_text, pin_memory=True)
        document_effort = 0
        gold_translations = [x[2] for x in document]
        post_interactive_text = []

        for batch in dataloader:
            
            predictions = model(batch[0]).squeeze()

            for i, prediction in enumerate(predictions):
                if prediction >= threshold:
                    sent_effort_score = calculate_effort(batch[1][i], batch[2], i)
                    document_effort += sent_effort_score
                    post_interactive_text.append(document[i][2])
                else:
                    post_interactive_text.append(document[i][1])

        bleu_score = sacrebleu.corpus_bleu(post_interactive_text, [gold_translations], lowercase=True).score
        chrf_score = sacrebleu.corpus_chrf(post_interactive_text, [gold_translations]).score

        doc_effort_scores.append(document_effort)
        doc_bleu_scores.append(bleu_score)
        doc_chrf_scores.append(chrf_score)


    return doc_effort_scores, doc_bleu_scores, doc_chrf_scores


def calculate_effort(x: str, y: str, base_effort: int = 10) -> int:
    return len([i for i in ndiff(x, y) if i[0] != ' '])



if __name__ == "__main__":
    doc_effort_scores, doc_bleu_scores, doc_chrf_scores = main(0.5)