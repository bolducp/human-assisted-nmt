import os
import sys
import pickle
from difflib import ndiff
import torch
from torch.utils.data import DataLoader
import sentencepiece as spm
import sacrebleu
from hnmt.utils import calculate_effort
from hnmt.nmt.main import get_document_nmt_output
from hnmt.feedback_requester.model import LSTMClassifier
from hnmt.feedback_requester.data import collate_pad_with_gold_text


def main(threshold: float, model_path: str, docs_path: str):
    model = LSTMClassifier(1586, 1586)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    doc_effort_scores = []
    doc_bleu_scores = []
    doc_chrf_scores = []

    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    for document in documents:
        dataloader = DataLoader(document, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_pad_with_gold_text, pin_memory=True)
        document_effort = 0
        gold_translations = [x[2] for x in document]
        post_interactive_text = []

        for batch in dataloader:
            predictions = model(batch[0]).squeeze()

            for i, prediction in enumerate(predictions):
                if prediction >= threshold:
                    sent_effort_score = calculate_effort(batch[1][i], batch[2][i])
                    document_effort += sent_effort_score
                    post_interactive_text.append(batch[2][i])

                else:
                    post_interactive_text.append(batch[1][i])

        bleu_score = sacrebleu.corpus_bleu(post_interactive_text, [gold_translations], lowercase=True).score
        chrf_score = sacrebleu.corpus_chrf(post_interactive_text, [gold_translations]).score

        doc_effort_scores.append(document_effort)
        doc_bleu_scores.append(bleu_score)
        doc_chrf_scores.append(chrf_score)

    return doc_effort_scores, doc_bleu_scores, doc_chrf_scores



if __name__ == "__main__":
    MODEL_PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/saved/epoch_9.pt'
    DOCS_PATH = "/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/preprocessed_docs/docs_final_20000.p"
    doc_effort_scores, doc_bleu_scores, doc_chrf_scores = main(0.5, MODEL_PATH, DOCS_PATH)

    with open("/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/scores.p", 'wb') as f:
        pickle.dump([doc_effort_scores, doc_bleu_scores, doc_chrf_scores], f)
