import sys
from hnmt.nmt.main import get_document_nmt_output
from hnmt.feedback_requester.data import generate_source_sent_embeddings, generate_word_piece_sequential_input
import sentencepiece as spm
from hnmt.feedback_requester.model import LSTMClassifier
import torch
import pickle
import os
from torch.utils.data import DataLoader
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn, prediction_collate_pad_fn, collate_pad_with_gold_text



current_dir = os.path.dirname(os.path.realpath(__file__))
# PATH = current_dir + "/saved/epoch_9.pt"
PATH = '/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/saved/epoch_9.pt'


model = LSTMClassifier(1586, 1586)
model.load_state_dict(torch.load(PATH))
model.eval()


with open("/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/experiments/preprocessed_docs/docs_final_20000.p", "rb") as f:
    documents = pickle.load(f)

for document in documents:
    dataloader = DataLoader(document, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_pad_with_gold_text, pin_memory=True)


    for batch in dataloader:
        predictions = model(batch[0]).squeeze()
        print(predictions)