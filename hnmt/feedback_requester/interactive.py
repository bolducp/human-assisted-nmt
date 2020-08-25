import sys
from hnmt.nmt.main import get_document_nmt_output
from hnmt.feedback_requester.data import generate_source_sent_embeddings, generate_word_piece_sequential_input
import sentencepiece as spm
from hnmt.feedback_requester.model import LSTMClassifier
import torch
import os
from torch.utils.data import DataLoader
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn, prediction_collate_pad_fn

current_dir = os.path.dirname(os.path.realpath(__file__))
PATH = current_dir + "/saved/epoch_9.pt"

print("\n\n\n\nPaste your document here, one sentence per line. When finished, type ctrl+d or crtl+z on Windows")
document_to_translate = sys.stdin.read().split("\n")
sents_to_translate = [sent for sent in document_to_translate if sent] 

tokenizer = spm.SentencePieceProcessor(model_file='/Users/paigefink/human-assisted-nmt/hnmt/nmt/corpus/enja_spm_models/spm.ja.nopretok.model')
tokenized_doc = [" ".join(tokenizer.encode(sent, out_type=str))
                        for sent in sents_to_translate]

print("\n\n\n\nGenerating NMT system suggestions. Please be patient.")

nmt_output = get_document_nmt_output(tokenized_doc)
source_sent_embeds = generate_source_sent_embeddings(nmt_output)
final_output = generate_word_piece_sequential_input(nmt_output, source_sent_embeds, training=False)

model = LSTMClassifier(1586, 1586)
model.load_state_dict(torch.load(PATH))
model.eval()

dataloader = DataLoader(final_output, batch_size=4, shuffle=True, num_workers=4, collate_fn=prediction_collate_pad_fn, pin_memory=True)

for batch in dataloader:
    predictions = model(batch).squeeze()

