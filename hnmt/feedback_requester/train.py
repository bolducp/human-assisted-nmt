import os
from pathlib import Path
from torch.utils.data import DataLoader
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn

current_dir = os.path.dirname(os.path.realpath(__file__))
sample_nmt_output_file = current_dir + "/preprocessing_outputs/final_out_sample.p"
dataset = NMTOutputDataset(sample_nmt_output_file)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

