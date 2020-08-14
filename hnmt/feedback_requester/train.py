import os
from typing import Callable
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn
from hnmt.feedback_requester.model import LSTMClassifier


def loss_function(nmt_output: Tensor, chrf_scores: Tensor) -> float:
    return sum((nmt_output * chrf_scores) + ((1 - nmt_output) * (1 - chrf_scores)))


def train(
        model: LSTMClassifier,
        iterator: DataLoader, 
        optimizer, 
        criterion
    ) -> float:
    epoch_loss = 0
    model.train()

    for batch in iterator:

        optimizer.zero_grad()
        packed_data, chrf_scores = batch

        # convert to 1D tensor
        predictions = model(packed_data).squeeze()

        loss = criterion(predictions, chrf_scores)
        print(loss)

        loss.backward()

        # update the weights
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)



def run_training():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    sample_nmt_output_file = current_dir + "/preprocessing_outputs/final_out_sample.p"
    dataset = NMTOutputDataset(sample_nmt_output_file)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

    N_EPOCHS = 5
    model = LSTMClassifier(1586, 1586)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, loss_function)
        print("train loss (epoch {}):".format(epoch), train_loss)





