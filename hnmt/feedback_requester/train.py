import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn
from hnmt.feedback_requester.model import LSTMClassifier
from torch.utils.tensorboard import SummaryWriter


def loss_function(nmt_output: Tensor, chrf_scores: Tensor):
    return sum(1.75 * (nmt_output * chrf_scores) + ((1 - nmt_output) * (1 - chrf_scores)))


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



def run_training(
    train_file_name: str,
    validation_file_name: str
):
    writer = SummaryWriter()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset = NMTOutputDataset(current_dir + "/preprocessing_outputs/{}.p".format(train_file_name))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

    validation_set = NMTOutputDataset(current_dir + "/preprocessing_outputs/{}.p".format(validation_file_name))
    valid_dataloader = DataLoader(validation_set, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

    N_EPOCHS = 10
    model = LSTMClassifier(1586, 1586)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, loss_function)
        print("train loss (epoch {}):".format(epoch), train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        torch.save(model.state_dict(), current_dir + "/saved_state_dicts/baseline/epoch_{}.pt".format(epoch))

        model.eval()
        with torch.no_grad():
            epoch_loss = 0

            for batch in valid_dataloader:
                packed_data, chrf_scores = batch
                predictions = model(packed_data).squeeze()
                loss = loss_function(predictions, chrf_scores)
                epoch_loss += loss.item()

            validation_loss = epoch_loss / len(valid_dataloader)
            writer.add_scalar("Loss/validation", validation_loss, epoch)
            print("validation_loss loss (epoch {}):".format(epoch), validation_loss)
            print("\n\n\n")

    writer.close()


if __name__ == "__main__":
    run_training("final_out_3k", "final_out_1k_validation")
