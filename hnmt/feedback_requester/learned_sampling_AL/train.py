import os
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn
from hnmt.feedback_requester.learned_sampling_AL.model import LearnedALSamplingLSTMClassifier
from torch.utils.tensorboard import SummaryWriter

current_dir = os.path.dirname(os.path.realpath(__file__))

def user_obj_loss_function(prediction: Tensor, chrf_score: Tensor):
    return 1.75 * (prediction * chrf_score) + ((1 - prediction) * (1 - chrf_score))


def train(
        model: LearnedALSamplingLSTMClassifier,
        iterator: DataLoader, 
        optimizer, 
        user_obj_criterion,
        sys_obj_criterion
    ) -> float:
    epoch_loss = 0
    model.train()

    for batch in iterator:
        batch_start = time.time()

        optimizer.zero_grad()
        packed_data, chrf_scores = batch

        user_obj_predictions, sys_obj_predictions = model(packed_data)
        user_obj_predictions = user_obj_predictions.squeeze()
        sys_obj_predictions = sys_obj_predictions.squeeze()

        loss_1 = 0
        loss_2 = 0

        for usr_pred, chrf, sys_pred in zip(user_obj_predictions, chrf_scores, sys_obj_predictions):
            sent_loss = user_obj_criterion(usr_pred, chrf)
            loss_1 += sent_loss

            sent_loss.backward(retain_graph=True)
            mean = sum((torch.mean(torch.abs(param.grad))
                        for param in model.parameters() 
                        if param.requires_grad and param.grad is not None))

            sent_loss_2 = sys_obj_criterion(sys_pred, mean)
            loss_2 += sent_loss_2

        loss_2.backward()
        optimizer.step()
        epoch_loss += loss_1.item() + loss_2.item()

        print("batch time: ", time.time() - batch_start)
        batch_start = time.time()

    return epoch_loss / len(iterator)



def run_training(
    train_file_name: str,
    validation_file_name: str
):
    writer = SummaryWriter(current_dir + "/runs")

    dataset = NMTOutputDataset(train_file_name)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

    validation_set = NMTOutputDataset(validation_file_name)
    valid_dataloader = DataLoader(validation_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

    N_EPOCHS = 2
    model = LearnedALSamplingLSTMClassifier(1586, 1586)
    optimizer = optim.Adam(model.parameters())

    sys_obj_criterion = nn.MSELoss()

    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, user_obj_loss_function, sys_obj_criterion)
        print("train loss (epoch {}):".format(epoch), train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        torch.save(model.state_dict(), current_dir + "/saved_state_dicts/epoch_{}.pt".format(epoch))

        model.eval()
        
        epoch_loss = 0

        for batch in valid_dataloader:
            optimizer.zero_grad()

            packed_data, chrf_scores = batch

            user_obj_predictions, sys_obj_predictions = model(packed_data)
            user_obj_predictions = user_obj_predictions.squeeze()
            sys_obj_predictions = sys_obj_predictions.squeeze()

            loss_1 = 0
            loss_2 = 0

            for usr_pred, chrf, sys_pred in zip(user_obj_predictions, chrf_scores, sys_obj_predictions):
                sent_loss = user_obj_loss_function(usr_pred, chrf)
                loss_1 += sent_loss

                sent_loss.backward(retain_graph=True)
                mean = sum((torch.mean(torch.abs(param.grad))
                            for param in model.parameters() 
                            if param.requires_grad and param.grad is not None))

            with torch.no_grad():
                sent_loss_2 = sys_obj_criterion(sys_pred, mean)
                loss_2 += sent_loss_2

                epoch_loss += loss_1.item() + loss_2.item()

        validation_loss = epoch_loss / len(valid_dataloader)
        writer.add_scalar("Loss/validation", validation_loss, epoch)
        print("validation_loss loss (epoch {}):".format(epoch), validation_loss)
        print("\n\n\n")

    writer.close()


if __name__ == "__main__":
    train_data_path = "/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/preprocessing_outputs/final_out_3k.p"
    validation_data_path = "/Users/paigefink/human-assisted-nmt/hnmt/feedback_requester/preprocessing_outputs/final_out_3k.p"
    run_training(train_data_path, validation_data_path)
