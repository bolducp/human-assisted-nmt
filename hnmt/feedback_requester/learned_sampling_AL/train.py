import os
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from hnmt.feedback_requester.data import NMTOutputDataset, collate_pad_fn
from hnmt.feedback_requester.learned_sampling_AL.model import LearnedALSamplingLSTMClassifier
from torch.utils.tensorboard import SummaryWriter


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

        optimizer.zero_grad()
        packed_data, chrf_scores = batch

        user_obj_predictions, sys_obj_predictions = model(packed_data)
        user_obj_predictions = user_obj_predictions.squeeze()
        sys_obj_predictions = sys_obj_predictions.squeeze()

        loss_1 = 0
        loss_2 = 0

        #step through the predictions and calculate the loss one at a time
        import time
        for i, x  in enumerate(zip(user_obj_predictions, chrf_scores)):
            start_sent = time.time()
            print("i: ", i)
            pred, chrf = x
            start = time.time()

            sent_loss = user_obj_criterion(pred, chrf)
            print("sent loss 1: ", sent_loss)
            loss_1 += sent_loss


            start = time.time()
            sent_loss.backward(retain_graph=True)
            print("time to do backwards pass", time.time() - start)
            start = time.time()
            # gradients = loss_1.grad

            # instead, just take the mean of each tensor of weights
      
            mean = sum((torch.mean(torch.abs(param.grad))
                        for param in model.parameters() 
                        if param.requires_grad and param.grad is not None))
            
            # grads = list((torch.abs(param.grad) for param in model.parameters() 
            #         if param.requires_grad and param.grad is not None))
            # sizes = list((torch.numel(g) for g in grads))
            # total = sum(sizes)
            # weights = (s / total for s in sizes)
            # mean = sum(g.mean() * w for g, w in zip(grads, weights))
        
            print("mean: ", mean)
            print("sys_obj_predictions[{}]: ".format(i), sys_obj_predictions[i])

            start = time.time()
            sent_loss_2 = sys_obj_criterion(sys_obj_predictions[i], mean)
            print("sent loss 2: ", sent_loss_2)
            print("time to calculate sent_loss 2", time.time() - start)
            start = time.time()
            print()


            loss_2 += sent_loss_2
            print("time full sent", time.time() - start_sent, "\n\n")

        # loss_2 = 

        # print("loss 1 total:   ", loss_1)
        print("loss 2 total:   ", loss_2)

        loss_2.backward()

        optimizer.step()

        # epoch_loss += loss_1.item() + loss_2.item()
        epoch_loss += loss_2.item()
        print()
        print()
        print()

    return epoch_loss / len(iterator)



def run_training(
    train_file_name: str,
    validation_file_name: str
):
    writer = SummaryWriter()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset = NMTOutputDataset(train_file_name)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

    validation_set = NMTOutputDataset(validation_file_name)
    valid_dataloader = DataLoader(validation_set, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_pad_fn, pin_memory=True)

    N_EPOCHS = 2
    model = LearnedALSamplingLSTMClassifier(1586, 1586)
    optimizer = optim.Adam(model.parameters())

    sys_obj_criterion = nn.MSELoss()

    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, user_obj_loss_function, sys_obj_criterion), 
        print("train loss (epoch {}):".format(epoch), train_loss)
        writer.add_scalar("Loss/train", train_loss, epoch)
        torch.save(model.state_dict(), current_dir + "/saved_state_dicts/epoch_{}.pt".format(epoch))

        model.eval()
        
        epoch_loss = 0

        for batch in valid_dataloader:
            optimizer.zero_grad()

            packed_data, chrf_scores = batch

            user_obj_predictions, sys_obj_predictions = model(packed_data).squeeze()
            user_obj_predictions = user_obj_predictions.squeeze()
            sys_obj_predictions = sys_obj_predictions.squeeze()

            loss_1 = user_obj_loss_function(user_obj_predictions, chrf_scores)

            loss_1.backward()
            gradients = loss_1.grad

            with torch.no_grad():
                loss_2 = sys_obj_criterion(sys_obj_predictions, gradients)

                # loss_2 = 

                print("loss 2", loss_2)

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
