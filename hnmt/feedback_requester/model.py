import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim):
		super(LSTMClassifier, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False)
		self.hidden2out_user_objective = nn.Linear(hidden_dim, 1)
		self.hidden2out_system_objective = nn.Linear(hidden_dim, 1)
		self.sigmoid = nn.Sigmoid()


	def forward(self, packed_batch):
		
		outputs, (_ht, _ct) = self.lstm(packed_batch) # outputs.data.shape = torch.Size([1352, 1586])
		outputs, seq_lengths = pad_packed_sequence(outputs, batch_first=True)  # torch.Size([64, 45, 1586]),  torch.Size([64])
		user_obj_outputs = self.hidden2out_user_objective(outputs) # torch.Size([64, 45, 1])
		system_obj_outputs = self.hidden2out_system_objective(outputs)
		
		seq_len_indices = [length - 1 for length in seq_lengths]
		batch_indices = [i for i in range(len(seq_lengths))]
		user_obj_outputs = user_obj_outputs[batch_indices, seq_len_indices, :] # torch.Size([64, 1586])
		system_obj_outputs = system_obj_outputs[batch_indices, seq_len_indices, :] # torch.Size([64, 1586])

		user_obj_predictions = self.sigmoid(user_obj_outputs * torch.tensor(0.2))
		system_obj_predictions = self.sigmoid(system_obj_outputs)

		return user_obj_predictions, system_obj_predictions


