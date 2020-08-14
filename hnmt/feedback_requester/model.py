import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence


class LSTMClassifier(nn.Module):

	def __init__(self, embedding_dim, hidden_dim):
		super(LSTMClassifier, self).__init__()

		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False)
		self.hidden2out = nn.Linear(hidden_dim, 1)
		self.sigmoid = nn.Sigmoid()


	def forward(self, packed_batch):
		
		outputs, (_ht, _ct) = self.lstm(packed_batch) # outputs.data.shape = [X, 1586]
		outputs, seq_lengths = pad_packed_sequence(outputs, batch_first=True)
		outputs = self.hidden2out(outputs)
		
		seq_len_indices = [length - 1 for length in seq_lengths]
		batch_indices = [i for i in range(len(seq_lengths))]
		outputs = outputs[batch_indices, seq_len_indices, :]

		final_output = self.sigmoid(outputs)

		return final_output