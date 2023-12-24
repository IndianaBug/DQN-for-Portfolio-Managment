import torch
import torch.nn as nn

class DQN(nn.Module):
	def __init__(self, state_size, number_of_layers):
		super(DQN, self).__init__()
		self.first_two_layers = nn.Sequential(
			nn.Linear(state_size, number_of_layers),
			nn.ELU(),                              
			nn.Linear(number_of_layers, number_of_layers),
			nn.ELU()
		)
		self.lstm = nn.LSTM(number_of_layers, number_of_layers, 1, batch_first=True)
		self.last_linear = nn.Linear(number_of_layers, 3) 
	def forward(self, input):
		x = self.first_two_layers(input)
		lstm_out, hs = self.lstm(x)
		batch_size, seq_len, mid_dim = lstm_out.shape
		linear_in = lstm_out.contiguous().view(seq_len * batch_size, mid_dim)
		return self.last_linear(linear_in)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, targets):
        assert preds.shape == targets.shape, "Predictions and targets must have the same shape."

        quantile_losses = []
        for q in self.quantiles:
            errors = targets - preds
            loss = torch.max((q - 1) * errors, q * errors)
            quantile_losses.append(loss)

        total_loss = torch.mean(torch.sum(torch.stack(quantile_losses, dim=1), dim=1))
        return total_loss
