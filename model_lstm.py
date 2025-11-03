
import torch.nn as nn
import torch

class TemporalValidatorLSTM(nn.Module):
    
    def __init__(self, input_size=2048, lstm_hidden_size=64, num_lstm_layers=2, dropout=0.5):
        super(TemporalValidatorLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        self.batch_norm = nn.BatchNorm1d(lstm_hidden_size * 2) 
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        norm_out = self.batch_norm(last_time_step_out)
        fc1_out = self.relu(self.fc1(norm_out))
        dropout_out = self.dropout(fc1_out)
        final_out = self.sigmoid(self.fc_out(dropout_out))
        return final_out