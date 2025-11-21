import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, embed_size, dim, num_layers, dropout, residual_embeddings=True):
        super(LSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.rnn_dim = dim // 2
        self.linear = nn.Linear(dim + embed_size, dim)
        self.rnn = nn.LSTM(embed_size, self.rnn_dim, num_layers=num_layers, dropout=dropout,
                           bidirectional=True, batch_first=True)
        self.residual_embeddings = residual_embeddings
        self.init_hidden = nn.Parameter(nn.init.xavier_uniform_(torch.empty(2 * 2 * num_layers, self.rnn_dim)))
        self.num_layers = num_layers

    def forward(self, inputs):
        batch = inputs.size(0)
        h0 = self.init_hidden[:2 * self.num_layers].unsqueeze(1).expand(2 * self.num_layers,
                                                                        batch, self.rnn_dim).contiguous()
        c0 = self.init_hidden[2 * self.num_layers:].unsqueeze(1).expand(2 * self.num_layers,
                                                                        batch, self.rnn_dim).contiguous()

        outputs, hidden_t = self.rnn(inputs, (h0, c0))

        if self.residual_embeddings:
            outputs = torch.cat([inputs, outputs], dim=-1)
        outputs = self.linear(self.dropout(outputs))

        return F.normalize(outputs, dim=-1)
    
class TemporalGating(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TemporalGating, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [B, T, 2H]
        scores = self.fc(lstm_out)  # [B, T, 3]
        return self.softmax(scores)
