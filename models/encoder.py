import torch
import torch.nn as nn
from utils.config import *

class ContextRNN(nn.Module):
    # def __init__(self, input_size, hidden_size, dropout, n_layers=1, device='cpu'):
    def __init__(self, args, device='cpu'):
        super(ContextRNN, self).__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
        self.W = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.alloc_device = device

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return (torch.zeros(2, bsz, self.hidden_size, device=self.alloc_device))

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        outputs = self.W(outputs)
        return outputs.transpose(0,1), hidden
