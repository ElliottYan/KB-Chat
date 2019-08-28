import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.config import *

import numpy as np
import pdb

class ContextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1, device='cuda'):
        super(ContextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.W = nn.Linear(2*hidden_size, hidden_size)
        self.device = device

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return torch.zeros(2, bsz, self.hidden_size).to(self.device)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # bsz * (length * max_token) * hdd
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        # bsz * length * max_token * hdd
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        # bsz * length * hdd
        embedded = torch.sum(embedded, 2).squeeze(2)
        # two dropout layer ??
        embedded = self.dropout_layer(embedded)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(0))
        # if input_lengths:
        #     embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        # hidden: 2 * bsz * hdd, output: bsz * length * hdd
        outputs, hidden = self.gru(embedded, hidden)
        # if input_lengths:
        #    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # 1 * bsz * hdd
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        # bsz * length * hdd
        outputs = self.W(outputs)
        return outputs, hidden

class RNNWithMemoryEncoder(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(RNNWithMemoryEncoder, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        self.rnn = ContextRNN(vocab, embedding_dim, dropout, n_layers=1)
        self.memory = EncoderMemNN(vocab, embedding_dim, hop, self.dropout, self.unk_mask, input_query=True)

    def forward(self, data):
        rnn_output, rnn_hidden = self.rnn(data['conv_seqs'], data['conv_lengths'])
        memory_out = self.memory(data, input_query=rnn_hidden)
        return memory_out

class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask, input_query=False):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.input_query = input_query

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        # not sure whether to add 'requires_grad'
        return torch.zeros(bsz, self.embedding_dim, device=self.device, requires_grad=True)

    def forward(self, data, input_query=None):
        # todo : add lm.
        story = data['src_seqs']
        # story = story.transpose(0, 1)
        story_size = story.size()  # b * m * 3
        if self.unk_mask:
            if (self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.tensor(ones, device=self.device))
                story = story * a.long()

        if not self.input_query:
            # list elements: bsz * hdd
            u = [self.get_state(story.size(0))]
        elif input_query is not None:
            u = [input_query.squeeze(0)]
        else:
            raise(ValueError("The input query option for Encoder MemNN is set to True, but the input is None."))

        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long())  # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            m_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A * u_temp, 2))
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = torch.sum(embed_C, 2).squeeze(2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        global_index = self.sigmoid(torch.sum(m_C, dim=-1))
        return global_index, u_k.unsqueeze(0)


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
