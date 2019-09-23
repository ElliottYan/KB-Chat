import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.config import *

import numpy as np
import pdb

from utils.logging import logger

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
        self.memory = ExternalKnowledge(vocab, embedding_dim, hop, self.dropout, self.unk_mask, input_query=True)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(2*embedding_dim, embedding_dim)

    def forward(self, data):
        # input for rnn is only the conversation seqs.
        # rnn_output: bz * length * dim, rnn_hidden: 1 * bz * dim
        rnn_output, rnn_hidden = self.rnn(data['conv_seqs'], data['conv_lengths'])
        global_pointer, kb_readout = self.memory(data, rnn_hidden, rnn_output, encode_or_decode='encode')
        # bz * (2 * dim)
        encoded_out = torch.cat([rnn_hidden.squeeze(0), kb_readout], dim=-1)
        # 1 * bz * dim
        encoded_out = self.relu(self.projector(encoded_out)).unsqueeze(0)
        assert len(encoded_out.shape) == 3
        return global_pointer, encoded_out

'''
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

    def decoder_forward(self, query_vector, global_pointer):
        u = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            # todo : not use global_pointer for now.
            # if not args["ablationG"]:
            #     m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if(len(list(u[-1].size()))==1):
                u[-1] = u[-1].unsqueeze(0) ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A*u_temp, 2)
            prob_soft   = self.softmax(prob_logits)
            m_C = self.m_story[hop+1]
            # if not args["ablationG"]:
            #     m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return prob_soft, prob_logits
'''

class ExternalKnowledge(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask, input_query=False):
        super(ExternalKnowledge, self).__init__()
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        # create a new set used in decoding
        for hop in range(self.max_hops + 1):
            E = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            E.weight.data.normal_(0, 0.1)
            self.add_module("E_{}".format(hop), E)
        self.E = AttrProxy(self, "E_")

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        # self.conv_layer = nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2)
        # self.sigmoid = nn.Sigmoid()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.input_query = input_query
        # default False
        self.share_memnet = True
        if not self.share_memnet:
            logger.info("No sharing memnet in ExternalKnowledge!")

    def add_lm_embedding(self, full_memory, kb_len, conv_len, hiddens):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + hiddens[bi, :conv_len[bi], :]
        return full_memory

    def forward(self, *args, encode_or_decode='decode', use_global=False):
        if encode_or_decode == 'decode':
            '''
            :param query_vector: bz * dim
            :param global_pointer: bz * length
            :return: prob_soft, prob_logits
            '''
            data = args[0]
            query_vector = args[1]
            global_pointer = args[2]
            u = [query_vector]
            story = data['src_seqs']
            story_size = story.size()

            for hop in range(self.max_hops):
                # bz * length * dim
                if self.share_memnet:
                    m_A = self.m_story[hop]
                else:
                    embed_A = self.E[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
                    embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
                    embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
                    # miss a adding lm layer here.
                    m_A = self.dropout_layer(embed_A)

                if use_global:
                    # the point is mask some of the memory ??
                    m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
                if (len(list(u[-1].size())) == 1):
                    u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
                # bz * length * dim
                u_temp = u[-1].unsqueeze(1).expand_as(m_A)
                # bz * length
                prob_logits = torch.sum(m_A * u_temp, 2)
                # bz * length
                prob_soft = self.softmax(prob_logits)

                # bz * length * dim
                if self.share_memnet:
                    m_C = self.m_story[hop + 1]
                else:
                    embed_C = self.E[hop + 1](story.contiguous().view(story_size[0], -1).long())
                    embed_C = embed_C.view(story_size + (embed_C.size(-1),))
                    m_C = torch.sum(embed_C, 2).squeeze(2)

                # if not args["ablationG"]:
                if use_global:
                    m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
                prob = prob_soft.unsqueeze(2).expand_as(m_C)
                o_k = torch.sum(m_C * prob, 1)
                u_k = u[-1] + o_k
                u.append(u_k)
            return prob_soft, prob_logits

        elif encode_or_decode == 'encode':
            data = args[0]
            input_query = args[1]
            dh_outputs = args[2]
            kb_len = data['mem_kb_arr_lengths']
            conv_len = data['conv_lengths']

            story = data['src_seqs']
            # Forward multiple hop mechanism
            u = [input_query.squeeze(0)]
            story_size = story.size()
            self.m_story = []
            for hop in range(self.max_hops):
                embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
                embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
                embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
                # if not args["ablationH"]:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, dh_outputs)
                embed_A = self.dropout_layer(embed_A)

                embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
                embed_C = embed_C.view(story_size + (embed_C.size(-1),))
                embed_C = torch.sum(embed_C, 2).squeeze(2)
                # if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, dh_outputs)

                # compute global indexes.
                if (len(list(u[-1].size())) == 1):
                    u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
                u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
                prob_logit = torch.sum(embed_A * u_temp, 2)
                prob_ = self.softmax(prob_logit)

                prob = prob_.unsqueeze(2).expand_as(embed_C)
                o_k = torch.sum(embed_C * prob, 1)
                u_k = u[-1] + o_k
                u.append(u_k)
                self.m_story.append(embed_A)
            self.m_story.append(embed_C)
            return self.sigmoid(prob_logit), u[-1]


class DecoderTreeNN(nn.Module):
    def __init__(self, vocab, n_type, embedding_dim, hop, dropout, unk_mask, device=None, args=None, shared_embedding=None):
        super(DecoderTreeNN, self).__init__()
        self.args = args
        self.num_vocab = vocab
        self.num_type = n_type
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        self.unk_mask = unk_mask

        # add an option to pass in the shared embeddings.
        if shared_embedding is not None and args.share_embed is True:
            self.embedding = shared_embedding
        else:
            E = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            E.weight.data.normal_(0, 0.1)
            self.embedding = E

        # type embedding
        self.T = nn.Embedding(self.num_type, embedding_dim, padding_idx=TYPE_PAD_token)
        self.T.weight.data.normal_(0, 0.1)

        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        # C[max_hop] will not be used for updates
        if self.max_hops > 1:
            logger.info("Max hops : {}".format(self.max_hops))
            self.C[self.max_hops].weight.requires_grad = False
        # self.T = AttrProxy(self, "T_")
        # self.TM = nn.Embedding(self.num_type, embedding_dim * embedding_dim, padding_idx=TYPE_PAD_token)

        self.softmax = nn.Softmax(dim=1)
        # self.W = nn.Linear(embedding_dim, 1)
        # self.W1 = nn.Linear(2 * embedding_dim, self.num_vocab)
        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # ensemble parameter (ensemble_v3).
        self.ensemble_fc = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        # ensemble attention score. (ensemble_v5)
        self.att_w1 = nn.Parameter(torch.empty((self.embedding_dim, self.embedding_dim)))
        self.att_w2 = nn.Parameter(torch.empty((self.embedding_dim, self.embedding_dim)))
        self.v = nn.Parameter(torch.empty(1, self.embedding_dim))
        nn.init.kaiming_normal_(self.v.data, mode='fan_out')
        nn.init.kaiming_normal_(self.att_w1.data, mode='fan_out')
        nn.init.kaiming_normal_(self.att_w2.data, mode='fan_out')

        self.use_global = args.use_global

        # todo : batch_first
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')

        self.ensemble_ret = None

        # embed for do-not-use-kb-infos
        if args.no_kb_embed:
            self.no_kb_embed = nn.Parameter(torch.empty(1, self.embedding_dim))
            nn.init.kaiming_normal_(self.no_kb_embed.data, mode='fan_out')

        self.add_dropout = args.add_dropout
        self.add_relu = args.add_relu
        self.add_skip_con = args.add_skip_con
        self.add_norm = args.add_norm

        self.relu = nn.ReLU()
        # layer norm in ensemble.
        if self.args.add_norm:
            self.layer_norm = nn.LayerNorm(self.embedding_dim)

        self.use_kb_tree = False

    # # load the origin inputs.
    def load_memory(self, story):
        story_size = story.size()  # b * m * 3
        if self.unk_mask:
            if (self.training):
                # random masking ? Some kind of dropout ?
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.tensor(ones, device=self.device))
                story = story * a.long()
        self.m_story = []
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1))  # .long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            m_A = embed_A
            # embed_C is not used until last hop... waste of computation.
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def load_tree_memory(self, data, hop, is_key=True):

        def dfs(tree):
            val_idx = tree.val_idx
            # ret : k * embed_size
            if is_key:
                ret = self.C[hop](val_idx).sum(0, keepdim=True)
            else:
                ret = self.C[hop+1](val_idx).sum(0, keepdim=True)
            if tree.children:
                # child_embed : n_tree * embed_size
                child_embeds = [dfs(item) for item in tree.children]
                child_embeds = torch.cat(child_embeds, dim=0)
                # merge child embeddings
                child_embeds = torch.sum(child_embeds, dim=0, keepdim=True) # 1 * embed_size
                # todo : lambda, no type embedding.
                ret = child_embeds + ret
            return ret

        batch_trees = data['kb_tree']
        max_num_trees = max([len(item) for item in batch_trees])
        # at least one. for the ease of processing.
        max_num_trees = max(1, max_num_trees)
        # padding trees
        batch_embeds = []
        for ix, trees in enumerate(batch_trees):
            tmp = []
            for tree in trees:
                # list of tensor with shape : 1 * embed_size
                tmp.append(dfs(tree))
            # padding, consistent with other paddings.
            pad_tokens = torch.tensor([PAD_token] * (max_num_trees - len(trees)), device=self.device).long()
            pad_embeds = self.C[hop](pad_tokens)
            tmp.append(pad_embeds)
            trees_embeds = torch.cat(tmp, dim=0)
            batch_embeds.append(trees_embeds)

        return torch.stack(batch_embeds)

    def forward(self, extKnow, decoder_input, data, hidden_states, global_index):

        embed_q = self.embedding(decoder_input)  # b * e
        if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
        # add dropout
        embed_q = self.dropout_layer(embed_q)
        # gru for update hidden state.
        output, hidden = self.gru(embed_q.unsqueeze(0), hidden_states)

        if self.use_kb_tree:
            # for each time step, we compute the kb_attn_features based on current hidden state.
            # todo : has accumulating problem ??
            kb_attn_features, kb_attn_weights = self.compute_global_ranking(data, hidden)
            # kb_attn_features, kb_attn_weights = self.compute_global_ranking_v2(data, hidden_states)
            # current state
            cur_state = hidden + kb_attn_features.unsqueeze(0)
            cur_state = self.dropout_layer(cur_state)
        else:
            cur_state = hidden

        # remove kb_tree to debug.
        # embed_q = self.embedding(decoder_input)  # b * e
        # embed_q = self.dropout_layer(embed_q)
        # output, hidden = self.gru(embed_q.unsqueeze(0), hidden_states)

        p_ptr, p_vocab, decoder_hidden = self.ptrMemDecoder(extKnow, data, cur_state, global_index)
        return p_ptr, p_vocab, decoder_hidden

    def compute_global_ranking(self, data, hidden_states):
        '''
        Compute the kb ensemble feature with respect to hidden_states.
        :param data: dict, contains all features
        :param hidden_states: T * B * hidden_size
        :return:
        '''
        # todo : this result can be used for all time step.
        # todo : self-attention in ensemble ?
        roots_embed, attention_bias = self.ensemble_v5_batch(data, hidden_states)
        # roots_embed, attention_bias = self.ensemble_relation_matrix(data)

        if self.args.no_kb_embed:
            no_kb_shape = list(roots_embed.shape)
            no_kb_shape[1] = 1
            no_kb_embeds = self.no_kb_embed.expand(no_kb_shape)
            # concat at the end
            roots_embed = torch.cat([roots_embed, no_kb_embeds], dim=1)
            tmp_shape = list(attention_bias.shape)
            tmp_shape[1] = 1
            attention_bias = torch.cat([attention_bias, torch.zeros(tmp_shape, device=self.device).float()], dim=1)

        # B * 1 * hidden_size
        query = self.Q(hidden_states[-1]).unsqueeze(-1)
        # B * Nt * hidden_size
        key = self.K(roots_embed)
        # B * Nt * hidden_size
        value = self.V(roots_embed)
        attn_weights = F.softmax(torch.bmm(key, query) + attention_bias, dim=1)
        attn_features = (attn_weights * value).sum(1)

        return attn_features, attn_weights

    def ensemble_v5_batch(self, data, hidden_state):
        # ensemble dynamically with hidden_state
        """
        :param data:
            dict, contains all features
        :param hidden_state:
            tensor, last hidden state
        :return:
            root_embeds -> B * Nt * hidden_size,
            attention_weights -> B * Nt * 1
        """
        types = data['pad_kb_types']
        fathers = data['pad_kb_fathers']
        values = data['pad_kb_values']
        n_layers = data['pad_kb_n_layers']
        batch_size = len(values)
        padding_idx = n_layers[:, :, 0] == -1
        padding_mask = (1 - padding_idx).float()

        # get the embeddings of each tree node.
        # B * n_trees * n_nodes * n_tokens * hidden_size
        node_value_embeds = self.embedding(values)
        # sum over each nodes. BOW now.
        node_value_embeds = node_value_embeds.sum(-2)
        node_type_embeds = self.T(types)

        # create a pivot position.
        pad_shape = list(node_value_embeds.shape)
        pad_shape[1] = 1
        # use last item as padding
        # shape : B * n_trees * (n_nodes + 1) * hidden_size
        node_value_embeds = torch.cat([node_value_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
        node_type_embeds = torch.cat([node_type_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
        # shape : B * n_trees * (n_nodes + 1)
        fathers = torch.cat([fathers, torch.ones(pad_shape[:-1], device=self.device).long() * (-1)], dim=1)
        n_layers = torch.cat([n_layers, torch.ones(pad_shape[:-1], device=self.device).long() * (-1)], dim=1)
        # how much step that we need. 0, 1, 2, -1
        comp_step = torch.max(n_layers).item()  # 2 # 不算根，迭代两次
        # start from the second to last layer.

        n_layers = n_layers.max(-1)[0].unsqueeze(-1) - n_layers
        # 2, 1, 0, 3 （先处理0， 原先最大的数2）

        for i in range(comp_step):
            # at each step, we update the node_value_embeds and compute a new node_embeds.
            # next_step_embeds acts like a delta.
            node_embeds = self.ensemble_fc(
                torch.cat([node_value_embeds, node_type_embeds], dim=-1))  # B * n_trees * (n_nodes + 1) * hidden_size
            # todo : dropout after fc, and relu
            # node that needs to be used for update
            node_update_idx = (n_layers == 0).float()  # 从离根最远的节点开始
            # node that doesn't need to be used for update
            node_padding_mask = 1 - node_update_idx  # 先不考虑的节点

            # node_att_bias = node_padding_mask * 1e9

            # additive attention
            def attention_score(a, b):
                # B * n_trees * (n_nodes + 1) * hidden_size
                s1 = torch.matmul(b, self.att_w1)

                # L * B * hidden_size
                # todo : only take the last hidden state now.
                s2 = torch.matmul(a[-1], self.att_w2)
                s2 = s2.contiguous().unsqueeze(1).unsqueeze(1)
                # B * n_trees * (n_nodes + 1)
                score = torch.matmul(torch.tanh(s1 + s2), self.v.t()).squeeze(-1)
                # todo: attention add self.att_bias ?
                return score

            # B * n_trees * (n_nodes + 1)
            att_score = attention_score(hidden_state, node_type_embeds)
            old_att_score = att_score
            att_score = att_score * node_update_idx

            next_step_embeds = torch.zeros_like(node_value_embeds,
                                                device=self.device)  # B * n_trees * (n_nodes + 1) * hidden_size

            att_scores = torch.zeros_like(n_layers,
                                          device=self.device,
                                          dtype=torch.float32)  # B * n_trees * (n_nodes + 1)

            back_shape = next_step_embeds.shape
            # [B * n_tree * (n_nodes + 1)] * hidden_size
            next_step_embeds = next_step_embeds.view(-1, self.embedding_dim)
            # [B * n_tree * (n_nodes + 1)]
            att_scores = att_scores.contiguous().view(-1)

            ind = torch.stack([torch.arange(fathers.shape[0] * fathers.shape[1])] * fathers.shape[-1]).t().to(
                self.device)
            # update the -1 item in fathers
            fathers = ((fathers == -1) * fathers.shape[-1]).long() + fathers
            final_ind = ind.contiguous().view(-1) * ind.shape[-1] + fathers.view(-1)

            # compute the max of each tree (layer)
            # B * n_tree
            max_att_score = att_score.max(dim=2)[0]
            # expand as att_score, B * n_tree * (n_nodes + 1)
            max_att_score = max_att_score.unsqueeze(-1).expand_as(att_score)
            # mask unused, cannot be variable, must detach.
            max_att_score = (max_att_score * node_update_idx).detach()
            # modified att_score
            mod_att_score = att_score - max_att_score
            # pdb.set_trace()

            # sum of exponentials
            att_scores = att_scores.index_add(0, final_ind, torch.exp(mod_att_score).contiguous().view(-1))

            # put each divisor at corresponding index.
            tmp_att_scores = torch.index_select(att_scores, 0, final_ind)  # todo: tmp_att_scores too large 1e8

            # add bias to prevent under-flow
            # tmp_att_scores = tmp_att_scores + node_att_bias.contiguous().view(-1)
            tmp_att_scores = tmp_att_scores + 1e-8

            # both over-flow and under-flow needs to be solved.
            att_weights = (torch.exp(mod_att_score).view(-1) / tmp_att_scores) * node_update_idx.contiguous().view(-1)
            try:
                assert torch.isnan(att_weights).sum() == 0
            except:
                pdb.set_trace()
            att_weights = att_weights.contiguous().view(-1, 1)  # todo: att_weights too small

            next_step_embeds = next_step_embeds.index_add(0,
                                                          final_ind,
                                                          node_embeds.contiguous().view(-1,
                                                                                        self.embedding_dim) * att_weights)

            next_step_embeds = next_step_embeds.view(back_shape)  # n_trees * (n_nodes + 1) * hidden_size

            if self.add_skip_con:
                next_step_embeds = next_step_embeds + node_value_embeds
            if self.add_dropout:
                next_step_embeds = self.dropout_layer(next_step_embeds)
            if self.add_relu:
                next_step_embeds = self.relu(next_step_embeds)

            node_value_embeds = node_value_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)

            if self.add_norm:
                node_value_embeds = self.layer_norm(node_value_embeds)

            # todo : add activation
            # update the indicator
            n_layers -= 1
        # only take the root embeds and remove the pivot.
        root_embeds = node_value_embeds[:, :-1, 0]
        attention_bias = padding_idx.float() * -1e9
        return root_embeds, attention_bias.unsqueeze(-1)

    def ptrMemDecoder(self, extKnow, data, hidden, global_index):
        # hidden: 1 * bz * dim, embed : vocab * dim
        # p_vocab: bz * vocab
        p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))

        # hidden = self.relu(self.projector(encode_hidden))
        p_ptr_soft, p_ptr_logits = extKnow(data, hidden.squeeze(0), global_index, encode_or_decode='decode', use_global=self.use_global)

        '''
        temp = []
        u = [hidden[0].squeeze()]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if self.args.use_global:
                m_A = m_A * global_index.unsqueeze(2).expand_as(m_A)
            if (len(list(u[-1].size())) == 1): u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A * u_temp, 2)
            prob_ = self.softmax(prob_lg)
            m_C = self.m_story[hop + 1]
            if self.args.use_global:
                m_C = m_C * global_index.unsqueeze(2).expand_as(m_C)

            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            # if (hop == 0):
            #     p_vocab = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg
        '''
        return p_ptr_logits, p_vocab, hidden

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        return scores_


    def treeDecoder(self, enc_query, hidden_states):
        return


'''
    def ensemble(self, data):
        # ensemble by type and word embeddings.
        """
        :param data:
            dict, contains all features
        :return:
            root_embeds -> B * Nt * hidden_size,
            attention_weights -> B * Nt * 1
        """
        kb_types = data['kb_types']
        kb_fathers = data['kb_fathers']
        kb_values = data['kb_values']
        kb_n_layers = data['kb_n_layers']
        batch_size = len(kb_values)

        root_embeds = []
        for batch_ix in range(batch_size):
            values = kb_values[batch_ix]
            fathers = kb_fathers[batch_ix]
            types = kb_types[batch_ix]
            n_layers = kb_n_layers[batch_ix]
            # todo : not specify the hop value yet.
            # get the embeddings of each tree node.
            node_value_embeds = self.C[0](values)
            # sum over each nodes. BOW now.
            node_value_embeds = node_value_embeds.sum(2)
            node_type_embeds = self.T[0](types)
            node_embeds = node_type_embeds * node_value_embeds

            # create a pivot position.
            pad_shape = list(node_embeds.shape)
            pad_shape[1] = 1
            # use last item as padding
            # shape : n_trees * (n_nodes + 1) * hidden_size
            node_embeds = torch.cat([node_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            # pad fathers too.
            # shape : n_trees * (n_nodes + 1)
            fathers = torch.cat([fathers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # pad n_layers
            n_layers = torch.cat([n_layers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # how much step that we need.
            comp_step = torch.max(n_layers).item()
            # start from the second to last layer.
            n_layers = comp_step - 1 - n_layers

            for i in range(comp_step):
                # next_step_embeds acts like a delta.
                next_step_embeds = torch.zeros_like(node_embeds, device=self.device)  # n_trees * (n_nodes + 1) * hidden_size
                ind = torch.stack([torch.arange(fathers.shape[0])] * fathers.shape[1]).t()
                next_step_embeds.index_put_((ind, fathers), node_embeds, accumulate=True)
                node_embeds = node_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)
                # update the indicator
                n_layers -= 1

            # gather all embedding of roots
            root_embeds.append(node_embeds[:, 0])
        # pad the embedding of roots
        root_embeds = nn.utils.rnn.pad_sequence(root_embeds, batch_first=True)
        # compute the attention bias
        padding_idx = root_embeds.sum(-1) == 0
        attention_bias = padding_idx.float() * -1e9

        return root_embeds, attention_bias.unsqueeze(-1)

    def ensemble_v3(self, data):
        # ensemble type and value by fc layer.
        """
        :param data:
            dict, contains all features
        :return:
            root_embeds -> B * Nt * hidden_size,
            attention_weights -> B * Nt * 1
        """
        kb_types = data['kb_types']
        kb_fathers = data['kb_fathers']
        kb_values = data['kb_values']
        kb_n_layers = data['kb_n_layers']
        batch_size = len(kb_values)

        root_embeds = []
        for batch_ix in range(batch_size):
            values = kb_values[batch_ix]
            fathers = kb_fathers[batch_ix]
            types = kb_types[batch_ix]
            n_layers = kb_n_layers[batch_ix]
            # todo : not specify the hop value yet.
            # get the embeddings of each tree node.
            node_value_embeds = self.C[0](values)
            # sum over each nodes. BOW now.
            node_value_embeds = node_value_embeds.sum(2)
            node_type_embeds = self.T[0](types)

            # create a pivot position.
            pad_shape = list(node_value_embeds.shape)
            pad_shape[1] = 1
            # use last item as padding
            # shape : n_trees * (n_nodes + 1) * hidden_size
            node_value_embeds = torch.cat([node_value_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            node_type_embeds = torch.cat([node_type_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            # pad fathers too.
            # shape : n_trees * (n_nodes + 1)
            fathers = torch.cat([fathers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # pad n_layers
            n_layers = torch.cat([n_layers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # how much step that we need.
            comp_step = torch.max(n_layers).item()
            # start from the second to last layer.
            n_layers = comp_step - 1 - n_layers

            for i in range(comp_step):
                # at each step, we update the node_value_embeds and compute a new node_embeds.
                # next_step_embeds acts like a delta.
                node_embeds = self.ensemble_fc(torch.cat([node_value_embeds, node_type_embeds], dim=-1))
                if self.add_dropout:
                    node_embeds = self.dropout_layer(node_embeds)
                next_step_embeds = torch.zeros_like(node_value_embeds, device=self.device)  # n_trees * (n_nodes + 1) * hidden_size
                back_shape = next_step_embeds.shape
                next_step_embeds = next_step_embeds.view(-1, self.embedding_dim)  # [n_tree * (n_nodes + 1)] * hidden_size
                ind = torch.stack([torch.arange(fathers.shape[0])] * fathers.shape[1]).t().to(self.device)
                # next_step_embeds.index_put_((ind, fathers), node_embeds, accumulate=True)

                # update the -1 item in fathers
                fathers = ((fathers == -1) * fathers.shape[1]).long() + fathers
                final_ind = ind.contiguous().view(-1) * ind.shape[1] + fathers.view(-1)
                next_step_embeds = next_step_embeds.index_add(0,
                                                              final_ind,
                                                              node_embeds.contiguous().view(-1, self.embedding_dim))
                next_step_embeds = next_step_embeds.view(back_shape)  # n_trees * (n_nodes + 1) * hidden_size
                if self.add_relu:
                    next_step_embeds = self.relu(next_step_embeds)

                node_value_embeds = node_value_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)
                # todo : add activation
                # update the indicator
                n_layers -= 1
            # gather all embedding of roots
            # todo : node_value_embeds or node_embeds or what..? need more careful thoughts.
            root_embeds.append(node_value_embeds[:, 0])
        # pad the embedding of roots
        root_embeds = nn.utils.rnn.pad_sequence(root_embeds, batch_first=True)
        # compute the attention bias
        padding_idx = root_embeds.sum(-1) == 0
        attention_bias = padding_idx.float() * -1e9

        return root_embeds, attention_bias.unsqueeze(-1)

    def ensemble_v3_batch(self, data):
        kb_types = data['pad_kb_types']
        kb_fathers = data['pad_kb_fathers']
        kb_values = data['pad_kb_values']
        kb_n_layers = data['pad_kb_n_layers']
        batch_size = len(kb_values)
        padding_idx = kb_n_layers[:, :, 0] == -1

        # get the embeddings of each tree node.
        node_value_embeds = self.C[0](kb_values)
        # sum over each nodes. BOW now.
        node_value_embeds = node_value_embeds.sum(-2)
        node_type_embeds = self.T[0](kb_types)

        # create a pivot position.
        pad_shape = list(node_value_embeds.shape)
        pad_shape[1] = 1
        # use last item as padding
        # shape : B * n_trees * (n_nodes + 1) * hidden_size
        node_value_embeds = torch.cat([node_value_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
        node_type_embeds = torch.cat([node_type_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
        # shape : B * n_trees * (n_nodes + 1)
        fathers = torch.cat([kb_fathers, torch.ones(pad_shape[:-1], device=self.device).long() * (-1)], dim=1)
        n_layers = torch.cat([kb_n_layers, torch.ones(pad_shape[:-1], device=self.device).long() * (-1)], dim=1)
        # how much step that we need.
        comp_step = torch.max(n_layers).item()
        # start from the second to last layer.
        n_layers = comp_step - 1 - n_layers

        for i in range(comp_step):
            # at each step, we update the node_value_embeds and compute a new node_embeds.
            # next_step_embeds acts like a delta.
            node_embeds = self.ensemble_fc(torch.cat([node_value_embeds, node_type_embeds], dim=-1))
            if self.add_dropout:
                node_embeds = self.dropout_layer(node_embeds)
            next_step_embeds = torch.zeros_like(node_value_embeds,
                                                device=self.device)  # B * n_trees * (n_nodes + 1) * hidden_size
            back_shape = next_step_embeds.shape
            # B * [n_tree * (n_nodes + 1)] * hidden_size
            next_step_embeds = next_step_embeds.view(-1, self.embedding_dim)
            ind = torch.stack([torch.arange(fathers.shape[0] * fathers.shape[1])] * fathers.shape[-1]).t().to(self.device)
            # next_step_embeds.index_put_((ind, fathers), node_embeds, accumulate=True)

            # update the -1 item in fathers
            fathers = ((fathers == -1) * fathers.shape[-1]).long() + fathers
            final_ind = ind.contiguous().view(-1) * ind.shape[-1] + fathers.view(-1)
            next_step_embeds = next_step_embeds.index_add(0,
                                                          final_ind,
                                                          node_embeds.contiguous().view(-1, self.embedding_dim))
            next_step_embeds = next_step_embeds.view(back_shape)  # n_trees * (n_nodes + 1) * hidden_size
            if self.add_relu:
                next_step_embeds = self.relu(next_step_embeds)

            node_value_embeds = node_value_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)
            # todo : add activation
            # update the indicator
            n_layers -= 1
        # only take the root embeds and remove the pivot.
        root_embeds = node_value_embeds[:, :-1, 0]
        attention_bias = padding_idx.float() * -1e9
        return root_embeds, attention_bias.unsqueeze(-1)

    def ensemble_v4(self, data):
        # ensemble type and value by fc layer + relu
        # and skip-connection (not implemented yet).
        """
        :param data:
            dict, contains all features
        :return:
            root_embeds -> B * Nt * hidden_size,
            attention_weights -> B * Nt * 1
        """
        kb_types = data['kb_types']
        kb_fathers = data['kb_fathers']
        kb_values = data['kb_values']
        kb_n_layers = data['kb_n_layers']
        batch_size = len(kb_values)

        root_embeds = []
        for batch_ix in range(batch_size):
            values = kb_values[batch_ix]
            fathers = kb_fathers[batch_ix]
            types = kb_types[batch_ix]
            n_layers = kb_n_layers[batch_ix]
            # todo : not specify the hop value yet.
            # get the embeddings of each tree node.
            node_value_embeds = self.C[0](values)
            # sum over each nodes. BOW now.
            node_value_embeds = node_value_embeds.sum(2)
            node_type_embeds = self.T[0](types)
            # node_embeds = node_type_embeds * node_value_embeds
            # create a pivot position.
            pad_shape = list(node_value_embeds.shape)
            pad_shape[1] = 1
            # use last item as padding
            # shape : n_trees * (n_nodes + 1) * hidden_size
            # node_embeds = torch.cat([node_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            node_value_embeds = torch.cat([node_value_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            node_type_embeds = torch.cat([node_type_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            # pad fathers too.
            # shape : n_trees * (n_nodes + 1)
            fathers = torch.cat([fathers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # pad n_layers
            n_layers = torch.cat([n_layers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # how much step that we need.
            comp_step = torch.max(n_layers).item()
            # start from the second to last layer.
            n_layers = comp_step - 1 - n_layers
            for i in range(comp_step):
                # at each step, we update the node_value_embeds and compute a new node_embeds.
                # next_step_embeds acts like a delta.
                node_embeds = self.ensemble_fc(torch.cat([node_value_embeds, node_type_embeds], dim=-1))
                # skip-connection
                node_embeds = self.relu(node_embeds)
                node_embeds += node_value_embeds
                next_step_embeds = torch.zeros_like(node_value_embeds, device=self.device)  # n_trees * (n_nodes + 1) * hidden_size
                back_shape = next_step_embeds.shape
                next_step_embeds = next_step_embeds.view(-1, self.embedding_dim)  # [n_tree * (n_nodes + 1)] * hidden_size
                ind = torch.stack([torch.arange(fathers.shape[0])] * fathers.shape[1]).t().to(self.device)
                # update the -1 item in fathers
                fathers = ((fathers == -1) * fathers.shape[1]).long() + fathers
                final_ind = ind.contiguous().view(-1) * ind.shape[1] + fathers.view(-1)
                next_step_embeds = next_step_embeds.index_add(0,
                                                              final_ind,
                                                              node_embeds.contiguous().view(-1, self.embedding_dim))
                next_step_embeds = next_step_embeds.view(back_shape)  # n_trees * (n_nodes + 1) * hidden_size
                # update the indicator
                node_value_embeds = node_value_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)
                n_layers -= 1
            # gather all embedding of roots
            # todo : node_value_embeds or node_embeds or what..? need more careful thoughts.
            root_embeds.append(node_value_embeds[:, 0])
        # pad the embedding of roots
        root_embeds = nn.utils.rnn.pad_sequence(root_embeds, batch_first=True)
        # compute the attention bias
        padding_idx = root_embeds.sum(-1) == 0
        attention_bias = padding_idx.float() * -1e9

        return root_embeds, attention_bias.unsqueeze(-1)

    def ensemble_v4_batch(self, data):
        kb_types = data['pad_kb_types']
        kb_fathers = data['pad_kb_fathers']
        kb_values = data['pad_kb_values']
        kb_n_layers = data['pad_kb_n_layers']
        batch_size = len(kb_values)
        padding_idx = kb_n_layers[:, :, 0] == -1

        # get the embeddings of each tree node.
        node_value_embeds = self.C[0](kb_values)
        # sum over each nodes. BOW now.
        node_value_embeds = node_value_embeds.sum(-2)
        node_type_embeds = self.T[0](kb_types)

        # create a pivot position.
        pad_shape = list(node_value_embeds.shape)
        pad_shape[1] = 1
        # use last item as padding
        # shape : B * n_trees * (n_nodes + 1) * hidden_size
        node_value_embeds = torch.cat([node_value_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
        node_type_embeds = torch.cat([node_type_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
        # shape : B * n_trees * (n_nodes + 1)
        fathers = torch.cat([kb_fathers, torch.ones(pad_shape[:-1], device=self.device).long() * (-1)], dim=1)
        n_layers = torch.cat([kb_n_layers, torch.ones(pad_shape[:-1], device=self.device).long() * (-1)], dim=1)
        # how much step that we need.
        comp_step = torch.max(n_layers).item()
        # start from the second to last layer.
        n_layers = comp_step - 1 - n_layers

        for i in range(comp_step):
            # at each step, we update the node_value_embeds and compute a new node_embeds.
            # next_step_embeds acts like a delta.
            node_embeds = self.ensemble_fc(torch.cat([node_value_embeds, node_type_embeds], dim=-1))
            # node_embeds += node_value_embeds
            if self.add_relu:
                node_embeds = self.relu(node_embeds)
            next_step_embeds = torch.zeros_like(node_value_embeds,
                                                device=self.device)  # B * n_trees * (n_nodes + 1) * hidden_size
            back_shape = next_step_embeds.shape
            # B * [n_tree * (n_nodes + 1)] * hidden_size
            next_step_embeds = next_step_embeds.view(-1, self.embedding_dim)
            ind = torch.stack([torch.arange(fathers.shape[0] * fathers.shape[1])] * fathers.shape[-1]).t().to(self.device)
            # next_step_embeds.index_put_((ind, fathers), node_embeds, accumulate=True)

            # update the -1 item in fathers
            fathers = ((fathers == -1) * fathers.shape[-1]).long() + fathers
            final_ind = ind.contiguous().view(-1) * ind.shape[-1] + fathers.view(-1)
            next_step_embeds = next_step_embeds.index_add(0,
                                                          final_ind,
                                                          node_embeds.contiguous().view(-1, self.embedding_dim))
            next_step_embeds = next_step_embeds.view(back_shape)  # B * n_trees * (n_nodes + 1) * hidden_size
            if self.add_dropout:
                next_step_embeds = self.dropout_layer(next_step_embeds)

            node_value_embeds = node_value_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)
            # todo : add activation
            # update the indicator
            n_layers -= 1
        # only take the root embeds and remove the pivot.
        root_embeds = node_value_embeds[:, :-1, 0]
        attention_bias = padding_idx.float() * -1e9
        return root_embeds, attention_bias.unsqueeze(-1)

   def ensemble_v2(self, data):
        # ensemble by type and word embeddings.
        """
        :param data:
            dict, contains all features
        :return:
            root_embeds -> B * Nt * hidden_size,
            attention_weights -> B * Nt * 1
        """
        kb_types = data['kb_types']
        kb_fathers = data['kb_fathers']
        kb_values = data['kb_values']
        kb_n_layers = data['kb_n_layers']
        batch_size = len(kb_values)

        root_embeds = []
        for batch_ix in range(batch_size):
            values = kb_values[batch_ix]
            fathers = kb_fathers[batch_ix]
            types = kb_types[batch_ix]
            n_layers = kb_n_layers[batch_ix]
            # todo : not specify the hop value yet.
            # get the embeddings of each tree node.
            node_value_embeds = self.C[0](values)
            # sum over each nodes. BOW now.
            node_value_embeds = node_value_embeds.sum(2)
            node_type_embeds = self.T[0](types)
            # node_embeds = node_type_embeds * node_value_embeds
            # create a pivot position.
            pad_shape = list(node_value_embeds.shape)
            pad_shape[1] = 1
            # use last item as padding
            # shape : n_trees * (n_nodes + 1) * hidden_size
            # node_embeds = torch.cat([node_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            node_value_embeds = torch.cat([node_value_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            node_type_embeds = torch.cat([node_type_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            # pad fathers too.
            # shape : n_trees * (n_nodes + 1)
            fathers = torch.cat([fathers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # pad n_layers
            n_layers = torch.cat([n_layers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # how much step that we need.
            comp_step = torch.max(n_layers).item()
            # start from the second to last layer.
            n_layers = comp_step - 1 - n_layers
            for i in range(comp_step):
                # at each step, we update the node_value_embeds and compute a new node_embeds.
                # next_step_embeds acts like a delta.
                node_embeds = node_value_embeds * node_type_embeds
                # n_trees * (n_nodes + 1) * hidden_size
                next_step_embeds = torch.zeros_like(node_value_embeds, device=self.device)
                ind = torch.stack([torch.arange(fathers.shape[0])] * fathers.shape[1]).t()
                next_step_embeds.index_put_((ind, fathers), node_embeds, accumulate=True)
                node_value_embeds = node_value_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)
                # todo : add activation
                # update the indicator
                n_layers -= 1
            # gather all embedding of roots
            # todo : node_value_embeds or node_embeds or what..? need more careful thoughts.
            root_embeds.append(node_value_embeds[:, 0])
        # pad the embedding of roots
        root_embeds = nn.utils.rnn.pad_sequence(root_embeds, batch_first=True)
        # compute the attention bias
        padding_idx = root_embeds.sum(-1) == 0
        attention_bias = padding_idx.float() * -1e9

        return root_embeds, attention_bias.unsqueeze(-1)

   def ensemble_relation_matrix(self, data):
        # ensemble by type and word embeddings.
        """
        :param data:
            dict, contains all features
        :return:
            root_embeds -> B * Nt * hidden_size,
            attention_weights -> B * Nt * 1
        """
        kb_types = data['kb_types']
        kb_fathers = data['kb_fathers']
        kb_values = data['kb_values']
        kb_n_layers = data['kb_n_layers']
        batch_size = len(kb_values)

        root_embeds = []
        for batch_ix in range(batch_size):
            values = kb_values[batch_ix]
            fathers = kb_fathers[batch_ix]
            types = kb_types[batch_ix]
            n_layers = kb_n_layers[batch_ix]
            # todo : not specify the hop value yet.
            # get the embeddings of each tree node.
            node_value_embeds = self.C[0](values)
            # sum over each nodes. BOW now.
            node_value_embeds = node_value_embeds.sum(2)
            node_type_embeds = self.TM(types)
            # node_embeds = node_type_embeds * node_value_embeds
            # create a pivot position.
            pad_shape = list(node_value_embeds.shape)
            pad_shape[1] = 1
            # use last item as padding
            # shape : n_trees * (n_nodes + 1) * hidden_size
            # node_embeds = torch.cat([node_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            node_value_embeds = torch.cat([node_value_embeds, torch.zeros(pad_shape, device=self.device)], dim=1)
            # n_trees * (n_nodes + 1) * hidden_size^2
            type_pad_shape = pad_shape[:]
            type_pad_shape[-1] = self.embedding_dim * self.embedding_dim
            node_type_embeds = torch.cat([node_type_embeds, torch.zeros(type_pad_shape, device=self.device)], dim=1)
            # pad fathers too.
            # shape : n_trees * (n_nodes + 1)
            fathers = torch.cat([fathers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # pad n_layers
            n_layers = torch.cat([n_layers, torch.ones(pad_shape[:2], device=self.device).long() * (-1)], dim=1)
            # how much step that we need.
            comp_step = torch.max(n_layers).item()
            # start from the second to last layer.
            n_layers = comp_step - 1 - n_layers
            for i in range(comp_step):
                # at each step, we update the node_value_embeds and compute a new node_embeds.
                # next_step_embeds acts like a delta.

                tmp_shape = node_value_embeds.shape
                # [ n_trees * (n_nodes + 1) ] * hidden_size * hidden_size
                node_type_embeds = node_type_embeds.contiguous().view(-1, self.embedding_dim, self.embedding_dim)
                # [ n_trees * (n_nodes + 1) ] * hidden_size
                node_value_embeds = node_value_embeds.contiguous().view(-1, 1, self.embedding_dim)
                # n_trees * (n_nodes + 1) * hidden_size
                node_embeds = torch.bmm(node_value_embeds, node_type_embeds).view(tmp_shape)
                # reshape back
                node_value_embeds = node_value_embeds.view(tmp_shape)
                next_step_embeds = torch.zeros_like(node_value_embeds, device=self.device)  # n_trees * (n_nodes + 1) * hidden_size
                ind = torch.stack([torch.arange(fathers.shape[0])] * fathers.shape[1]).t()
                next_step_embeds.index_put_((ind, fathers), node_embeds, accumulate=True)
                node_value_embeds = node_value_embeds + next_step_embeds * (n_layers == 0).float().unsqueeze(-1)
                # todo : add activation
                # update the indicator
                n_layers -= 1
            # gather all embedding of roots
            # todo : node_value_embeds or node_embeds or what..? need more careful thoughts.
            root_embeds.append(node_value_embeds[:, 0])
        # pad the embedding of roots
        root_embeds = nn.utils.rnn.pad_sequence(root_embeds, batch_first=True)
        # compute the attention bias
        padding_idx = root_embeds.sum(-1) == 0
        attention_bias = padding_idx.float() * -1e9

        return root_embeds, attention_bias.unsqueeze(-1)
    '''


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
