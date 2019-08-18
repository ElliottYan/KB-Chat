import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
from utils.masked_cross_entropy import *
from utils.config import *
import random
import numpy as np
from utils.logging import logger
import datetime
from utils.measures import wer, moses_multi_bleu
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pdb
import os
from sklearn.metrics import f1_score
import json
import math
import time

from utils.until_temp import entityList

cnt = 0

import operator
from functools import reduce

def prod(factors):
    return reduce(operator.mul, factors, 1)

random.seed(1234)
torch.manual_seed(1234)

np.random.seed(1234)


class Tree2Seq(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask, args):
        super(Tree2Seq, self).__init__()
        # if args.gpu:
        #     self.device = torch.device(args.gpu)
        # else:
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')

        self.name = "Tree2Seq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.n_types = lang.n_types
        self.hidden_size = hidden_size
        self.max_len = max_len  ## max input
        self.max_r = max_r  ## max responce len
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask

        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th')
                self.decoder = torch.load(str(path) + '/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path) + '/enc.th', lambda storage, loc: storage)
                self.decoder = torch.load(str(path) + '/dec.th', lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            # self.encoder = EncoderTreeSpanNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask, device=self.device)
            # self.encoder = EncoderTreeNN(lang.n_words, lang.n_types, hidden_size, n_layers, self.dropout, self.unk_mask)
            # self.decoder = DecoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderTreeNN(lang.n_words, lang.n_types, hidden_size, n_layers, self.dropout, self.unk_mask, args=args)

        # Initialize optimizers and criterion
        # self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        # self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        # only anneal the decoder lr rate...
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
        #                                                 min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.to(self.device)
            self.decoder.to(self.device)

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)

    def save_model(self, dec_type):
        name_data = "KVR/" if self.task == '' else "BABI/"
        directory = 'save/tree2seq-' + name_data + str(self.task) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(
            args.batch) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'lr' + str(self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    def forward(self, data, teacher_forcing_ratio):
        input_batches = data['src_seqs']
        input_lengths = data['src_lengths']
        target_batches = data['trg_seqs']
        target_lengths = data['trg_lengths']
        target_index = data['ind_seqs']
        target_gate = data['gate_s']

        kb_trees = data['kb_tree']

        if USE_CUDA:
            cuda_device = torch.device('cuda')
        else:
            cuda_device = torch.device('cpu')

        batch_size = len(input_batches)

        global_index, decoder_hidden = self.encoder(data)  # L * B * D

        # decoder take inputs as memory.
        self.decoder.load_memory(input_batches)

        # Prepare input and output variables
        decoder_input = torch.tensor([SOS_token] * batch_size, device=cuda_device).long()

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = torch.zeros(max_target_length,
                                                batch_size,
                                                self.output_size,
                                                device=cuda_device)

        all_decoder_outputs_ptr = torch.zeros(max_target_length,
                                                batch_size,
                                                input_batches.size(1),
                                                device=cuda_device)

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        # use_teacher_forcing = True

        # what's teacher forcing?
        # teacher forcing means which to use, target sequence or generated results.
        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vocab, decoder_hidden = self.decoder(decoder_input, data, decoder_hidden, global_index)
                # decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vocab
                all_decoder_outputs_ptr[t] = decoder_ptr
                # target_batches : b * L
                decoder_input = target_batches[:, t]  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                # global cnt
                # cnt += 1
                # print(cnt)
                # if cnt == 45:
                #     pdb.set_trace()

                # decoder_ptr : b * L
                # decoder_vocab : b * V
                # decoder_hidden : 1 * b * 128
                # decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                decoder_ptr, decoder_vocab, decoder_hidden = self.decoder(decoder_input, data, decoder_hidden, global_index)
                if torch.isnan(decoder_ptr).sum() != 0 or torch.isnan(decoder_vocab).sum() != 0:
                    continue
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vocab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vocab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                '''
                debug info
                logger.info(toppi.shape)
                logger.info(toppi.min())
                logger.info(toppi.max())
                logger.info(torch.isnan(toppi).sum())
                logger.info(decoder_ptr.data)
                logger.info(input_batches.shape)
                logger.info('\n')
                '''
                top_ptr_i = torch.gather(input_batches[:, :, 0], 1, Variable(toppi))
                # todo : in each iteration, the gradient cannot flow back ? check whether it's correct?
                next_in = [top_ptr_i.squeeze(1)[i].data.item() if (toppi.squeeze(1)[i].item() < input_lengths[i] - 1) else int(
                toppi.squeeze(1)[i].item()) for i in range(batch_size)]
                decoder_input = Variable(torch.tensor(next_in, device=cuda_device).long())  # Chosen word is next input

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, global_index

    @staticmethod
    def compute_prf(gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            # count the golden word that appeared in prediction.
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            # count wrong entities
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
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

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        # not sure whether to add 'requires_grad'
        return torch.zeros(bsz, self.embedding_dim, device=self.device, requires_grad=True)

    def forward(self, data):
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
        u = [self.get_state(story.size(0))]
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
        return self.sigmoid(m_C), u_k.unsqueeze(0)

class EncoderTreeNN(nn.Module):
    def __init__(self, vocab, n_types, embedding_dim, hop, dropout, unk_mask):
        super(EncoderTreeNN, self).__init__()
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
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.type_embed = nn.Embedding(n_types, embedding_dim, padding_idx=TYPE_PAD_TOKEN)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        # not sure whether to add 'requires_grad'
        return torch.zeros(bsz, self.embedding_dim, device=self.device, requires_grad=True)

    def forward(self, data):
        # story = data['src_seqs']
        story = data['conv_seqs']
        # story = story.transpose(0, 1)
        story_size = story.size()  # b * m * 5
        # tree_embeds = self.load_tree_embedding(data)
        if self.unk_mask:
            if (self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.tensor(ones, device=self.device))
                story = story * a.long()
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            # load tree memories for each hop.
            tree_mems_A = self.load_tree_memory(data, hop, is_key=True)
            embed_A = self.C[hop](story.contiguous().view(story.size(0), -1).long())  # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
            # add through axis of 5.
            m_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
            m_A = torch.cat([m_A, tree_mems_A], dim=1)

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A * u_temp, 2))
            tree_mems_C = self.load_tree_memory(data, hop, is_key=False)
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = torch.cat([m_C, tree_mems_C], dim=1)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return u_k

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

class EncoderTreeSpanNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask, device):
        super(EncoderTreeSpanNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask
        # Embedding for conv seqs.
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        for hop in range(self.max_hops + 1):
            K = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            K.weight.data.normal_(0, 0.1)
            self.add_module("K_{}".format(hop), K)
        self.K = AttrProxy(self, "K_")

        self.softmax = nn.Softmax(dim=1)
        self.device = device
        # self.type_embed = nn.Embedding(n_types, embedding_dim, padding_idx=TYPE_PAD_TOKEN)
        # self.w1 = nn.Parameter(torch.randn(MEM_TOKEN_SIZE, self.embedding_dim))
        # self.w2 = nn.Parameter(torch.randn(MEM_TOKEN_SIZE, self.embedding_dim))

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        # not sure whether to add 'requires_grad'
        return torch.zeros(bsz, self.embedding_dim, device=self.device, requires_grad=True)

    def forward(self, data):

        # kb_arr is a tuple...
        kb_arr = data['kb_arr']
        conv_seqs = data['conv_seqs']

        bs = len(conv_seqs)
        # todo : how to update query?
        # query_state = self.get_state(bs).contiguous().view(bs, 1, self.embedding_dim) # B * 1 * D
        conv_feature = torch.zeros(bs, conv_seqs.shape[1], self.embedding_dim, device=self.device) # B * L * D
        for hop in range(self.max_hops):
            # self.hop_func(hop, conv_seqs, kb_arr, query_state)
            conv_feature = self.hop_func_simple(hop, conv_seqs, kb_arr, conv_feature)
        return conv_feature.transpose(0, 1)

    '''
    def hop_func(self, hop, conv_seqs, kb_arr, query_state):
        # unfinished yet.
        bs = len(conv_seqs)
        m_c = conv_seqs.shape[-1]
        m_k = kb_arr.shape[-1]
        conv_feature = self.C[hop](conv_seqs) # B * L_c * M_c * D
        kb_feature = self.K[hop](kb_arr) # B * L_k * M_k * D
        conv_feature = conv_feature.contiguous().view(bs, -1, self.embedding_dim) # B * (L_c * M_c) * D
        kb_feature = kb_feature.contiguous().view(bs, -1, self.embedding_dim) # B * (L_k * M_k) * D

        # dynamic attention
        att_feat = torch.bmm(conv_feature, query_state.transpose(1, 2)).contiguous().view(bs, -1, m_c) # B * L_c * M_c
        att_prob = F.softmax(att_feat, -1).unsqueeze(-1) # B * L_c * M_c * 1
        conv_feature = torch.sum(att_prob * conv_feature, dim=2) # B * L_c * D

        # dynamic attention
        att_feat = torch.bmm(kb_feature, query_state.transpose(1, 2)).contiguous().view(bs, -1, m_k) # B * L_k * M_k
        att_prob = F.softmax(att_feat, -1).unsqueeze(-1) # B * L_k * M_k * 1
        kb_feature = torch.sum(att_prob * kb_feature, dim=2) # B * L_k * D

        # update conv_feature using kb_feature
        att_feat = torch.bmm(conv_feature, kb_feature.transpose(1, 2)).contiguous().view(bs, -1, L_k) # B * L_c * L_k
        att_prob = F.softmax(att_feat, -1) # B * L_c * L_k
        # todo : add activation ?
        conv_feature = torch.sum(att_prob * kb_feature, dim=-1) + conv_feature
    '''

    def hop_func_simple(self, hop, conv_seqs, kb_arr, prev_conv_feature):
        bs = len(conv_seqs)
        m_c = conv_seqs.shape[-1]
        # m_k = kb_arr.shape[-1]
        # get kb features
        conv_feature = self.C[hop](conv_seqs)  # B * L_c * M_c * D
        conv_feature = torch.sum(conv_feature, dim=2)  # B * L_c * M_c * D

        kb_feature = []
        for i in range(bs):
            item_kb_arr = kb_arr[i].to(self.device).long()
            item_kb_feature = self.K[hop](item_kb_arr)  # L_k * M_k * D
            item_kb_feature = torch.sum(item_kb_feature, dim=1) # L_k * D
            kb_feature.append(item_kb_feature)
        kb_feature = nn.utils.rnn.pad_sequence(kb_feature).transpose(0, 1)  # B * L_k * D
        L_k = kb_feature.shape[1]
        # update conv_feature using kb_feature
        att_feat = torch.bmm(conv_feature, kb_feature.transpose(1, 2)).contiguous().view(bs, -1, L_k)  # B * L_c * L_k
        att_prob = F.softmax(att_feat, -1)  # B * L_c * L_k
        conv_feature = torch.sum(att_prob.unsqueeze(-1) * kb_feature.unsqueeze(1), dim=-2)  # B * L_c * D
        return conv_feature + prev_conv_feature


class DecoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(DecoderMemNN, self).__init__()
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
        # self.W = nn.Linear(embedding_dim, 1)
        self.W1 = nn.Linear(2 * embedding_dim, self.num_vocab)
        # todo : batch_first
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')

    # load the origin inputs.
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
            embed_C = self.C[hop + 1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            m_C = embed_C
            self.m_story.append(m_A)
        self.m_story.append(m_C)

    def ptrMemDecoder(self, enc_query, last_hidden):
        embed_q = self.C[0](enc_query)  # b * e
        output, hidden = self.gru(embed_q.unsqueeze(0), last_hidden)
        temp = []
        u = [hidden[0].squeeze()]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if (len(list(u[-1].size())) == 1): u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A * u_temp, 2)
            prob_ = self.softmax(prob_lg)
            m_C = self.m_story[hop + 1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            if (hop == 0):
                p_vocab = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg
        return p_ptr, p_vocab, hidden


class DecoderTreeNN(nn.Module):
    def __init__(self, vocab, n_type, embedding_dim, hop, dropout, unk_mask, device=None, args=None):
        super(DecoderTreeNN, self).__init__()
        self.args = args
        self.num_vocab = vocab
        self.num_type = n_type
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        self.unk_mask = unk_mask
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.T = nn.Embedding(self.num_type, embedding_dim, padding_idx=TYPE_PAD_token)
        self.T.weight.data.normal_(0, 0.1)
        self.C = AttrProxy(self, "C_")
        # C[max_hop] will not be used for updates
        if self.max_hops > 1:
            logger.info("Max hops : {}".format(self.max_hops))
            self.C[self.max_hops].weight.requires_grad = False
        # self.T = AttrProxy(self, "T_")
        # self.TM = nn.Embedding(self.num_type, embedding_dim * embedding_dim, padding_idx=TYPE_PAD_token)
        self.softmax = nn.Softmax(dim=1)
        # self.W = nn.Linear(embedding_dim, 1)
        self.W1 = nn.Linear(2 * embedding_dim, self.num_vocab)
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

    # load the origin inputs.
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

    def forward(self, decoder_input, data, hidden_states, global_index):
        # for each time step, we compute the kb_attn_features based on current hidden state.
        # todo : has accumulating problem ??
        # global cnt
        # if cnt >= 45:
        #     pdb.set_trace()
        kb_attn_features, kb_attn_weights = self.compute_global_ranking(data, hidden_states)
        # kb_attn_features, kb_attn_weights = self.compute_global_ranking_v2(data, hidden_states)
        # current state
        embed_q = self.C[0](decoder_input)  # b * e
        # gru for update hidden state.
        output, hidden = self.gru(embed_q.unsqueeze(0), hidden_states)
        cur_state = hidden + kb_attn_features.unsqueeze(0)

        p_ptr, p_vocab, decoder_hidden = self.ptrMemDecoder(cur_state)
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
        node_value_embeds = self.C[0](values)
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
        comp_step = torch.max(n_layers).item() # 2 # 不算根，迭代两次
        # start from the second to last layer.

        n_layers = n_layers.max(-1)[0].unsqueeze(-1) - n_layers
        # 2, 1, 0, 3 （先处理0， 原先最大的数2）
        # pdb.set_trace()

        for i in range(comp_step):
            # at each step, we update the node_value_embeds and compute a new node_embeds.
            # next_step_embeds acts like a delta.
            node_embeds = self.ensemble_fc(torch.cat([node_value_embeds, node_type_embeds], dim=-1))  # B * n_trees * (n_nodes + 1) * hidden_size
            # todo : dropout after fc, and relu
            # node that needs to be used for update
            node_update_idx = (n_layers == 0).float() # 从离根最远的节点开始
            # node that doesn't need to be used for update
            node_padding_mask = 1 - node_update_idx # 先不考虑的节点
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

            # sum of exponentials
            att_scores = att_scores.index_add(0,final_ind,torch.exp(att_score).contiguous().view(-1))

            # put each divisor at corresponding index.
            tmp_att_scores = torch.index_select(att_scores, 0, final_ind) # todo: tmp_att_scores too large 1e8

            # add bias to prevent under-flow
            # tmp_att_scores = tmp_att_scores + node_att_bias.contiguous().view(-1)
            tmp_att_scores = tmp_att_scores + 1e-8

            # both over-flow and under-flow needs to be solved.
            att_weights = (torch.exp(att_score).view(-1) / tmp_att_scores) * node_update_idx.contiguous().view(-1)
            att_weights = att_weights.contiguous().view(-1, 1) # todo: att_weights too small

            next_step_embeds = next_step_embeds.index_add(0,
                                                          final_ind,
                                                          node_embeds.contiguous().view(-1, self.embedding_dim) * att_weights)

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


    def ptrMemDecoder(self, hidden):
        temp = []
        u = [hidden[0].squeeze()]
        for hop in range(self.max_hops):
            m_A = self.m_story[hop]
            if (len(list(u[-1].size())) == 1): u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob_lg = torch.sum(m_A * u_temp, 2)
            prob_ = self.softmax(prob_lg)
            m_C = self.m_story[hop + 1]
            temp.append(prob_)
            prob = prob_.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            # only use the first hop for p_vocab ??
            if (hop == 0):
                p_vocab = self.W1(torch.cat((u[0], o_k), 1))
            u_k = u[-1] + o_k
            u.append(u_k)
        p_ptr = prob_lg
        return p_ptr, p_vocab, hidden

    def treeDecoder(self, enc_query, hidden_states):
        return


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

class Tree2SeqTrainer(object):
    def __init__(self, model, lr, args=None):
        model_parameters = [para for para in model.parameters() if para.requires_grad]
        self.optimizer = optim.Adam(model_parameters, lr=lr)

        self.args = args
        # if args.gpu:
        #     self.device = torch.device(args.gpu)
        # else:
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        '''
        def pre_hook(module, input):
            pdb.set_trace()
            global forward_cnt
            input[0]['ids'] = forward_cnt % num_parallel_calls
            forward_cnt += 1

        model.register_forward_pre_hook(pre_hook)
        '''
        self.criterion_bce = nn.BCELoss()

    def train_batch(self, model, data, batch_size, clip, teacher_forcing_ratio, reset):
        """
        input_batches = data['src_seqs']
        input_lengths = data['src_lengths']
        """
        # for debugging in backward.
        # torch.autograd.set_detect_anomaly(True)

        # target_batches = data['trg_seqs']
        target_batches = data['sketch_seqs']
        target_lengths = data['trg_lengths']
        target_index = data['ind_seqs']
        target_gate = data['gate_s']
        kb_trees = data['kb_tree']

        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()
        loss_Vocab, loss_Ptr = 0, 0

        all_decoder_outputs_vocab, all_decoder_outputs_ptr, global_index = model(data, teacher_forcing_ratio)
        
        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),  # -> batch x seq
            target_index.contiguous(),  # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        pdb.set_trace()
        if self.args.use_global_loss:
            loss_g = self.criterion_bce(global_index, target_gate)
            loss += loss_g

        loss.backward()

        # todo: ignore "Clip gradient norms"

        # Update parameters with optimizers
        if self.args.distributed:
            model = model.module

        clip = 1
        # ec = torch.nn.utils.clip_grad_value_(model.encoder.parameters(), clip)
        # dc = torch.nn.utils.clip_grad_value_(model.decoder.parameters(), clip)
        # ec = nn.utils.clip_grad_value_(model.parameters(), 1)
        self.optimizer.step()

        self.loss += loss.data.item()
        self.loss_ptr += loss_Ptr.data.item()
        self.loss_vac += loss_Vocab.data.item()

        return loss.data.item(), loss_Ptr.data.item(), loss_Vocab.data.item()

    def evaluate_batch(self, model, data):
        if self.args and self.args.distributed:
            model = model.module
        input_batches = data['src_seqs']
        input_lengths = data['src_lengths']
        target_batches = data['trg_seqs']
        target_lengths = data['trg_lengths']
        target_index = data['ind_seqs']
        target_gate = data['gate_s']
        src_plain = data['src_plain']

        kb_trees = data['kb_tree']

        batch_size = len(input_batches)

        device = torch.device('cuda' if USE_CUDA else 'cpu')
        # Run words through encoder
        decoder_hidden = model.encoder(data)
        model.decoder.load_memory(input_batches)

        # Prepare input and output variables
        decoder_input = torch.tensor([SOS_token] * batch_size, device=self.device).long()

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(model.max_r, batch_size, model.output_size, device=device))
        all_decoder_outputs_ptr = Variable(torch.zeros(model.max_r, batch_size, input_batches.size(1), device=device))
        # all_decoder_outputs_gate = Variable(torch.zeros(model.max_r, batch_size))

        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)

        self.from_whichs = []
        acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        for t in range(model.max_r):
            # decoder_ptr, decoder_vocab, decoder_hidden = model.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            decoder_ptr, decoder_vocab, decoder_hidden = model.decoder(decoder_input, data, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vocab
            topv, topvi = decoder_vocab.data.topk(1)
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:, :, 0], 1, Variable(toppi.view(-1, 1)))
            next_in = [top_ptr_i.squeeze(0)[i].data.item() if (toppi.squeeze(0)[i].item() < input_lengths[i] - 1) else
                topvi.squeeze(0)[i] for i in range(batch_size)]

            decoder_input = torch.tensor(next_in, device=device)  # Chosen word is next input

            temp = []
            from_which = []
            for i in range(batch_size):
                if (toppi.squeeze(0)[i].item() < len(p[i]) - 1):
                    temp.append(p[i][toppi.squeeze(0)[i]])
                    from_which.append('p')
                else:
                    ind = topvi.squeeze(0)[i].item()
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(model.lang.index2word[ind])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)

        # ## acc pointer
        # y_ptr_hat = all_decoder_outputs_ptr.topk(1)[1].squeeze()
        # y_ptr_hat = torch.index_select(y_ptr_hat, 0, indices)
        # y_ptr = target_index
        # acc_ptr = y_ptr.eq(y_ptr_hat).sum()
        # acc_ptr = acc_ptr.data[0]/(y_ptr_hat.size(0)*y_ptr_hat.size(1))
        # ## acc vocab
        # y_vac_hat = all_decoder_outputs_vocab.topk(1)[1].squeeze()
        # y_vac_hat = torch.index_select(y_vac_hat, 0, indices)
        # y_vac = target_batches
        # acc_vac = y_vac.eq(y_vac_hat).sum()
        # acc_vac = acc_vac.data[0]/(y_vac_hat.size(0)*y_vac_hat.size(1))

        # Set back to training mode
        # model.encoder.train(True)
        # model.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)
