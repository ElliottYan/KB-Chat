import pickle
import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.config import *
import logging
import datetime
import ast
import pdb
import copy
import collections
import tqdm
import os

from itertools import chain

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

MEM_TOKEN_SIZE = 5

if USE_CUDA:
    alloc_device = torch.device('cuda')
else:
    alloc_device = torch.device('cpu')


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS", SOS_token: "SOS"}
        self.n_words = 4  # Count default tokens
        self.type2index = {TYPE_PAD_token: "PAD"}
        self.type2count = {}
        self.index2type = {}
        self.n_types = 0

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_trees(self, trees):
        for tree in trees:
            self.index_tree(tree)

    def index_tree(self, tree):
        queue = [tree]
        while queue:
            node = queue.pop()
            # deal with type.
            type = node.type
            if type not in self.type2index:
                self.type2index[type] = self.n_types
                self.type2count[type] = 1
                self.index2type[self.n_types] = type
                self.n_types += 1
            else:
                self.type2count[type] += 1
            queue += node.children

class DatasetNew(data.Dataset):
    def __init__(self, data_dict):
        # check for the length of items in data_dict
        global_features = set(['lang', 'max_len', 'src_word2index', 'trg_word2index'])
        lengths = [len(val) for key, val in data_dict.items() if key not in global_features]
        assert max(lengths) == min(lengths)

        self.length = lengths[0]

        self.data_dict = data_dict
        self.processed_data_dict = collections.defaultdict(list)

        self.keys = self.processed_data_dict.keys()

        self.preprocess_all()

    def __len__(self):
        return len(self.data_dict['src_seqs'])

    def preprocess_all(self):
        # un-used yet.
        spanned_kb_seqs = []
        logging.info("Preprocessing all.")

        max_n_tree = 0
        sum_n_tree = 0
        for i in tqdm.tqdm(range(self.length)):
            src_seq = self.preprocess(self.data_dict['src_seqs'][i], self.data_dict['src_word2index'], trg=False)
            trg_seq = self.preprocess(self.data_dict['trg_seqs'][i], self.data_dict['trg_word2index'])
            sketch_seq = self.preprocess(self.data_dict['sketch_seqs'][i], self.data_dict['trg_word2index'])
            # append token with max_len of src_seq. Don't know why.
            index_s = self.preprocess_inde(self.data_dict['r_index'][i], src_seq)
            gate_s = self.preprocess_gate(self.data_dict['gate'][i])
            conv_seq = self.preprocess(self.data_dict['conversation_arr'][i], self.data_dict['src_word2index'], trg=False)
            kb_plain = self.data_dict['kb_arr'][i]
            # kb_arr = self.preprocess_and_padding(self.data_dict['kb_arr'][i], self.data_dict['src_word2index'], trg=False)
            # mem_kb_arr = self.preprocess(self.data_dict['mem_kb_arr'][i], self.data_dict['src_word2index'], trg=False)
            # exchange kb_arr and mem_kb_arr
            kb_arr = self.preprocess(self.data_dict['kb_arr'][i], self.data_dict['src_word2index'], trg=False)
            mem_kb_arr = self.preprocess_and_padding(self.data_dict['mem_kb_arr'][i], self.data_dict['src_word2index'], trg=False)

            kb_tree, spanned_kb_tree = self.preprocess_tree(self.data_dict['kb_roots'][i], self.data_dict['src_word2index'])
            kb_index = self.preprocess_inde(self.data_dict['kb_index'][i], kb_arr)
            fathers, types, values, n_layers = self.flatten_kb_tree(kb_tree)
            max_n_tree = max(max_n_tree, len(kb_tree))
            sum_n_tree += len(kb_tree)

            self.processed_data_dict['src_seqs'].append(src_seq)
            self.processed_data_dict['trg_seqs'].append(trg_seq)
            self.processed_data_dict['sketch_seqs'].append(sketch_seq)
            self.processed_data_dict['ind_seqs'].append(index_s)
            self.processed_data_dict['gate_s'].append(gate_s)
            self.processed_data_dict['conv_seqs'].append(conv_seq)
            self.processed_data_dict['kb_plain'].append(kb_plain)
            self.processed_data_dict['kb_arr'].append(kb_arr)
            self.processed_data_dict['mem_kb_arr'].append(mem_kb_arr)
            self.processed_data_dict['kb_tree'].append(kb_tree)
            spanned_kb_seqs.append(spanned_kb_tree)
            self.processed_data_dict['kb_ind_seqs'].append(kb_index)
            self.processed_data_dict['kb_fathers'].append(fathers)
            self.processed_data_dict['kb_values'].append(values)
            self.processed_data_dict['kb_types'].append(types)
            self.processed_data_dict['kb_n_layers'].append(n_layers)

            self.processed_data_dict['max_len'].append(self.data_dict['max_len'])

        logging.info("Max number of tree in one sample: {}".format(max_n_tree))
        logging.info("Average number of tree in one sample: {}".format(float(sum_n_tree) / self.length))

        # add other informations
        self.processed_data_dict['src_plain'] = self.data_dict['src_seqs']
        self.processed_data_dict['trg_plain'] = self.data_dict['trg_seqs']
        self.processed_data_dict['entity'] = self.data_dict['ent_index']
        self.processed_data_dict['entity_cal'] = self.data_dict['ent_index_calendar']
        self.processed_data_dict['entity_nav'] = self.data_dict['ent_index_navigation']
        self.processed_data_dict['entity_wet'] = self.data_dict['ent_index_weather']


    def __getitem__(self, item):
        ret = {key: self.processed_data_dict[key][item] for key in self.keys}
        return ret

    def flatten_kb_tree(self, item):
        # process kb trees
        def func(node):
            node.val_idx = torch.tensor(list(node.val_idx), device=alloc_device).long()
            # type is an integer.
            node.type_idx = torch.tensor([node.type_idx], device=alloc_device).long()

        fathers = []
        types = []
        values = []
        n_kbs = len(item)
        n_layers = []

        # if there's no kb item in one dialogue
        if not item:
            n_kbs += 1
            fathers = torch.tensor([-1], device=alloc_device).contiguous().view(1, 1)
            types = torch.tensor([TYPE_PAD_token], device=alloc_device).contiguous().view(1, 1)
            # item in values should be 2D tensor.
            values = torch.tensor([[PAD_token]], device=alloc_device).contiguous().view(1, 1, 1)
            n_layers = torch.tensor([[-1]], device=alloc_device).view(1, 1)

        else:
            for tree in item:
                layer_traverse_tree(tree, func)
                father_item, type_item, value_item, n_layer = get_spanned_tree(tree)
                fathers.append(father_item)
                types.append(type_item)
                values.append(value_item)
                n_layers.append(n_layer)

            fathers = nn.utils.rnn.pad_sequence(fathers, padding_value=-1).squeeze(0)
            n_layers = nn.utils.rnn.pad_sequence(n_layers, padding_value=-1).squeeze(0)
            types = nn.utils.rnn.pad_sequence(types, padding_value=TYPE_PAD_token).squeeze(0).t()
            tmp = [item.t().contiguous().view(-1) for item in values]
            n_nodes = values[0].shape[0]
            # n_kbs * n_nodes * max_value_length
            values = nn.utils.rnn.pad_sequence(tmp, padding_value=PAD_token).t().contiguous().view(n_kbs, -1,
                                                                                                   n_nodes).transpose(1,
                                                                                                                      2)
        return fathers, types, values, n_layers

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        try:
            story = torch.Tensor(story)
        except:
            pdb.set_trace()
            print(sequence)
            print(story)
        return story

    def preprocess_and_padding(self, sequence, word2id, trg=True):
        # the case of empty sequence
        max_len = max([len(word_triple) for word_triple in sequence]) if sequence else 1

        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                length = len(word_triple)
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
                story[i] += [PAD_token] * (max_len - length)

        try:
            story = torch.Tensor(story)
        except:
            pdb.set_trace()
            print(sequence)
            print(story)
        return story

    def preprocess_inde(self, sequence, src_seq):
        """Converts words to ids."""
        # sequence = sequence + [len(src_seq) - 1]
        # sequence =
        sequence = torch.Tensor(sequence)
        return sequence

    # todo : remove last token in gate. keep in mind.
    def preprocess_gate(self, sequence):
        """Converts words to ids."""
        sequence = sequence
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_tree(self, trees, word2id):
        spanned_trees = []
        # traverse the trees in layer-wise order
        for tree in trees:
            queue = [tree]
            ret = []
            while queue:
                node = queue.pop()
                node.type_idx = self.data_dict['lang'].type2index[node.type]
                if node.val:
                    # node.val_idx is a sequence.
                    node.val_idx = [word2id[word] if word in word2id else UNK_token for word in node.val.split(' ')]

                    if node.val == '-':
                        node.type_idx = TYPE_PAD_token
                # each node here is processed in layer wise.
                ret.append(node)
                queue += node.children
            spanned_trees.append(ret)
        # return the root node and spanned tree.
        return trees, spanned_trees

'''
class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, index_seq, gate_seq, src_word2id, trg_word2id, max_len, entity, entity_cal,
                 entity_nav, entity_wet, conv_seq, kb_arr, mem_kb_arr, kb_trees, kb_indexs, lang):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq
        self.gate_seq = gate_seq
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len
        self.entity = entity
        self.entity_cal = entity_cal
        self.entity_nav = entity_nav
        self.entity_wet = entity_wet
        self.conv_seq = conv_seq
        self.kb_arr = kb_arr
        self.mem_kb_arr = mem_kb_arr
        self.kb_trees = kb_trees
        self.lang = lang
        self.n_types = lang.n_types
        self.kb_indexs = kb_indexs

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        # len(index_s) = len(trg_seq)
        index_s = self.index_seqs[index]
        gete_s = self.gate_seq[index]
        conv_seq = self.conv_seq[index]
        # read kb data
        kb_arr = self.kb_arr[index]
        mem_kb_arr = self.mem_kb_arr[index]
        kb_tree = self.kb_trees[index]

        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        # append token with max_len of src_seq. Don't know why.
        index_s = self.preprocess_inde(index_s, src_seq)
        gete_s = self.preprocess_gate(gete_s)

        conv_seq = self.preprocess(conv_seq, self.src_word2id, trg=False)

        kb_plain = kb_arr
        kb_arr = self.preprocess_and_padding(kb_arr, self.src_word2id, trg=False)
        mem_kb_arr = self.preprocess(mem_kb_arr, self.src_word2id, trg=False)
        kb_tree, spanned_kb_tree = self.preprocess_tree(kb_tree, self.src_word2id)

        kb_index = self.preprocess_inde(self.kb_indexs[index], kb_arr)

        return src_seq, trg_seq, index_s, gete_s, self.max_len, self.src_seqs[index], \
               self.trg_seqs[index], self.entity[index], self.entity_cal[index],\
               self.entity_nav[index], self.entity_wet[index], conv_seq, kb_plain, kb_arr, mem_kb_arr, kb_tree, kb_index


    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        try:
            story = torch.Tensor(story)
        except:
            print(sequence)
            print(story)
        return story

    def preprocess_and_padding(self, sequence, word2id, trg=True):
        max_len = max([len(word_triple) for word_triple in sequence])

        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                length = len(word_triple)
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
                story[i] += [PAD_token] * (max_len - length)

        try:
            story = torch.Tensor(story)
        except:
            pdb.set_trace()
            print(sequenfce)
            print(story)
        return story

    def preprocess_inde(self, sequence, src_seq):
        """Converts words to ids."""
        sequence = sequence + [len(src_seq) - 1]
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_gate(self, sequence):
        """Converts words to ids."""
        sequence = sequence + [0]
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_tree(self, trees, word2id):
        spanned_trees = []
        # traverse the trees in layer-wise order
        for tree in trees:
            queue = [tree]
            ret = []
            while queue:
                node = queue.pop()
                node.type_idx = self.lang.type2index[node.type]
                if node.val:
                    # node.val_idx is a sequence.
                    node.val_idx = [word2id[word] if word in word2id else UNK_token for word in node.val.split(' ')]

                    if node.val == '-':
                        node.type_idx = TYPE_PAD_token
                # each node here is processed in layer wise.
                ret.append(node)
                queue += node.children
            spanned_trees.append(ret)
        # return the root node and spanned tree.
        return trees, spanned_trees
'''

# def collate_fn(data):
#     # padding
#     def merge(sequences, max_len):
#         lengths = [len(seq) for seq in sequences]
#         if (max_len):
#             # B * L * MEM_TOKEN_SIZE
#             padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
#             for i, seq in enumerate(sequences):
#                 end = lengths[i]
#                 padded_seqs[i, :end, :] = seq[:end]
#         else:
#             padded_seqs = torch.ones(len(sequences), max(lengths)).long()
#             for i, seq in enumerate(sequences):
#                 end = lengths[i]
#                 padded_seqs[i, :end] = seq[:end]
#         return padded_seqs, lengths
#
#     # sort a list by sequence length (descending order) to use pack_padded_sequence
#     data.sort(key=lambda x: len(x[-1]), reverse=True)
#     # seperate source and target sequences
#     # cool operation.
#     src_seqs, trg_seqs, ind_seqs, gete_s, \
#     max_len, src_plain, trg_plain, entity, \
#     entity_cal, entity_nav, entity_wet, \
#     conv_seq, kb_arr, kb_tree = zip(
#         *data)
#
#     # merge sequences (from tuple of 1D tensor to 2D tensor)
#     src_seqs, src_lengths = merge(src_seqs, max_len)
#     trg_seqs, trg_lengths = merge(trg_seqs, None)
#     ind_seqs, _ = merge(ind_seqs, None)
#     gete_s, _ = merge(gete_s, None)
#     conv_seqs, conv_lengths = merge(conv_seq, max_len)
#
#     src_seqs = Variable(src_seqs).transpose(0, 1)
#     trg_seqs = Variable(trg_seqs).transpose(0, 1)
#     ind_seqs = Variable(ind_seqs).transpose(0, 1)
#     gete_s = Variable(gete_s).transpose(0, 1)
#     conv_seqs = Variable(conv_seqs).transpose(0, 1)
#
#     if USE_CUDA:
#         src_seqs = src_seqs.cuda()
#         trg_seqs = trg_seqs.cuda()
#         ind_seqs = ind_seqs.cuda()
#         gete_s = gete_s.cuda()
#         conv_seqs = conv_seqs.cuda()
#
#     return src_seqs, src_lengths, trg_seqs, trg_lengths, \
#            ind_seqs, gete_s, src_plain, trg_plain, \
#            entity, entity_cal, entity_nav, entity_wet, \
#            conv_seqs, conv_lengths, kb_arr

# this is for old dataset object
# def collate_fn(data):
#     # padding
#     def merge(sequences, max_len):
#         lengths = [len(seq) for seq in sequences]
#         if (max_len):
#             # B * L * MEM_TOKEN_SIZE
#             padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE, device=alloc_device).long()
#             for i, seq in enumerate(sequences):
#                 end = lengths[i]
#                 if not end:
#                     continue
#                 padded_seqs[i, :end, :] = seq[:end]
#         else:
#             padded_seqs = torch.ones(len(sequences), max(lengths), device=alloc_device).long()
#             for i, seq in enumerate(sequences):
#                 end = lengths[i]
#                 if not end:
#                     continue
#                 padded_seqs[i, :end] = seq[:end]
#         return padded_seqs, lengths
#     # sort a list by sequence length (descending order) to use pack_padded_sequence
#     data.sort(key=lambda x: len(x[-2]), reverse=True)
#     # seperate source and target sequences
#     # cool operation.
#     src_seqs, trg_seqs, ind_seqs, gate_s, \
#     max_len, src_plain, trg_plain, entity, \
#     entity_cal, entity_nav, entity_wet, \
#     conv_seq, kb_plain, kb_arr, mem_kb_arr, \
#     kb_tree, kb_index = zip(*data)
#     # merge sequences (from tuple of 1D tensor to 2D tensor)
#     max_len = max(max_len)
#
#     src_seqs, src_lengths = merge(src_seqs, max_len)
#     trg_seqs, trg_lengths = merge(trg_seqs, None)
#     gate_s, _ = merge(gate_s, None)
#     ind_seqs, _ = merge(ind_seqs, None)
#     kb_ind_seqs, _ = merge(kb_index, None)
#     conv_seqs, conv_lengths = merge(conv_seq, max_len)
#     mem_kb_arr = merge(mem_kb_arr, max_len)
#
#     # process kb trees
#     def func(node):
#         node.val_idx = torch.tensor(list(node.val_idx), device=alloc_device).long()
#         # type is an integer.
#         node.type_idx = torch.tensor([node.type_idx], device=alloc_device).long()
#
#     batch_fathers = []
#     batch_types = []
#     batch_values = []
#     batch_n_layers = []
#
#     # process the tree nodes.
#     for batch_item in kb_tree:
#         # batch_item
#         fathers = []
#         types = []
#         values = []
#         n_kbs = len(batch_item)
#         n_layers = []
#
#         # if there's no kb item in one dialogue
#         if not batch_item:
#             n_kbs += 1
#             fathers = torch.tensor([-1], device=alloc_device).contiguous().view(1, 1)
#             types = torch.tensor([TYPE_PAD_token], device=alloc_device).contiguous().view(1, 1)
#             # item in values should be 2D tensor.
#             values = torch.tensor([[PAD_token]], device=alloc_device).contiguous().view(1, 1, 1)
#             n_layers = torch.tensor([[-1]], device=alloc_device).view(1, 1)
#
#         else:
#             for tree in batch_item:
#                 layer_traverse_tree(tree, func)
#                 father_item, type_item, value_item, n_layer = get_spanned_tree(tree)
#                 fathers.append(father_item)
#                 types.append(type_item)
#                 values.append(value_item)
#                 n_layers.append(n_layer)
#
#             fathers = nn.utils.rnn.pad_sequence(fathers, padding_value=-1).squeeze(0)
#             n_layers = nn.utils.rnn.pad_sequence(n_layers, padding_value=-1).squeeze(0)
#             types = nn.utils.rnn.pad_sequence(types, padding_value=TYPE_PAD_token).squeeze(0).t()
#             tmp = [item.t().contiguous().view(-1) for item in values]
#             n_nodes = values[0].shape[0]
#             # n_kbs * n_nodes * max_value_length
#             values = nn.utils.rnn.pad_sequence(tmp, padding_value=PAD_token).t().contiguous().view(n_kbs, -1, n_nodes).transpose(1, 2)
#
#         batch_fathers.append(fathers)
#         batch_types.append(types)
#         batch_values.append(values)
#         batch_n_layers.append(n_layers)
#
#     def padding_2d_sequence(sequence, padding_idx=0):
#         # pad the first two dimension of sequence. allow 2d and 3d sequence.
#         assert len(sequence[0].shape) >= 2
#         batch_size = len(sequence)
#         max_dim_1 = max([item.shape[0] for item in sequence])
#         max_dim_2 = max([item.shape[1] for item in sequence])
#
#         padded_shape = [batch_size, ] + [max_dim_1, max_dim_2,]
#         if len(sequence[0].shape) == 3:
#             max_dim_3 = max([item.shape[2] for item in sequence])
#             padded_shape += [max_dim_3]
#         else:
#             max_dim_3 = 0
#         ret = torch.empty(padded_shape, dtype=torch.int64, device=alloc_device).fill_(padding_idx)
#         for i, item in enumerate(sequence):
#             d1, d2 = item.shape[:2]
#             if not max_dim_3:
#                 ret[i][:d1, :d2] = item
#             else:
#                 d3 = item.shape[2]
#                 ret[i][:d1, :d2, :d3] = item
#         return ret
#
#     padded_batch_fathers = padding_2d_sequence(batch_fathers, -1)
#     padded_batch_n_layers = padding_2d_sequence(batch_n_layers, -1)
#     padded_batch_values = padding_2d_sequence(batch_values, PAD_token)
#     padded_batch_types = padding_2d_sequence(batch_types, TYPE_PAD_token)
#
#     ret = {
#         'src_seqs': src_seqs,
#         'src_lengths': src_lengths,
#         'trg_seqs': trg_seqs,
#         'trg_lengths': trg_lengths,
#         'ind_seqs': ind_seqs,
#         'gate_s': gate_s,
#         'src_plain': src_plain,
#         'trg_plain': trg_plain,
#         'entity': entity,
#         'entity_cal': entity_cal,
#         'entity_nav': entity_nav,
#         'entity_wet': entity_wet,
#         'conv_seqs': conv_seqs,
#         'conv_lengths': conv_lengths,
#         'kb_plain': kb_plain,
#         'kb_arr': kb_arr,
#         'mem_kb_arr': mem_kb_arr,
#         'kb_tree': kb_tree,
#         # list of tensor (1 * N_nodes)
#         'kb_fathers': batch_fathers,
#         # list of tensor (N_nodes)
#         'kb_types': batch_types,
#         # list of tensor (N_nodes * max_len for each node)
#         'kb_values': batch_values,
#         'kb_n_layers': batch_n_layers,
#         'pad_kb_fathers': padded_batch_fathers,
#         'pad_kb_types': padded_batch_types,
#         'pad_kb_values': padded_batch_values,
#         'pad_kb_n_layers': padded_batch_n_layers,
#         'kb_ind_seqs': kb_ind_seqs,
#     }
#
#     return ret


# this is for DatasetNew Object
def collate_fn_new(data):
    # padding
    def merge(sequences, story_dim):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        if (story_dim):
            # B * L * MEM_TOKEN_SIZE
            padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE, device=alloc_device).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                if not end:
                    continue
                padded_seqs[i, :end, :] = seq[:end]
        else:
            padded_seqs = torch.ones(len(sequences), max_len, device=alloc_device).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                if not end:
                    continue
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def padding_2d_sequence(sequence, padding_idx=0):
        # pad the first two dimension of sequence. allow 2d and 3d sequence.
        assert len(sequence[0].shape) >= 2
        batch_size = len(sequence)
        max_dim_1 = max([item.shape[0] for item in sequence])
        max_dim_2 = max([item.shape[1] for item in sequence])

        padded_shape = [batch_size, ] + [max_dim_1, max_dim_2,]
        if len(sequence[0].shape) == 3:
            max_dim_3 = max([item.shape[2] for item in sequence])
            padded_shape += [max_dim_3]
        else:
            max_dim_3 = 0
        ret = torch.empty(padded_shape, dtype=torch.int64, device=alloc_device).fill_(padding_idx)
        for i, item in enumerate(sequence):
            d1, d2 = item.shape[:2]
            if not max_dim_3:
                ret[i][:d1, :d2] = item
            else:
                d3 = item.shape[2]
                ret[i][:d1, :d2, :d3] = item
        return ret

    def merge_index(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths), device=alloc_device).float()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    ret = dict()
    # because it is gonna used for pack-pad sequence.
    data.sort(key=lambda x: len(x['conv_seqs']), reverse=True)
    # merge into dict of lists
    keys = data[0].keys()
    for key in keys:
        ret[key] = [item[key] for item in data]

    ret['max_len'] = max(ret['max_len'])
    ret['src_seqs'], ret['src_lengths'] = merge(ret['src_seqs'], True)
    ret['trg_seqs'], ret['trg_lengths'] = merge(ret['trg_seqs'], False)
    ret['sketch_seqs'], ret['sketch_lengths'] = merge(ret['sketch_seqs'], False)
    ret['gate_s'], _ = merge_index(ret['gate_s'])
    ret['ind_seqs'], _ = merge(ret['ind_seqs'], False)
    ret['kb_ind_seqs'], _ = merge(ret['kb_ind_seqs'], False)
    ret['conv_seqs'], ret['conv_lengths'] = merge(ret['conv_seqs'], True)
    ret['kb_arr'], ret['kb_arr_lengths'] = merge(ret['kb_arr'], True)
    ret['mem_kb_arr'], ret['mem_kb_arr_lengths'] = merge(ret['mem_kb_arr'], True)

    ret['pad_kb_fathers'] = padding_2d_sequence(ret['kb_fathers'], -1)
    ret['pad_kb_n_layers'] = padding_2d_sequence(ret['kb_n_layers'], -1)
    ret['pad_kb_values'] = padding_2d_sequence(ret['kb_values'], PAD_token)
    ret['pad_kb_types'] = padding_2d_sequence(ret['kb_types'], TYPE_PAD_token)

    return ret

def layer_traverse_tree(tree, func):
    queue = [tree]
    ret = []
    while queue:
        node = queue.pop(0)
        if node.val:
            func(node)
        # each node here is processed in layer wise.
        ret.append(node)
        queue += node.children
    return


def get_spanned_tree(tree):
    # todo : the length of kbs is padded to make sure it always exists.
    queue = [tree]
    idx = -1
    father = [idx]
    idx += 1
    ret = []
    values = []
    types = []
    n_layer = []
    n = 0
    while queue:
        # pop the first item in the queue.
        node = queue.pop(0)
        # each node here is processed in layer wise.
        ret.append(node)
        # list of tensor
        values.append(node.val_idx)
        # list of tensor
        types.append(node.type_idx)
        queue += node.children
        # list
        father += [idx] * len(node.children)
        idx += 1
        # update layer index.
        n_layer.append(node.layer)

    father = torch.tensor([father], device=alloc_device).long()
    n_layer = torch.tensor([n_layer], device=alloc_device).long()
    types = torch.cat(types)
    # need padding
    values = nn.utils.rnn.pad_sequence(values, padding_value=PAD_token).t()
    # assert there's no type token with length > 1
    assert values.shape[0] == types.shape[0]
    return father, types, values, n_layer


def read_langs(file_name, tree_file_name, max_line=None):
    logging.info(("Reading lines from {}".format(file_name)))
    data = []
    contex_arr = []
    conversation_arr = []
    kb_arr = []
    old_kb_arr = []
    entity = {}
    u = None
    r = None
    with open(tree_file_name, 'rb') as f:
        kb_roots = pickle.load(f)
    # index all kb_roots

    data_root = os.environ['DATA_ROOT']
    # with open(os.path.join(data_root, 'KVR/kvret_entities.json')) as f:
    with open(os.path.join(data_root, 'KVR/glmp_kvret_entities.json')) as f:
        global_entity = json.load(f)

    with open(file_name) as fin:
        cnt_convs = 0
        cnt_ptr = 0
        cnt_voc = 0
        cnt_kb_ptr = 0
        cnt_non_kb = 0
        max_r_len = 0
        cnt_lin = 1
        user_counter = 0
        system_counter = 0
        system_res_counter = 0
        KB_counter = 0
        dialog_counter = 0
        for line in fin:
            # process KBs
            # just for one layer propagate.
            if not KB_counter:
                kbs = []
                try:
                    kb_root = kb_roots[cnt_convs]
                except:
                    pdb.set_trace()
                for ix, root in enumerate(kb_root):
                    KB_counter += 1

                    def dfs(node, father):
                        ret = []
                        tmp = [node.val, node.type]
                        if father:
                            tmp.append(father.val)
                        else:
                            tmp.append('PAD')

                        child_ret = []
                        if node.children:
                            for item in node.children:
                                tmp.append(item.val)
                                tmp.append(item.type)

                            for child in node.children:
                                child_ret += dfs(child, node)

                        # last item in tmp is the tree index.
                        tmp.append('T_'+str(ix))
                        ret.append(tmp)
                        ret += child_ret

                        return ret

                    kb = dfs(root, None)
                    kbs.append(kb)
                    contex_arr += kb

                # add a token for not choosing any existing kb.
                kbs.append([['NOT_KB', 'T_'+str(len(kbs))]])
                contex_arr.append(['NOT_KB', 'T_'+str(len(kbs))])
                KB_counter += 1
                # store the contex_arr
                kbs_flat = copy.deepcopy(contex_arr)
                new_kb_arr = kbs_flat

                # do not include kb items into context_arr
                contex_arr = []

            line = line.strip()

            if line:
                # indicates the task type
                if '#' in line:
                    line = line.replace("#", "")
                    task_type = line
                    continue
                # split for once.
                nid, line = line.split(' ', 1)
                # it has already been taken care of.
                # if not nid: continue
                if '\t' in line:

                    u, r, gold = line.split('\t')
                    user_counter += 1
                    system_counter += 1

                    # user content as memory.
                    gen_u = generate_memory(u, "$u", str(nid))
                    contex_arr += gen_u
                    conversation_arr += gen_u

                    r_index = []
                    kb_index = []
                    # this is the "selector index" in the GLMP code.
                    gate = []
                    kb_gate = []

                    for key in r.split(' '):
                        # for all, including entity and word appear in context
                        index = [loc for loc, val in enumerate(contex_arr) if (val[0] == key)]
                        # indicate kb_tree index for each word.
                        # # todo : if multiple hits, all of them need to be added into label.
                        kb_ind = [int(val[-1].split('_')[-1]) for loc, val in enumerate(new_kb_arr) if (val[0] == key)]
                        if (index):
                            index = max(index)
                            gate.append(1)
                            cnt_ptr += 1
                        else:
                            index = len(contex_arr)
                            gate.append(0)
                            cnt_voc += 1
                        # cnt for kb_indexes.
                        if kb_ind:
                            kb_ind = max(kb_ind)
                            kb_gate.append(1)
                            cnt_kb_ptr += 1
                        else:
                            kb_ind = len(kbs)
                            kb_gate.append(0)
                            cnt_non_kb += 1

                        r_index.append(index)
                        kb_index.append(kb_ind)
                        system_res_counter += 1

                    if len(r_index) > max_r_len:
                        max_r_len = len(r_index)
                    contex_arr_temp = contex_arr + [['$$$$'] * MEM_TOKEN_SIZE]

                    ent_index_calendar = []
                    ent_index_navigation = []
                    ent_index_weather = []

                    gold = ast.literal_eval(gold)
                    if task_type == "weather":
                        ent_index_weather = gold
                    elif task_type == "schedule":
                        ent_index_calendar = gold
                    elif task_type == "navigate":
                        ent_index_navigation = gold

                    ent_index = list(set(ent_index_calendar + ent_index_navigation + ent_index_weather))

                    # this only consider the entity.
                    ptr_index = []
                    for key in r.split():
                        # only index entity appear in context
                        index = [loc for loc, val in enumerate(contex_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index)
                        else:
                            index = len(contex_arr)
                        ptr_index.append(index)
                    ptr_index = ptr_index + [len(contex_arr)]
                    sketch_response = generate_template(global_entity, r, gold, old_kb_arr, task_type)
                    # each training example is one turn of dialogue

                    # selector index in from input side. gate is from response side.
                    # selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                    #                   contex_arr_temp] + [1]
                    # don't understand the last 1
                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      contex_arr] + [1]

                    # pdb.set_trace()
                    feature = {
                        'src_seqs': contex_arr_temp,
                        'trg_seqs': r,
                        # 'r_index': r_index,
                        'r_index': ptr_index,
                        'gate': selector_index,
                        'ent_index': ent_index,
                        'ent_index_calendar':list(set(ent_index_calendar)),
                        'ent_index_navigation': list(set(ent_index_navigation)),
                        'ent_index_weather': list(set(ent_index_weather)),
                        'conversation_arr': list(conversation_arr),
                        # the same as GLMP and Mem2Seq
                        'kb_arr': list(old_kb_arr),
                        'kb_index': kb_index,
                        'kb_roots': kb_roots[cnt_convs],
                        # kb_arr defined in mem2seq.
                        'mem_kb_arr': list(old_kb_arr),
                        'sketch_seqs': sketch_response,
                    }
                    data.append(feature)
                    selector_words = [contex_arr[i] for i in range(len(contex_arr)) if selector_index[i] == 1]

                    gen_r = generate_memory(r, "$s", str(nid))
                    contex_arr += gen_r
                    conversation_arr += gen_r

                else:
                    # also contain the kb info used in mem2seq.
                    r=line
                    for e in line.split(' '):
                        entity[e] = 0
                    kb_info = generate_memory(r, "", str(nid))
                    old_kb_arr += kb_info
                    # old_contex_arr += kb_info
                    contex_arr = kb_info + contex_arr

            else:
                cnt_lin += 1
                cnt_convs += 1
                if (max_line and cnt_lin >= max_line):
                    break
                contex_arr = []
                conversation_arr = []
                old_kb_arr = []
                dialog_counter += 1
                KB_counter = 0

    max_len = max([len(d['src_seqs']) for d in data])
    logging.info("Pointer percentage= {} ".format(cnt_ptr / (cnt_ptr + cnt_voc)))
    logging.info("KB pointer precentage= {}".format(cnt_kb_ptr / (cnt_non_kb + cnt_kb_ptr)))
    logging.info("Max responce Len: {}".format(max_r_len))
    logging.info("Max Input Len: {}".format(max_len))
    logging.info("Avg. User Utterances: {}".format(user_counter * 1.0 / dialog_counter))
    logging.info("Avg. Bot Utterances: {}".format(system_counter * 1.0 / dialog_counter))
    # logging.info("Avg. KB results: {}".format(KB_counter * 1.0 / dialog_counter))
    logging.info("Avg. responce Len: {}".format(system_res_counter * 1.0 / system_counter))

    print('Sample: ', data[1]['src_seqs'], data[1]['trg_seqs'], data[1]["r_index"], data[1]['gate'], data[1]["ent_index"])
    return data, max_len, max_r_len


def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    code from GLMP: https://github.com/jasonwu0731/GLMP/blob/master/utils/utils_Ent_kvr.py
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            # only replace the entities
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                # ignore it for weather?
                if domain != 'weather':
                    for kb_item in kb_arr:
                        if word == kb_item[0]:
                            ent_type = kb_item[1]
                            break
                # for the case it cannot catch and weather.
                if ent_type == None:
                    for key in global_entity.keys():
                        if key!='poi':
                            global_entity[key] = [x.lower() for x in global_entity[key]]
                            if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                                ent_type = key
                                break
                        else:
                            poi_list = [d['poi'].lower() for d in global_entity['poi']]
                            if word in poi_list or word.replace('_', ' ') in poi_list:
                                ent_type = key
                                break
                try:
                    sketch_response.append('@'+ent_type)
                except:
                    pdb.set_trace()
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_memory(sent, speaker, time):
    ''' Each memory always contains 5 tokens. Each word in dialogue history represents one memory.
    '''
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 't' + str(time), 'word'+str(idx)] + ["PAD"] * (MEM_TOKEN_SIZE - 4)
            sent_new.append(temp)
    else:
        sent_token = sent_token[::-1] + ["PAD"] * (MEM_TOKEN_SIZE - len(sent_token))
        sent_new.append(sent_token)
    return sent_new


# now it returns the dataset.
# def get_seq(pairs, lang, batch_size, type, max_len):
#     x_seq = []
#     y_seq = []
#     ptr_seq = []
#     gate_seq = []
#     entity = []
#     entity_cal = []
#     entity_nav = []
#     entity_wet = []
#     conv_seq = []
#     kb_arr = []
#     mem_kb_arr = []
#     kb_trees = []
#     kb_indexs = []
#
#     for pair in pairs:
#         '''
#             feature = {
#                 'src_seqs': contex_arr_temp,
#                 'trg_seqs': r,
#                 'r_index': r_index,
#                 'gate': gate,
#                 'ent_index': ent_index,
#                 'ent_index_calendar':list(set(ent_index_calendar)),
#                 'ent_index_navigation': list(set(ent_index_navigation)),
#                 'ent_index_weather': list(set(ent_index_weather)),
#                 'conversation_arr': list(conversation_arr),
#                 'kb_arr': list(kb_arr),
#                 'kb_index': kb_index,
#                 'kb_roots': kb_roots[cnt_convs],
#                 # kb_arr defined in mem2seq.
#                 'mem_kb_arr': list(old_kb_arr),
#                 'r_sketch': sketch_response,
#             }
#         '''
#         x_seq.append(pair['src_seqs'])
#         y_seq.append(pair['trg_seqs'])
#         ptr_seq.append(pair['r_index'])
#         gate_seq.append(pair['gate'])
#         entity.append(pair['ent_index'])
#         entity_cal.append(pair['ent_index_calendar'])
#         entity_nav.append(pair['ent_index_navigation'])
#         entity_wet.append(pair['ent_index_weather'])
#         conv_seq.append(pair['conversation_arr'])
#         kb_arr.append(pair['kb_arr'])
#         kb_trees.append(pair['kb_roots'])
#         kb_indexs.append(pair['kb_index'])
#         mem_kb_arr.append(pair['mem_kb_arr'])
#
#         # only train set will produce w2id and etc.
#         if (type):
#             lang.index_words(pair['src_seqs'])
#             lang.index_words(pair['kb_arr'])
#             lang.index_words(pair['trg_seqs'], trg=True)
#             lang.index_trees(pair['kb_roots'])
#
#     # dataset = Dataset(x_seq, y_seq, ptr_seq, gate_seq, lang.word2index, lang.word2index, max_len, entity, entity_cal,
#     #                   entity_nav, entity_wet, conv_seq, kb_arr, mem_kb_arr, kb_trees, kb_indexs, lang)
#     # return dataset
#     return (x_seq, y_seq, ptr_seq, gate_seq, lang.word2index, lang.word2index, max_len, entity, entity_cal,
#                       entity_nav, entity_wet, conv_seq, kb_arr, mem_kb_arr, kb_trees, kb_indexs, lang)

# remove batch_size in parameter.
def get_seq_new(pairs, lang, type, max_len):
    keys = pairs[0].keys()
    for pair in pairs:
        assert pair.keys() == keys
        lang.index_words(pair['src_seqs'])
        # lang.index_words(pair['kb_arr'])
        lang.index_words(pair['trg_seqs'], trg=True)
        lang.index_trees(pair['kb_roots'])
        lang.index_words(pair['sketch_seqs'], trg=True)

    length = len(pairs)
    keyed_pairs = dict()
    for key in keys:
        keyed_pairs[key] = [pairs[i][key] for i in range(length)]
    # global features
    keyed_pairs['src_word2index'] = lang.word2index
    keyed_pairs['trg_word2index'] = lang.word2index
    keyed_pairs['max_len'] = max_len
    keyed_pairs['lang'] = lang
    return keyed_pairs

def read_for_tree(file_path, lang):
    with open(file_path, 'trg_seqs') as fin:
        trees = pickle.load(fin)
    # construct spanned trees.
    type_dict = dict()
    span_trees = []
    for tree in trees:
        queue = [tree]
        ret = []
        while queue:
            node = queue.pop()
            if node.type not in type_dict:
                type_dict[node.type] = len(type_dict)
            node.type_idx = type_dict[node.type]
            if node.val:
                node.val_idx = lang.index_words(node.val)
            ret.append(node)
            queue += node.children
        span_trees.append(ret)
    n_type = len(type_dict)

    return trees, span_trees, n_type


def construct_dataset(args, new=True):
    if new:
        return DatasetNew(args)
    else:
        return Dataset(*args)


def traverse_all_combination(pair):
    # ret = []
    # pdb.set_trace()
    return pair


def prepare_data_seq(args, batch_size=100, shuffle=True):
    splits = ['train', 'dev', 'test']
    txt_files = []
    tree_files = []
    data_root = os.environ['DATA_ROOT']
    for s in splits:
        # todo: confirm difference between kvr_{} and {} files.
        # txt_files.append('data/KVR/kvr_{}.txt'.format(s))
        # txt_files.append(os.path.join(data_root, 'KVR/kvr_{}.txt'.format(s)))
        txt_files.append(os.path.join(data_root, 'KVR/glmp_{}.txt'.format(s)))
        tree_files.append(os.path.join(data_root, 'KVR/{}_example_kbs.dat'.format(s)))

    pair_train, max_len_train, max_r_train = read_langs(txt_files[0], tree_files[0], max_line=None)
    with open('tmp.dat', 'wb') as f:
        pickle.dump(pair_train, f)
    pair_dev, max_len_dev, max_r_dev = read_langs(txt_files[1], tree_files[1], max_line=None)
    pair_test, max_len_test, max_r_test = read_langs(txt_files[2], tree_files[2], max_line=None)

    if args['traverse-all-combination']:
        # padding should be done later. Since data may be changed after traversing all combinations.
        pair_train = traverse_all_combination(pair_train)
        pair_dev = traverse_all_combination(pair_dev)
        pair_test = traverse_all_combination(pair_test)

    max_r_test_OOV = 0
    max_len_test_OOV = 0

    max_len = max(max_len_train, max_len_dev, max_len_test, max_len_test_OOV) + 1
    max_r = max(max_r_train, max_r_dev, max_r_test, max_r_test_OOV) + 1
    lang = Lang()

    # now return is the datasets
    # train_args = get_seq(pair_train, lang, batch_size, True, max_len)
    # dev_args = get_seq(pair_dev, lang, batch_size, False, max_len)
    # test_args = get_seq(pair_test, lang, batch_size, False, max_len)

    if args['debug']:
        pair_train = pair_train[:300]
        pair_dev = pair_dev[:300]
        pair_test = pair_test[:300]

    train_args = get_seq_new(pair_train, lang, True, max_len)
    dev_args = get_seq_new(pair_dev, lang, False, max_len)
    test_args = get_seq_new(pair_test, lang, False, max_len)

    # test_new(train_args, train_args_new)

    train = construct_dataset(train_args)
    dev = construct_dataset(dev_args)
    test = construct_dataset(test_args)

    logging.info("Read %s sentence pairs train" % len(pair_train))
    logging.info("Read %s sentence pairs dev" % len(pair_dev))
    logging.info("Read %s sentence pairs test" % len(pair_test))
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_len, max_r

# test for args1 of old Dataset and args2 of new DatasetNew.
def test_new(args1, args2):
    logging.info('Test for new args.')
    test_keys = ['src_seqs',
                 'trg_seqs',
                 'r_index',
                 'gate',
                 'src_word2index',
                 'trg_word2index',
                 'max_len',
                 'ent_index',
                 'ent_index_calendar',
                 'ent_index_navigation',
                 'ent_index_weather',
                 'conversation_arr',
                 'kb_arr',
                 'mem_kb_arr',
                 'kb_roots',
                 'kb_index',
                 'lang']

    for i, key in enumerate(test_keys):
        logging.info(key)
        assert(args1[i] == args2[key])

