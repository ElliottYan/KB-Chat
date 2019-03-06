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
from utils.config_chitchat import *
import logging
import datetime
import ast
import codecs
from tqdm import tqdm
from collections import defaultdict
import pdb

random.seed(1234)
# MEM_TOKEN_SIZE = 5

# todo : hard code max_len
max_len = 30
max_r = max_len

if USE_CUDA:
    alloc_device = torch.device('cuda')
else:
    alloc_device = torch.device('cpu')

# construct words to index
class Lang:
    def __init__(self, dict_file_path):
        self.word2index = {}
        self.word2count = defaultdict(int)
        # self.index2word = {UNK_token: '[UNK]', PAD_token: "<PAD>", EOS_token: "<EOS>", SOS_token: "<SOS>"}
        self.index2word = {}
        self.n_words = 0  # Count default tokens

        self.dict_file_path = dict_file_path
        self.read_dict_file()

    def read_dict_file(self):
        with codecs.open(self.dict_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # add extra tokens.
        # indicating the token is a keyword.
        lines.append("$k")
        # indicating the token is from input query.
        lines.append('$i')

        # vocab file must includ <SOS>, <EOS>, <PAD> and <UNK>
        n = 0
        for line in lines:
            # cannot use strip directly.
            word = line.strip('\n')
            self.index2word[n] = word
            self.word2index[word] = n
            self.word2count[word] += 1
            self.n_words += 1
            n += 1


'''
class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, src_seq, trg_seq, index_seq, gate_seq, src_word2id, trg_word2id, max_len, entity, entity_cal,
                 entity_nav, entity_wet):
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

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        index_s = self.index_seqs[index]
        gete_s = self.gate_seq[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        index_s = self.preprocess_inde(index_s, src_seq)
        gete_s = self.preprocess_gate(gete_s)

        return src_seq, trg_seq, index_s, gete_s, self.max_len, self.src_seqs[index], self.trg_seqs[index], self.entity[
            index], self.entity_cal[index], self.entity_nav[index], self.entity_wet[index]

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        sequence = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        sequence = torch.Tensor(sequence)
        return sequence

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
'''

class SubDataset(data.Dataset):
    # this dataset contains chitchat data which is already been converted into ids.
    def __init__(self, lang, file_path, weight, max_len):
        self.file_path = file_path
        self.weight = weight
        self.max_len = max_len
        # language model (dictionaries)
        self.lang = lang
        # contains the data
        '''
        self.tgt_seqs = []
        self.src_seqs = []
        self.contexts = []
        self.inputs = []
        '''
        self.dict = []
        self.read_file()

        return

    def __len__(self):
        return len(self.dict)

    def read_file(self):
        # preprocess data, skip empty input and other weird data.
        logging.info("Reading file {}.".format(self.file_path))
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines):
            # choose the data based on some ratio.
            rnd = random.random()
            if rnd > self.weight:
                continue

            contexts, inputs, targets = line.split("@DLM@")

            if len(contexts) > 0:
                contexts = [int(c) for c in contexts.split(" ")]
                contexts = contexts[:self.max_len]
            else:
                contexts = [0]

            if len(inputs) > 0:
                inputs = [int(c) for c in inputs.split(" ")]
                inputs = inputs[:self.max_len]
            else:
                continue

            if len(targets) > 0:
                # add SOS in inputs.
                targets = [SOS_token,] + [int(c) for c in targets.split(" ")]
                targets = targets[:self.max_len - 1]
                # targets.append(EOS_token)
            else:
                continue

            cxt_arr = self.generate_memory(contexts, inputs)

            r_index = []
            # retrieve the index
            for token in targets:
                for idx in range(len(cxt_arr)-1, -1, -1):
                    # the first token
                    ref_token = cxt_arr[idx][0]
                    if ref_token == token:
                        r_index.append(idx)
                        break
                else:
                    r_index.append(len(cxt_arr))

            # '$$$$' has not been added into dictionary yet.
            cxt_arr_tmp = cxt_arr + [[self.lang.word2index['<EOS>']] * MEM_TOKEN_SIZE]
            self.dict.append((cxt_arr_tmp, targets, r_index))

        """
        # not right for tuple of lists...
        
        # removing duplicates
        logging.info("Removing duplicates.")
        self.dict = list(set(self.dict))
        """
        logging.info("Read in {} sentences for file {}.".format(len(self.dict), self.file_path))

    def __getitem__(self, item):
        ret = []
        for get_item in self.dict[item]:
            ret.append(torch.tensor(get_item, device=alloc_device).long())
        src_plain = [self.lang.index2word[item[0]] for item in self.dict[item][0]]
        tgt_plain = [self.lang.index2word[item] for item in self.dict[item][1]]
        return ret + [src_plain, tgt_plain]

    def generate_memory(self, contexts, inputs):
        # we can add more embeddings into this.
        sent_new = []
        for word in contexts:
            temp = [word, self.lang.word2index['$k']] + [self.lang.word2index['<PAD>']] * (MEM_TOKEN_SIZE - 2)
            sent_new.append(temp)
        # todo : no sep token added
        for word in inputs:
            temp = [word, self.lang.word2index['$i']] + [self.lang.word2index['<PAD>']] * (MEM_TOKEN_SIZE - 2)
            sent_new.append(temp)

        return sent_new

    def _get_natural_sentences(self, data):
        # use for debugging
        return [self.lang.index2word[item] for item in data]


class CombinedDataset(data.Dataset):
    def __init__(self, datasets, ratios, task='train'):
        """
        Combined dataset.
        :param datasets: sub-dataset list
        :param ratios: ratio for each sub-dataset
        :param task: define the task {train, dev, test}
        """
        self.task = task
        self.datasets = datasets
        self.ratios = ratios


def collate_fn(data_seqs):
    def merge(sequences,max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len):
            padded_seqs = torch.ones(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i,:end,:] = seq[:end]
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    try:
        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data_seqs.sort(key=lambda x: len(x[0]), reverse=True)
    except:
        pdb.set_trace()
    # seperate source and target sequences
    src_seqs, trg_seqs, ind_seqs, src_plain, trg_plain = zip(
        *data_seqs)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs, max_len)
    trg_seqs, trg_lengths = merge(trg_seqs, None)
    ind_seqs, _ = merge(ind_seqs, None)
    gete_s = None

    # all tensor is transposed for further rnn usage.
    src_seqs = Variable(src_seqs).transpose(0, 1)
    trg_seqs = Variable(trg_seqs).transpose(0, 1)
    ind_seqs = Variable(ind_seqs).transpose(0, 1)
    # gete_s = Variable(gete_s).transpose(0, 1)
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()
        # gete_s = gete_s.cuda()

    entity, entity_cal, entity_nav, entity_wet = [None] * 4
    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, gete_s, src_plain, trg_plain, entity, entity_cal, entity_nav, entity_wet


def get_seq(dataset, lang, batch_size, shuffle, max_len):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


def prepare_data_seq(task, batch_size=100, shuffle=True):
    path_train = 'data/chitchat/data_mixup/train_'
    path_dev = 'data/chitchat/data_mixup/dev_'
    # provide the data mixup rates

    dict_file_path = "data/chitchat/vocab_pool/vocab_filter1000_18394.txt"
    lang = Lang(dict_file_path)



    train_datasets = []
    dev_datasets = []
    for file_name, weight in SRC_WEIGHTS.items():
        train_file = path_train + file_name
        dev_file = path_dev + file_name
        train_sub_dat = SubDataset(lang, train_file, weight, max_len)
        dev_sub_dat = SubDataset(lang, dev_file, weight, max_len)
        train_datasets.append(train_sub_dat)
        dev_datasets.append(dev_sub_dat)

    train_datasets = data.ConcatDataset(train_datasets)
    dev_datasets = data.ConcatDataset(dev_datasets)

    train = get_seq(train_datasets, lang, batch_size, True, max_len)
    dev = get_seq(dev_datasets, lang, batch_size, False, max_len)

    test = None

    logging.info("Read %s sentence pairs train" % train.__len__())
    logging.info("Read %s sentence pairs dev" % dev.__len__())
    logging.info("Read %s sentence pairs test" % 0)
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, [], lang, max_len, max_r

