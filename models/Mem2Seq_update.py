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
import datetime
from utils.measures import wer, moses_multi_bleu
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn  as sns
import nltk
import os
from sklearn.metrics import f1_score
import json
from utils.until_temp import entityList
import pdb

torch.manual_seed(10)

"""
Mem2Seq that complies with the dict data. 
"""

class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
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
            self.decoder = DecoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
        # Initialize optimizers and criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        # only use scheduler in decoder
        # but not used in the code below???
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        self.criterion = nn.MSELoss()
        self.loss = 0
        self.loss_ptr = 0
        self.loss_vac = 0
        self.print_every = 1
        self.batch_size = 0
        # Move models to GPU
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_vac = self.loss_vac / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, VL:{:.2f}, PL:{:.2f}'.format(print_loss_avg, print_loss_vac, print_loss_ptr)

    def save_model(self, dec_type):
        name_data = "KVR/" if self.task == '' else "BABI/"
        directory = 'save/mem2seq-' + name_data + str(self.task) + 'HDD' + str(self.hidden_size) + 'BSZ' + str(
            args['batch']) + 'DR' + str(self.dropout) + 'L' + str(self.n_layers) + 'lr' + str(self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')

    '''
    def train_batch(self, input_batches, input_lengths, target_batches,
                    target_lengths, target_index, target_gate, batch_size, clip,
                    teacher_forcing_ratio, reset):

        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss_Vocab, loss_Ptr = 0, 0

        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        max_target_length = max(target_lengths)
        all_decoder_outputs_vocab = Variable(torch.zeros(max_target_length, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(max_target_length, batch_size, input_batches.size(0)))

        # Move new Variables to CUDA
        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            decoder_input = decoder_input.cuda()

        # Choose whether to use teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio

        if use_teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                decoder_input = target_batches[t]  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vacab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vacab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(toppi.view(1, -1)))
                # next_in = [top_ptr_i.squeeze()[i].data[0] if(int(toppi.squeeze()[i].item()) < input_lengths[i]-1) else toppi.squeeze()[i].item() for i in range(batch_size)]
                next_in = [
                    top_ptr_i.squeeze(0)[i].data.item() if (toppi.squeeze(0)[i].item() < input_lengths[i] - 1) else int(
                        toppi.squeeze(0)[i].item()) for i in range(batch_size)]
                decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()

        # Loss calculation and backpropagation
        loss_Vocab = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )
        loss_Ptr = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(),  # -> batch x seq
            target_index.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths
        )

        loss = loss_Vocab + loss_Ptr
        loss.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)
        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        self.loss += loss.item()
        self.loss_ptr += loss_Ptr.item()
        self.loss_vac += loss_Vocab.item()
    '''

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

        # Run words through encoder
        decoder_hidden = self.encoder(data).unsqueeze(0)

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
                decoder_ptr, decoder_vocab, decoder_hidden = self.decoder(decoder_input, data, decoder_hidden)
                # decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                all_decoder_outputs_vocab[t] = decoder_vocab
                all_decoder_outputs_ptr[t] = decoder_ptr
                # target_batches : b * L
                decoder_input = target_batches[:, t]  # Chosen word is next input
                if USE_CUDA: decoder_input = decoder_input.cuda()
        else:
            for t in range(max_target_length):
                # decoder_ptr : b * L
                # decoder_vocab : b * V
                # decoder_hidden : 1 * b * 128
                # decoder_ptr, decoder_vocab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
                decoder_ptr, decoder_vocab, decoder_hidden = self.decoder(decoder_input, data, decoder_hidden)
                _, toppi = decoder_ptr.data.topk(1)
                _, topvi = decoder_vocab.data.topk(1)
                all_decoder_outputs_vocab[t] = decoder_vocab
                all_decoder_outputs_ptr[t] = decoder_ptr
                ## get the correspective word in input
                top_ptr_i = torch.gather(input_batches[:, :, 0], 1, Variable(toppi))
                # todo : in each iteration, the gradient cannot flow back ? check whether it's correct?
                next_in = [top_ptr_i.squeeze(1)[i].data.item() if (toppi.squeeze(1)[i].item() < input_lengths[i] - 1) else int(
                toppi.squeeze(1)[i].item()) for i in range(batch_size)]
                decoder_input = Variable(torch.tensor(next_in, device=cuda_device).long())  # Chosen word is next input
        return all_decoder_outputs_vocab, all_decoder_outputs_ptr


    def evaluate_batch(self, batch_size, input_batches, input_lengths, target_batches, target_lengths, target_index,
                       target_gate, src_plain):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)
        # Run words through encoder
        decoder_hidden = self.encoder(input_batches).unsqueeze(0)
        self.decoder.load_memory(input_batches.transpose(0, 1))

        # Prepare input and output variables
        decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(self.max_r, batch_size, self.output_size))
        all_decoder_outputs_ptr = Variable(torch.zeros(self.max_r, batch_size, input_batches.size(0)))
        # all_decoder_outputs_gate = Variable(torch.zeros(self.max_r, batch_size))
        # Move new Variables to CUDA

        if USE_CUDA:
            all_decoder_outputs_vocab = all_decoder_outputs_vocab.cuda()
            all_decoder_outputs_ptr = all_decoder_outputs_ptr.cuda()
            # all_decoder_outputs_gate = all_decoder_outputs_gate.cuda()
            decoder_input = decoder_input.cuda()

        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)

        self.from_whichs = []
        acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        for t in range(self.max_r):
            decoder_ptr, decoder_vacab, decoder_hidden = self.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
            all_decoder_outputs_vocab[t] = decoder_vacab
            topv, topvi = decoder_vacab.data.topk(1)
            all_decoder_outputs_ptr[t] = decoder_ptr
            topp, toppi = decoder_ptr.data.topk(1)
            top_ptr_i = torch.gather(input_batches[:, :, 0], 0, Variable(toppi.view(1, -1)))
            next_in = [top_ptr_i.squeeze(0)[i].data[0] if (int(toppi.squeeze(0)[i].item()) < input_lengths[i] - 1) else
                       topvi.squeeze(0)[i] for i in range(batch_size)]

            decoder_input = Variable(torch.LongTensor(next_in))  # Chosen word is next input
            if USE_CUDA: decoder_input = decoder_input.cuda()

            temp = []
            from_which = []
            for i in range(batch_size):
                if (int(toppi.squeeze()[i].item()) < len(p[i]) - 1):
                    temp.append(p[i][toppi.squeeze()[i]])
                    from_which.append('p')
                else:
                    ind = topvi.squeeze()[i].item()
                    if ind == EOS_token:
                        temp.append('<EOS>')
                    else:
                        temp.append(self.lang.index2word[ind])
                    from_which.append('v')
            decoded_words.append(temp)
            self.from_whichs.append(from_which)
        self.from_whichs = np.array(self.from_whichs)

        # indices = torch.LongTensor(range(target_gate.size(0)))
        # if USE_CUDA: indices = indices.cuda()

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
        self.encoder.train(True)
        self.decoder.train(True)
        return decoded_words  # , acc_ptr, acc_vac

    def evaluate(self, dev, avg_best, BLEU=False):
        logging.info("STARTING EVALUATION")
        acc_avg = 0.0
        wer_avg = 0.0
        bleu_avg = 0.0
        acc_P = 0.0
        acc_V = 0.0
        microF1_PRED, microF1_PRED_cal, microF1_PRED_nav, microF1_PRED_wet = 0, 0, 0, 0
        microF1_TRUE, microF1_TRUE_cal, microF1_TRUE_nav, microF1_TRUE_wet = 0, 0, 0, 0
        ref = []
        hyp = []
        ref_s = ""
        hyp_s = ""
        dialog_acc_dict = {}

        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
        elif args['dataset'] == 'chitchat':
            global_entity_list = []
        else:
            if int(args["task"]) != 6:
                global_entity_list = entityList('data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', int(args["task"]))
            else:
                global_entity_list = entityList('data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt',
                                                int(args["task"]))

        pbar = tqdm(enumerate(dev), total=len(dev))
        for j, data_dev in pbar:
            if args['dataset'] == 'kvr':
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[6])
            else:
                words = self.evaluate_batch(len(data_dev[1]), data_dev[0], data_dev[1],
                                            data_dev[2], data_dev[3], data_dev[4], data_dev[5], data_dev[6])

            acc = 0
            w = 0
            temp_gen = []

            for i, row in enumerate(np.transpose(words)):
                st = ''
                for e in row:
                    if e == '<EOS>':
                        break
                    else:
                        st += e + ' '
                temp_gen.append(st)
                correct = data_dev[7][i]
                ### compute F1 SCORE
                st = st.lstrip().rstrip()
                correct = correct.lstrip().rstrip()
                if args['dataset'] == 'kvr':
                    f1_true, count = self.compute_prf(data_dev[8][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE += f1_true
                    microF1_PRED += count
                    f1_true, count = self.compute_prf(data_dev[9][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_cal += f1_true
                    microF1_PRED_cal += count
                    f1_true, count = self.compute_prf(data_dev[10][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_nav += f1_true
                    microF1_PRED_nav += count
                    f1_true, count = self.compute_prf(data_dev[11][i], st.split(), global_entity_list, data_dev[14][i])
                    microF1_TRUE_wet += f1_true
                    microF1_PRED_wet += count
                elif args['dataset'] == 'babi' and int(args["task"]) == 6:
                    f1_true, count = self.compute_prf(data_dev[10][i], st.split(), global_entity_list, data_dev[12][i])
                    microF1_TRUE += f1_true
                    microF1_PRED += count

                if args['dataset'] == 'babi':
                    if data_dev[11][i] not in dialog_acc_dict.keys():
                        dialog_acc_dict[data_dev[11][i]] = []
                    if (correct == st):
                        acc += 1
                        dialog_acc_dict[data_dev[11][i]].append(1)
                    else:
                        dialog_acc_dict[data_dev[11][i]].append(0)
                else:
                    if (correct == st):
                        acc += 1
                #    print("Correct:"+str(correct))
                #    print("\tPredict:"+str(st))
                #    print("\tFrom:"+str(self.from_whichs[:,i]))

                w += wer(correct, st)
                ref.append(str(correct))
                hyp.append(str(st))
                ref_s += str(correct) + "\n"
                hyp_s += str(st) + "\n"

            acc_avg += acc / float(len(data_dev[1]))
            wer_avg += w / float(len(data_dev[1]))
            pbar.set_description("R:{:.4f},W:{:.4f}".format(acc_avg / float(len(dev)),
                                                            wer_avg / float(len(dev))))

        # dialog accuracy
        if args['dataset'] == 'babi':
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            logging.info("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))

        if args['dataset'] == 'kvr':
            logging.info("F1 SCORE:\t{}".format(microF1_TRUE / float(microF1_PRED)))
            logging.info("\tCAL F1:\t{}".format(microF1_TRUE_cal / float(microF1_PRED_cal)))
            logging.info("\tWET F1:\t{}".format(microF1_TRUE_wet / float(microF1_PRED_wet)))
            logging.info("\tNAV F1:\t{}".format(microF1_TRUE_nav / float(microF1_PRED_nav)))
        elif args['dataset'] == 'babi' and int(args["task"]) == 6:
            logging.info("F1 SCORE:\t{}".format(microF1_TRUE / float(microF1_PRED)))

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        logging.info("BLEU SCORE:" + str(bleu_score))
        if (BLEU):
            if (bleu_score >= avg_best):
                self.save_model(str(self.name) + str(bleu_score))
                logging.info("MODEL SAVED")
            return bleu_score
        else:
            acc_avg = acc_avg / float(len(dev))
            if (acc_avg >= avg_best):
                self.save_model(str(self.name) + str(acc_avg))
                logging.info("MODEL SAVED")
            return acc_avg

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
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

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))

    def forward(self, data):
        story = data['src_seqs']
        story = story.transpose(0, 1)
        story_size = story.size()  # b * m * 3
        if self.unk_mask:
            if (self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA: a = a.cuda()
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
        return u_k


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
        self.W = nn.Linear(embedding_dim, 1)
        self.W1 = nn.Linear(2 * embedding_dim, self.num_vocab)
        self.gru = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)

    def load_memory(self, story):
        story_size = story.size()  # b * m * 3
        if self.unk_mask:
            if (self.training):
                ones = np.ones((story_size[0], story_size[1], story_size[2]))
                rand_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
                ones[:, :, 0] = ones[:, :, 0] * rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA:
                    a = a.cuda()
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

class Tree2SeqTrainer(object):
    def __init__(self, model, lr):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
        '''
        def pre_hook(module, input):
            pdb.set_trace()
            global forward_cnt
            input[0]['ids'] = forward_cnt % num_parallel_calls
            forward_cnt += 1

        model.register_forward_pre_hook(pre_hook)
        '''

    def train_batch(self, model, data, batch_size, clip, teacher_forcing_ratio, reset):
        """
        input_batches = data['src_seqs']
        input_lengths = data['src_lengths']
        """
        target_batches = data['trg_seqs']
        target_lengths = data['trg_lengths']
        target_index = data['ind_seqs']
        target_gate = data['gate_s']
        kb_trees = data['kb_tree']

        if reset:
            self.loss = 0
            self.loss_ptr = 0
            self.loss_vac = 0
            self.print_every = 1

        if USE_CUDA:
            cuda_device = torch.device('cuda')
        else:
            cuda_device = torch.device('cpu')

        self.batch_size = batch_size
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()
        loss_Vocab, loss_Ptr = 0, 0

        all_decoder_outputs_vocab, all_decoder_outputs_ptr = model(data, teacher_forcing_ratio)

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
        loss.backward()

        # todo: ignore "Clip gradient norms"
        # ec = torch.nn.utils.clip_grad_norm(model, clip)

        # Update parameters with optimizers
        self.optimizer.step()
        self.loss += loss.data.item()
        self.loss_ptr += loss_Ptr.data.item()
        self.loss_vac += loss_Vocab.data.item()

        return loss.data.item(), loss_Ptr.data.item(), loss_Vocab.data.item()


    def evaluate_batch(self, model, data):
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
        decoder_hidden = model.module.encoder(data).unsqueeze(0)
        model.module.decoder.load_memory(input_batches)

        # Prepare input and output variables
        decoder_input = torch.tensor([SOS_token] * batch_size, device=device).long()

        decoded_words = []
        all_decoder_outputs_vocab = Variable(torch.zeros(model.module.max_r, batch_size, model.module.output_size, device=device))
        all_decoder_outputs_ptr = Variable(torch.zeros(model.module.max_r, batch_size, input_batches.size(1), device=device))
        # all_decoder_outputs_gate = Variable(torch.zeros(model.max_r, batch_size))

        p = []
        for elm in src_plain:
            elm_temp = [word_triple[0] for word_triple in elm]
            p.append(elm_temp)

        self.from_whichs = []
        acc_gate, acc_ptr, acc_vac = 0.0, 0.0, 0.0
        # Run through decoder one time step at a time
        for t in range(model.module.max_r):
            decoder_ptr, decoder_vocab, decoder_hidden = model.module.decoder.ptrMemDecoder(decoder_input, decoder_hidden)
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
                        temp.append(model.module.lang.index2word[ind])
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
