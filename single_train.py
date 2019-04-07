import os
import random
import shutil
import time
import warnings
from tqdm import tqdm, trange
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from utils.logging import logger
from utils.statistics import Statistics
from models.Tree2Seq import *
# from models.Mem2Seq_update import *

import utils.utils_kvr_tree as utils_tree

def main_worker(args, gpu):
    # global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = gpu

    # read in the dataset
    # todo : clean this logic
    prepare_data_seq = utils_tree.prepare_data_seq
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(vars(args),batch_size=int(args.batch),shuffle=True)

    # create model
    model = globals()[args.decoder](int(args.hidden),
                                        max_len,max_r,lang,args.path,args.task,
                                        lr=float(args.learn),
                                        n_layers=int(args.layer),
                                        dropout=float(args.drop),
                                        unk_mask=bool(int(args.unk_mask))
                                        )


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch = int( args.batch / args.world_size)
            args.workers = int( args.workers / args.world_size)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.cuda()



    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)



    '''
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    '''

    cudnn.benchmark = True

    # Data loading code

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train)
    else:
        train_sampler = None

    # multiple workers are not allowed in multiprocessing !
    train_loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch, shuffle=(train_sampler is None),
            pin_memory=False, sampler=train_sampler, collate_fn=utils_tree.collate_fn_new)

    val_loader = torch.utils.data.DataLoader(dev,
            batch_size=args.batch, shuffle=False,
            pin_memory=False, collate_fn=utils_tree.collate_fn_new)


    '''
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    '''

    for epoch in range(0, 50):
        logger.info('In the training process now.')
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, args)

        trainer = Tree2SeqTrainer(model, lr=float(args.learn))

        # train for one epoch
        train_one_epoch(train_loader, model, trainer, epoch, args)

        # evaluate on validation set
        bleu_score = validate_one_epoch(val_loader, model, trainer, args)

        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #                                             and args.rank % args.world_size == 0):
        '''
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        '''
        # pass


def train_one_epoch(train_loader, model, trainer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    Losses = AverageMeter('Loss', ':.4e')
    V_Loss = AverageMeter('VL', ':6.2f')
    P_Loss = AverageMeter('PL', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, Losses, V_Loss,
                             P_Loss, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # input = input.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        loss = trainer.train_batch(model, data, len(data['src_seqs']), 10.0, 0.5, i == 0)

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        Losses.update(loss[0], data['src_seqs'].size(0))
        P_Loss.update(loss[1], data['src_seqs'].size(0))
        V_Loss.update(loss[2], data['src_seqs'].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate_one_epoch(val_loader, model, trainer, args):
    # switch to evaluate mode
    model.eval()

    val_stats = [Statistics() for i in range(5)]

    # read-in global entity list
    if args.dataset == 'kvr':
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
    else:
        raise NotImplementedError('Not implemented this val for datasets other than kvr yet.')

    with torch.no_grad():
        end = time.time()
        # for i, data in tqdm(enumerate(val_loader)):
        cnt = 0
        for data in tqdm(val_loader):
            cnt += 1
            decoded_words = trainer.evaluate_batch(model, data)
            # update val states for each batch.
            val_stats = compute_val_stat(data, decoded_words, global_entity_list, val_stats, args)
            logger.info(str((val_stats[0].n_words, val_stats[0].n_correct)))

    all_val_stats = Statistics.all_gather_stats_list(val_stats)
    logger.info("F1 SCORE:\t{}".format(str(all_val_stats[0].accuracy())))
    logger.info("\tCAL F1:\t{}".format(str(all_val_stats[1].accuracy())))
    logger.info("\tWET F1:\t{}".format(str(all_val_stats[2].accuracy())))
    logger.info("\tNAV F1:\t{}".format(str(all_val_stats[3].accuracy())))

    # bleu_score = all_val_stats[4].accuracy() / 100.0
    # not validated yet.
    bleu_score = 0.0
    logger.info("\tBleu Score:\t{}".format(str(bleu_score)))

    return bleu_score


def compute_val_stat(data_dev, words, global_entity_list, stats, args):
    w = 0
    temp_gen = []

    ref = []
    hyp = []
    ref_s = ""
    hyp_s = ""

    microF1_PRED, microF1_PRED_cal, microF1_PRED_nav, microF1_PRED_wet = 0, 0, 0, 0
    microF1_TRUE, microF1_TRUE_cal, microF1_TRUE_nav, microF1_TRUE_wet = 0, 0, 0, 0

    for i, row in enumerate(np.transpose(words)):
        st = ''
        for e in row:
            if e == '<EOS>':
                break
            else:
                st += e + ' '
        temp_gen.append(st)
        correct = data_dev['trg_plain'][i]
        ### compute F1 SCORE
        st = st.lstrip().rstrip()
        correct = correct.lstrip().rstrip()
        if args.dataset == 'kvr':
            # logger.info(str(data_dev['entity']))
            # logger.info(str(data_dev['kb_plain']))
            f1_true, count = Tree2Seq.compute_prf(data_dev['entity'][i], st.split(), global_entity_list,
                                              data_dev['kb_plain'][i])
            microF1_TRUE += f1_true
            microF1_PRED += count
            f1_true, count = Tree2Seq.compute_prf(data_dev['entity_cal'][i], st.split(), global_entity_list,
                                              data_dev['kb_plain'][i])
            microF1_TRUE_cal += f1_true
            microF1_PRED_cal += count
            f1_true, count = Tree2Seq.compute_prf(data_dev['entity_nav'][i], st.split(), global_entity_list,
                                              data_dev['kb_plain'][i])
            microF1_TRUE_nav += f1_true
            microF1_PRED_nav += count
            f1_true, count = Tree2Seq.compute_prf(data_dev['entity_wet'][i], st.split(), global_entity_list,
                                              data_dev['kb_plain'][i])
            microF1_TRUE_wet += f1_true
            microF1_PRED_wet += count

        w += wer(correct, st)
        ref.append(str(correct))
        hyp.append(str(st))
        ref_s += str(correct) + "\n"
        hyp_s += str(st) + "\n"

    with open('./tmp/ref_s.txt', 'a+') as f:
        f.write(ref_s)
    with open('./tmp/hyp_s.txt', 'a+') as f:
        f.write(hyp_s)

    # compute the bleu score
    # bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
    # bleu_stat = Statistics(n_correct=bleu_score, n_words=1)
    bleu_stat = Statistics()

    entity_stat = Statistics(n_correct=microF1_TRUE, n_words=microF1_PRED)
    entity_cal_stat = Statistics(n_correct=microF1_TRUE_cal, n_words=microF1_PRED_cal)
    entity_nav_stat = Statistics(n_correct=microF1_TRUE_nav, n_words=microF1_PRED_nav)
    entity_wet_stat = Statistics(n_correct=microF1_TRUE_wet, n_words=microF1_PRED_wet)
    new_stats = [entity_stat, entity_cal_stat, entity_nav_stat, entity_wet_stat, bleu_stat]

    for i in range(5):
        stats[i].update(new_stats[i])

    return stats

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

