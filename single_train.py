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
# from utils.general_utils import to_device

import random

random.seed(1234)
torch.manual_seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main_worker(args, gpu):

    model, train, dev, test = build_model(args, gpu)

    # Data loading code
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test)

    else:
        train_sampler = None
        dev_sampler = None
        test_sampler = None

    # multiple workers are not allowed in multiprocessing !
    train_loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch, shuffle=(train_sampler is None),
            pin_memory=False, sampler=train_sampler, collate_fn=utils_tree.collate_fn_new)


    val_loader = torch.utils.data.DataLoader(dev,
            batch_size=args.batch, shuffle=False, sampler=dev_sampler,
            pin_memory=False, collate_fn=utils_tree.collate_fn_new)

    test_loader = torch.utils.data.DataLoader(test,
            batch_size=args.batch, shuffle=False,sampler=test_sampler,
            pin_memory=False, collate_fn=utils_tree.collate_fn_new)

    best_bleu = 0.0
    best_f1 = 0.0
    trainer = Tree2SeqTrainer(model, lr=float(args.learn), args=args)
    scheduler = lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='max', factor=0.8, patience=5,
                                               min_lr=0.0001, verbose=True)
    logger.info("Built trainer and scheduler.")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            # trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # check for mode
    if args.mode == 'eval':
        logger.info('In eval mode.')
        validate_one_epoch(val_loader, model, trainer, args)
        return

    if args.mode == 'test':
        logger.info('In test mode.')
        # logger.info('Eval.')
        # validate_one_epoch(val_loader, model, trainer, args)
        logger.info('Test.')
        validate_one_epoch(test_loader, model, trainer, args)
        return

    best_f1s = []

    logger.info('In the training process now.')
    for epoch in range(0, args.max_epoch):
        logger.info("In epoch {}".format(str(epoch)))
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_one_epoch(train_loader, model, trainer, epoch, args)
        logger.info("Into train one epoch.")

        # evaluate on validation set
        check_result = True if epoch >= 25 else False
        bleu, f1s = validate_one_epoch(val_loader, model, trainer, args, check_result=check_result)

        # remember best acc@1 and save checkpoint
        is_best = f1s[0] > best_f1
        best_bleu = max(bleu, best_bleu)
        best_f1 = max(f1s[0], best_f1)
        if is_best:
            best_f1s = f1s
            best_bleu = bleu

        scheduler.step(f1s[0])

        if not args.distributed or (args.distributed and args.rank % args.world_size == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.decoder,
                'state_dict': model.state_dict(),
                'best_bleu': best_bleu,
                # 'optimizer': optimizer.state_dict(),
            }, is_best, args.experiment)


    logger.info("BEST F1 SCORE:\t{}".format(str(best_f1s[0])))
    logger.info("\tBEST CAL F1:\t{}".format(str(best_f1s[1])))
    logger.info("\tBEST WET F1:\t{}".format(str(best_f1s[2])))
    logger.info("\tBEST NAV F1:\t{}".format(str(best_f1s[3])))
    logger.info("\tBEST BLEU:\t{}".format(str(best_bleu)))

    logger.info('Test')
    validate_one_epoch(test_loader, model, trainer, args)
    return

def build_model(args, gpu):
    # global best_acc1
    if gpu == -1:
        args.gpu = 0
    else:
        args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = gpu

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
           # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch = int( args.batch / args.world_size)
            args.workers = int( args.workers / args.world_size)

    # read in the dataset
    # todo : clean this logic
    prepare_data_seq = utils_tree.prepare_data_seq
    # print(args.batch)
    logger.info('Batch-size per gpu: {}'.format(args.batch))
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(vars(args),batch_size=int(args.batch),shuffle=True)
    # create model
    logger.info(args.decoder)
    logger.info(str(globals()[args.decoder]))

    model = globals()[args.decoder](int(args.hidden),
                                        max_len,max_r,lang,args.path,args.task,
                                        lr=float(args.learn),
                                        n_layers=int(args.layer),
                                        dropout=float(args.drop),
                                        unk_mask=bool(int(args.unk_mask)),
                                        args=args
                                        )

    logger.info("Built model")
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            logger.info("Built model distributed.")
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            logger.info("Built model distributed.")
    else:
        model.cuda()

    return model, train, dev, test


def train_one_epoch(train_loader, model, trainer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    Losses = AverageMeter('Loss', ':6.2f')
    V_Loss = AverageMeter('VL', ':6.2f')
    P_Loss = AverageMeter('PL', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, Losses, V_Loss,
                             P_Loss, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # logger.info("Into train batch.")
        # measure data loading time
        data_time.update(time.time() - end)

        # data = to_device(data, torch.device('cuda'))
        # compute output
        loss = trainer.train_batch(model, data, len(data['src_seqs']), 1.0, 0.5, i == 0, i, accumulate_step=args.accumulate_step)

        # for debug
        # for name, param in model.named_parameters():
        #     print(name, param, True if param.grad is not None else False)
        # pdb.set_trace()

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


def validate_one_epoch(val_loader, model, trainer, args, check_result=False):
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
            # data = to_device()
            # end = time.time()
            cnt += 1
            decoded_words = trainer.evaluate_batch(model, data, check_result=check_result)
            # logger.info("Decode Time cost: {}".format(str(time.time() - end)))
            # end = time.time()
            # update val states for each batch.
            val_stats = compute_val_stat(data, decoded_words, global_entity_list, val_stats, args)
            # logger.info("Val Compute Time cost: {}".format(str(time.time() - end)))

    if args.distributed:
        all_val_stats = Statistics.all_gather_stats_list(val_stats)
    else:
        all_val_stats = val_stats
    f1 = all_val_stats[0].accuracy()
    cal_f1 = all_val_stats[1].accuracy()
    wet_f1 = all_val_stats[2].accuracy()
    nav_f1 = all_val_stats[3].accuracy()

    logger.info("F1 SCORE:\t{}".format(str(f1)))
    logger.info("\tCAL F1:\t{}".format(str(cal_f1)))
    logger.info("\tWET F1:\t{}".format(str(wet_f1)))
    logger.info("\tNAV F1:\t{}".format(str(nav_f1)))

    bleu_score = all_val_stats[4].accuracy() / 100.0
    # not validated yet.
    # bleu_score = 0.0
    logger.info("\tBleu Score:\t{}".format(str(bleu_score)))

    return bleu_score, [f1, cal_f1, wet_f1, nav_f1]


def compute_val_stat(data_dev, words, global_entity_list, stats, args):
    w = 0
    temp_gen = []

    ref = []
    hyp = []
    src = []
    ref_s = ""
    hyp_s = ""
    src_s = ""

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


        conv_src = [item[0] for item in data_dev['src_plain'][i] if '$' in item[1]]
        conv_src_s = " ".join(conv_src)
        src_s += conv_src_s + '\n'
        src.append(src_s)

        # w += wer(correct, st)
        ref.append(str(correct))
        hyp.append(str(st))

        ref_s += str(correct) + "\n"
        hyp_s += str(st) + "\n"

    with open('./tmp/{}_ref_s.txt'.format(args.experiment), 'a+') as f:
        f.write(ref_s)
    with open('./tmp/{}_hyp_s.txt'.format(args.experiment), 'a+') as f:
        f.write(hyp_s)
    with open('./tmp/{}_src_s.txt'.format(args.experiment), 'a+') as f:
        f.write(src_s)


    # compute the bleu score
    bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)

    bleu_stat = Statistics(n_correct=bleu_score, n_words=1)

    entity_stat = Statistics(n_correct=microF1_TRUE, n_words=microF1_PRED)
    entity_cal_stat = Statistics(n_correct=microF1_TRUE_cal, n_words=microF1_PRED_cal)
    entity_nav_stat = Statistics(n_correct=microF1_TRUE_nav, n_words=microF1_PRED_nav)
    entity_wet_stat = Statistics(n_correct=microF1_TRUE_wet, n_words=microF1_PRED_wet)
    new_stats = [entity_stat, entity_cal_stat, entity_nav_stat, entity_wet_stat, bleu_stat]

    for i in range(5):
        stats[i].update(new_stats[i])

    return stats

def save_checkpoint(state, is_best, experiment='Tree2Seq'):
    filename = './model/{}.pt'.format(experiment)
    torch.save(state, filename)
    if is_best:
        best_filename = './model/{}_best.pt'.format(experiment)
        shutil.copyfile(filename, best_filename)


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

