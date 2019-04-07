import numpy as np
import logging 
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
# from models.Mem2Seq import *
from models.Tree2Seq import *
import utils.utils_kvr_tree as utils_tree
# from models.Mem2Seq_Chitchat import

import torch.multiprocessing as mp
import torch.utils.data as data

def main(args, rank):
    args = vars(args)

    BLEU = False

    # only tree2seq now.
    if args['dataset'] == 'kvr' and args['decoder'] == 'Tree2Seq':
        prepare_data_seq = utils_tree.prepare_data_seq
    else:
        raise NotImplementedError()

    # Configure models
    avg_best,cnt,acc = 0.0,0,0.0
    cnt_1 = 0
    ### LOAD DATA
    train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args,batch_size=int(args['batch']),shuffle=True)

    model = globals()[args['decoder']](int(args['hidden']),
                                        max_len,max_r,lang,args['path'],args['task'],
                                        lr=float(args['learn']),
                                        n_layers=int(args['layer']),
                                        dropout=float(args['drop']),
                                        unk_mask=bool(int(args['unk_mask']))
                                        )

    # parallel
    gpu_count = torch.cuda.device_count()
    model.device_ids = list(range(gpu_count))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train)
    # init loaders
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=args.batch_size, collate_fn=utils_tree.collate_fn_new,
        num_workers=args['workers'], pin_memory=True, sampler=train_sampler)

    trainer = Tree2SeqTrainer(model, lr=float(args['learn']), num_parallel_calls=gpu_count)

    for epoch in range(300):
        logging.info("Epoch:{}".format(epoch))
        # Run the train function
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        for i, data in pbar:
            if args['decoder'] == 'Tree2Seq':
                trainer.train_batch(model, data, len(data['src_seqs']), 10.0, 0.5, i==0)
            pbar.set_description(trainer.print_loss())
        if((epoch+1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev,avg_best, BLEU)
            # todo : add Tree2Seq
            if 'Mem2Seq' in args['decoder'] or 'Tree2Seq' in args['decoder']:
                model.scheduler.step(acc)
            # early stopping
            if(acc >= avg_best):
                avg_best = acc
                cnt=0
            else:
                cnt+=1
            if(cnt == 5): break
            if(acc == 1.0): break

    return






