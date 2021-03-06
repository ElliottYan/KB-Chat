import numpy as np
import logging 
from tqdm import tqdm

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *
from models.Tree2Seq import *
from models.Mem2Seq_Chitchat import *

BLEU = False

# variation of Mem2Seq
if (args['decoder'].startswith("Mem2Seq")):
    if args['dataset']=='kvr':
        from utils.utils_kvr_mem2seq import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi_mem2seq import *
    elif args['dataset'] == 'chitchat':
        from utils.utils_chitchat import *
    else: 
        print("You need to provide the --dataset information")
elif (args['decoder'] == 'Tree2Seq'):
    if args['dataset']=='kvr':
        from utils.utils_kvr_tree import *
        BLEU = True
    elif args['dataset']=='babi':
        # todo: utils for babi tree is not defined yet.
        from utils.utils_babi_mem2seq import *
    else:
        print("You need to provide the --dataset information")
else:
    if args['dataset']=='kvr':
        from utils.utils_kvr import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi import *
    else:
        print("You need to provide the --dataset information")

# Configure models
avg_best,cnt,acc = 0.0,0,0.0
cnt_1 = 0   
### LOAD DATA
train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(args['task'],batch_size=int(args['batch']),shuffle=True)

# ugly...
if args['decoder'].startswith("Mem2Seq") or args['decoder'] == "Tree2Seq":
    model = globals()[args['decoder']](int(args['hidden']),
                                        max_len,max_r,lang,args['path'],args['task'],
                                        lr=float(args['learn']),
                                        n_layers=int(args['layer']), 
                                        dropout=float(args['drop']),
                                        unk_mask=bool(int(args['unk_mask']))
                                    )
else:
    model = globals()[args['decoder']](int(args['hidden']),
                                    max_len,max_r,lang,args['path'],args['task'],
                                    lr=float(args['learn']),
                                    n_layers=int(args['layer']), 
                                    dropout=float(args['drop'])
                                )

# parallel
'''
gpu_count = torch.cuda.device_count()
model.device_ids = list(range(gpu_count))
model = nn.DataParallel(model, device_ids=model.device_ids)
optimizer = optim.Adam(model.parameters(), lr=float(args['learn']))
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1,
                                                        min_lr=0.0001, verbose=True)
'''

for epoch in range(300):
    logging.info("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        if args['decoder'] == 'Tree2Seq':
            model.train_batch(data, len(data['src_seqs']), 10.0, 0.5, i==0)
        else:
            model.train_batch(data[0], data[1], data[2], data[3],data[4],data[5],
                        len(data[1]),10.0,0.5,i==0)
        pbar.set_description(model.print_loss())
    if((epoch+1) % int(args['evalp']) == 0):
        acc = model.evaluate(dev,avg_best, BLEU)
        # todo : add Tree2Seq
        if 'Mem2Seq' in args['decoder']:
            model.scheduler.step(acc)
        # early stopping
        if(acc >= avg_best):
            avg_best = acc
            cnt=0
        else:
            cnt+=1
        if(cnt == 5): break
        if(acc == 1.0): break 


