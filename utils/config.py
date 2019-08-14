import os
import logging 
import argparse
from tqdm import tqdm

UNK_token = 0
PAD_token = 1
EOS_token = 2
SOS_token = 3

TYPE_PAD_token = 0

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False
MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue bAbI')
parser.add_argument('-ds','--dataset', help='dataset, babi or kvr', required=False)
parser.add_argument('-t','--task', help='Task Number', required=False)
parser.add_argument('-dec','--decoder', help='decoder model', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size', type=int, required=False)
parser.add_argument('-bsz','--batch', help='Batch_size', type=int, required=False)
parser.add_argument('-exp','--experiment', help='Name of Experiment', type=str, required=True)
parser.add_argument('-lr','--learn', help='Learning Rate', type=float, required=False)
parser.add_argument('-dr','--drop', help='Drop Out', type=float, required=False)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', required=False, default=1)
parser.add_argument('-layer','--layer', help='Layer Number', type=int, required=False)
parser.add_argument('-lm','--limit', help='Word Limit', type=int, required=False,default=-10000)
parser.add_argument('-max-epoch','--max-epoch', help='Max Epoch', type=int, required=False,default=50)
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-test','--test', help='Testing mode', required=False)
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-useKB','--useKB', help='Put KB in the input or not', required=False, default=1)
parser.add_argument('-ep','--entPtr', help='Restrict Ptr only point to entity', required=False, default=0)
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=2)
parser.add_argument('-an','--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# define mode
parser.add_argument("--mode", type=str, default='train', choices=['train', 'eval', 'test'])
parser.add_argument('-resume','--resume', help='The path to resume', type=str, required=False, default='')

parser.add_argument('-debug', '--debug', help='Debug Mode.', action='store_true')

# if specified this, use "tac" mode.
parser.add_argument('-tac', '--traverse-all-combination', help='Data augmentation, will traverse all combinations for each kb.', dest='traverse-all-combination', action='store_true')
parser.add_argument('-dist', '--distributed', help='Distributed training.', dest='distributed', action='store_true')

# ensemble parameter
parser.add_argument('-add-relu', '--add-relu', help='add-relu in ensemble.', action='store_true')
parser.add_argument('-add-norm', '--add-norm', help='add-norm in ensemble.', action='store_true')
parser.add_argument('-add-dropout', '--add-dropout', help='add-dropout in ensemble.', action='store_true')
parser.add_argument('-add-skip-con', '--add-skip-con', help='add-skip-con in ensemble.', action='store_true')

parser.add_argument('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int, help="list of ranks of each process.")
parser.add_argument('--world_size', '-world_size', default=1, type=int,
          help="total number of distributed processes.")
parser.add_argument('--print_freq', '-print_freq', default=1, type=int,
          help="The frequency of printing output infos.")

# global ranking parameter
parser.add_argument('-no-kb-embed', '--no-kb-embed', help='Add embedding for no kb roots.', action='store_true')

args = vars(parser.parse_args())
print(args)

name = str(args['task'])+str(args['decoder'])+str(args['hidden'])+str(args['batch'])+str(args['learn'])+str(args['drop'])+str(args['layer'])+str(args['limit'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))

LIMIT = int(args["limit"])
USEKB = int(args["useKB"])
ENTPTR = int(args["entPtr"])
ADDNAME = args["addName"]

SRC_WEIGHTS = {
    'general_no_context.txt': 0.01,
    'general_with_context.txt': 0,
    'medical.txt': 0,
}

MEM_TOKEN_SIZE = 5

