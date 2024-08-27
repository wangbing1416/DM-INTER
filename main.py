import os
import sys
import json
import argparse
import logging
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='t5our', help='t5, t5emo, t5mdfend, t5our')
parser.add_argument('--dataset', default='gossip')
parser.add_argument('--dataset_id', default='')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--segment_type', default='year')
parser.add_argument('--aug_prob', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=150)

parser.add_argument('--early_stop', type=int, default=10)
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', default='0')
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--attention_head', type=int, default=2)
parser.add_argument('--inner_dim', type=int, default=384)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=7e-5)
parser.add_argument('--mlp_lr', type=float, default=7e-5)
parser.add_argument('--beta', type=float, default=0.0)

# optimizer
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

parser.add_argument('--emb_type', default='bert')
parser.add_argument('--save_log_dir', default='./logs')
parser.add_argument('--save_param_dir', default='./param_model')
parser.add_argument('--param_log_dir', default='./logs/param')

data_path = {
    'gossip': 'gossip/',
    'politifact': 'politifact/',
    'snopes': 'snopes/'
}

pretrain_model = {'bert': {'gossip': 'bert-base-uncased',
                           'politifact': 'bert-base-uncased',
                           'snopes': 'bert-base-uncased'},
                  't5': {'gossip': 'flan-t5-base',
                         'politifact': 'flan-t5-base',
                         'snopes': 'flan-t5-base'}, }

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args.data_path = os.path.join(args.root_path, data_path[args.dataset])
if 't5' in args.model_name:
    args.pretrain_name = pretrain_model['t5'][args.dataset]
else:
    args.pretrain_name = pretrain_model['bert'][args.dataset]

import torch
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

from models.bigru import Trainer as BiGRUTrainer
from models.bertTrainer import Trainer as BertTrainer
from models.T5Trainer import Trainer as T5Trainer
from models.T5EANNTrainer import Trainer as T5EANNTrainer
from models.bertlstm import Trainer as BertLSTMTrainer
from models.eannTrainer import Trainer as EANNTrainer
from models.mdfend import Trainer as MDFENDTrainer


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))  # logger output as print()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Run():
    def __init__(self, args):
        self.args = args

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def config2dict(self):
        config_dict = {}
        for k, v in self.args.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        param_log_dir = self.args.param_log_dir
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        nowtime = datetime.datetime.now().strftime("%m%d-%H%M")
        param_log_file = os.path.join(param_log_dir, self.args.model_name + '_' + 'param' + nowtime + '.txt')
        logger.addHandler(logging.FileHandler(param_log_file))
        logger.info('> training arguments:')
        for arg in self.args._get_kwargs():
            logger.info('>>> {0}: {1}'.format(arg[0], arg[1]))

        json_path = './logs/json/' + self.args.model_name + str(self.args.aug_prob) + '-' + nowtime + '.json'
        json_result = []

        best_metric = {}
        best_metric['metric'] = 0

        best_model_path = None
        if self.args.model_name == 't5oureann':
            trainer = T5EANNTrainer(self.args)
        elif 't5' in self.args.model_name:
            trainer = T5Trainer(self.args)
        elif self.args.model_name == 't5eann':
            trainer = T5EANNTrainer(self.args)
        else:
            trainer = T5Trainer(self.args)
        metrics, model_path = trainer.train(logger)
        json_result.append(metrics)
        if metrics['metric'] > best_metric['metric']:
            best_metric['metric'] = metrics['metric']
            best_model_path = model_path

        print("best model path:", best_model_path)
        print("best metric:", best_metric)
        logger.info("best model path:" + best_model_path)
        logger.info("best metric:" + str(best_metric))
        logger.info('--------------------------------------\n')

        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    Run(args=args).main()
