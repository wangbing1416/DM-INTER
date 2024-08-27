import logging
import sys
import os
import json
import datetime
import torch
import random
import numpy as np

from models.bigru import Trainer as BiGRUTrainer
from models.bertTrainer import Trainer as BertTrainer
from models.bertlstm import Trainer as BertLSTMTrainer
from models.eannTrainer import Trainer as EANNTrainer
from models.mdfend import Trainer as MDFENDTrainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))  # logger output as print()

def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump


class Run():
    def __init__(self, args):
        self.args = args
    

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
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

        if 'lstm' in self.args.model_name:
            trainer = BertLSTMTrainer(self.args)
        elif self.args.model_name == 'mdfend':
            trainer = MDFENDTrainer(self.args)
        elif self.args.model_name == 'bigru':
            trainer = BiGRUTrainer(self.args)
        elif self.args.model_name == 'eann':
            trainer = EANNTrainer(self.args)
        else:
            trainer = BertTrainer(self.args)
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
