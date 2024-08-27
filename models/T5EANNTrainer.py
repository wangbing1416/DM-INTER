import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel, AutoTokenizer
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader, word2input
from utils.dataloader import DataProcess
from model import EANNModel, T5EANNMDModel, T5OUREANNModel


class Trainer():
    def __init__(self, config):
        self.args = config

        self.save_path = os.path.join(self.args.save_param_dir, self.args.model_name)
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            self.save_param_dir = os.makedirs(self.save_path)
        self.reasoning_query = {
            'public': "Is this article aimed at the public?",
            'emotion': "Is there any emotional expression in this article?",
            'individual': "Does this article express any personal points?",
            'popularize': "Is this an article aimed at popularization?",
            'clout': "Is this an article aimed at pursuing attention?",
            'conflict': "Is this article attempting to create conflict?",
            'smear': "Is this article smearing others?",
            'bias': "Is there any bias in this article?",
            'connect': "Is this article just seeking interaction and connection?"
        }
        self.answer = {
            'yes': "yes",
            'no': "no"
        }
        tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_name)
        for k in self.reasoning_query.keys():
            self.reasoning_query[k] = torch.tensor(tokenizer.encode(self.reasoning_query[k])).cuda()
        for k in self.answer.keys():
            self.answer[k] = torch.tensor(tokenizer.encode(self.answer[k])[:-1]).cuda()

    def train(self, logger=None):
        save_path = os.path.join(self.save_path, 'parameter_bert-' + self.args.dataset + '.pkl')
        if (logger):
            logger.info('start training......')
        if 't5our' in self.args.model_name:
            self.detector = T5OUREANNModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.reasoning_query, self.answer)
        else:
            self.detector = T5EANNMDModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout)

        loss_fn = torch.nn.BCELoss()
        recorder = Recorder(self.args.early_stop)

        train_loader = get_dataloader(self.args.data_path + 'train' + self.args.dataset_id + '.json',
                                      self.args.data_path + 'train_emo.npy',
                                      self.args.max_len, self.args.batch_size, shuffle=True,
                                      aug_prob=self.args.aug_prob, pretrain_name=self.args.pretrain_name)
        val_loader = get_dataloader(self.args.data_path + 'val' + self.args.dataset_id + '.json',
                                    self.args.data_path + 'val_emo.npy',
                                    self.args.max_len, self.args.batch_size, shuffle=False, aug_prob=self.args.aug_prob,
                                    pretrain_name=self.args.pretrain_name)
        test_loader = get_dataloader(self.args.data_path + 'test' + self.args.dataset_id + '.json',
                                     self.args.data_path + 'test_emo.npy',
                                     self.args.max_len, self.args.batch_size, shuffle=False,
                                     aug_prob=self.args.aug_prob, pretrain_name=self.args.pretrain_name)
        self.detector = self.detector.cuda()

        diff_part = ["bertModel.embeddings", "bertModel.encoder"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.detector.named_parameters() if any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.args.lr
            },
            {
                "params": [p for n, p in self.detector.named_parameters() if not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": self.args.mlp_lr
            },
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, eps=self.args.adam_epsilon)

        logger.info("Training the fake news detector based on {}".format(self.args.pretrain_name))
        for epoch in range(self.args.epoch):
            self.detector.train()

            train_data_iter = tqdm.tqdm(train_loader)
            avg_loss = Averager()
            alpha = max(2. / (1. + np.exp(-10 * epoch / self.args.epoch)) - 1, 1e-1)

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, use_cuda=True)
                label = batch_data['label']
                domain_label = batch_data['year']

                pred, domain_pred, loss_t5 = self.detector(**batch_data, alpha=alpha)
                loss = loss_fn(pred, label.float())
                loss_adv = F.nll_loss(F.log_softmax(domain_pred, dim=1), domain_label)
                loss = loss + loss_adv + loss_t5 * self.args.beta
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            logger.info('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader)
            mark = recorder.add(results)
            if mark == 'save':
                torch.save(self.detector.state_dict(), save_path)
            elif mark == 'esc':
                break
            else:
                continue

        logger.info("Stage: testing...")
        self.detector.load_state_dict(torch.load(save_path))

        future_results = self.test(test_loader)
        logger.info("test score: {}.".format(future_results))
        logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.args.lr, self.args.aug_prob,
                                                                           future_results['metric']))
        print('test results:', future_results)
        return future_results, save_path

    def test(self, dataloader):
        pred = []
        label = []
        self.detector.eval()
        data_iter = tqdm.tqdm(dataloader)
        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                batch_data = data2gpu(batch, use_cuda=True)
                batch_label = batch_data['label']

                batch_pred, domain_pred, _ = self.detector(**batch_data, alpha=-1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())

        return metrics(label, pred)