import os
import torch
import tqdm
import datetime
import copy
from .layers import *
import numpy as np
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, data2gpu_noemo, Averager, metrics, Recorder
from utils.dataloader import get_dataloader, get_dataloader_noemo, word2input
from model import BERTModel, BERTEmoModel, EANNModel, MDFENDModel, T5MDModel, T5EMOMDModel, T5MDFENDMDModel, T5OURModel, T5OUREMOModel, Ranker
from transformers import BertTokenizer, AutoTokenizer


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
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrain_name)
        for k in self.reasoning_query.keys():
            self.reasoning_query[k] = torch.tensor(self.tokenizer.encode(self.reasoning_query[k])).cuda()
        for k in self.answer.keys():
            self.answer[k] = torch.tensor(self.tokenizer.encode(self.answer[k])[:-1]).cuda()

    def train(self, logger=None):
        nowtime = datetime.datetime.now().strftime("%m%d-%H%M")
        save_path = os.path.join(self.save_path, 'parameter_bert-' + self.args.dataset + '.pkl')
        if (logger):
            logger.info('start training......')
        if self.args.model_name == 't5':
            self.detector = T5MDModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.args.attention_head)
        elif self.args.model_name == 't5emo':
            self.detector = T5EMOMDModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.args.attention_head)
        elif self.args.model_name == 't5mdfend':
            self.detector = T5MDFENDMDModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.args.attention_head)
        elif self.args.model_name == 't5our':
            self.detector = T5OURModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.reasoning_query, self.answer)
        elif self.args.model_name == 't5ouremo':
            self.detector = T5OUREMOModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.reasoning_query, self.answer)
        # elif self.args.model_name == 't5ourmdfend':
        #     self.detector = T5OURMDFENDModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.reasoning_query, self.answer)
        else:
            self.detector = T5MDModel(self.args.pretrain_name, self.args.emb_dim, self.args.inner_dim, self.args.dropout, self.args.attention_head)

        recorder = Recorder(self.args.early_stop)

        if self.args.dataset == 'gossip':
            dataloader = get_dataloader
            gpuload = data2gpu
        else:
            dataloader = get_dataloader_noemo
            gpuload = data2gpu_noemo

        train_loader = dataloader(self.args.data_path + 'train' + self.args.dataset_id + '.json',
                                  self.args.data_path + 'train_emo.npy',
                                  self.args.max_len, self.args.batch_size, shuffle=True, aug_prob=self.args.aug_prob,
                                  pretrain_name=self.args.pretrain_name)
        val_loader = dataloader(self.args.data_path + 'val' + self.args.dataset_id + '.json',
                                self.args.data_path + 'val_emo.npy',
                                self.args.max_len, self.args.batch_size, shuffle=False, aug_prob=self.args.aug_prob,
                                pretrain_name=self.args.pretrain_name)
        test_loader = dataloader(self.args.data_path + 'test' + self.args.dataset_id + '.json',
                                 self.args.data_path + 'test_emo.npy',
                                 self.args.max_len, self.args.batch_size, shuffle=False, aug_prob=self.args.aug_prob,
                                 pretrain_name=self.args.pretrain_name)

        self.detector = self.detector.cuda()
        loss_fn = torch.nn.BCELoss()

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

            for step_n, batch in enumerate(train_data_iter):
                batch_data = gpuload(batch, use_cuda=True)
                label = batch_data['label']

                pred, _, loss_t5, _ = self.detector(**batch_data)
                loss = loss_fn(pred, label.float()) + loss_t5 * self.args.beta

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())
            logger.info('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(val_loader, flag='val')
            mark = recorder.add(results)  # early stop with validation metrics
            if mark == 'save':
                torch.save(self.detector.state_dict(), save_path)
            elif mark == 'esc':
                break
            else:
                continue

        logger.info("Stage: testing...")
        self.detector.load_state_dict(torch.load(save_path))

        future_results = self.test(test_loader, flag='test')

        logger.info("test score: {}.".format(future_results))
        logger.info("lr: {}, aug_prob: {}, avg test score: {}.\n\n".format(self.args.lr, self.args.aug_prob,
                                                                           future_results['metric']))
        print('test results:', future_results)

        return future_results, save_path

    def test(self, dataloader, flag):
        pred = []
        label = []
        sequence = []
        origin_text = []
        self.detector.eval()
        data_iter = tqdm.tqdm(dataloader)

        if self.args.dataset == 'gossip' or self.args.dataset == 'weibo':
            gpuload = data2gpu
        else:
            gpuload = data2gpu_noemo

        with torch.no_grad():
            for step_n, batch in enumerate(data_iter):
                batch_data = gpuload(batch, use_cuda=True)
                batch_label = batch_data['label']

                batch_pred, batch_feature, _, pred_sequence = self.detector(**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_pred.detach().cpu().numpy().tolist())
                if flag == 'test':
                    for i, s in enumerate(pred_sequence):
                        sequence.append(self.tokenizer.decode(s, skip_special_tokens=True))
                        origin_text.append(self.tokenizer.decode(batch[0][i], skip_special_tokens=True))
                    import pandas as pd
                    temp = [[origin_text[i], sequence[i], label[i]] for i in range(len(sequence))]
                    data = pd.DataFrame(temp)
                    writer = pd.ExcelWriter('case.xlsx')  # 写入Excel文件
                    data.to_excel(writer, 'page_0', float_format='%.5f')
                    writer.save()
                    writer.close()
        return metrics(label, pred)