import torch
import random
import tqdm
import pandas as pd
import json
import numpy as np
import nltk
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader

label_dict = {
    "real": 0,
    "fake": 1
}

category_dict = {
    "2000": 0, "2001": 0, "2002": 0, "2003": 0, "2005": 0, "2004": 0,
    "2006": 0, "2007": 0, "2008": 0, "2009": 0, "2010": 0, "2011": 0,
    "2012": 0, "2013": 0, "2014": 0, "2015": 0, "2016": 0,
    "2017": 1, "2018": 2
}

def word2input(texts, max_len, tokenizer, path):
    token_ids = []
    print("\nData Processing: tokenizing text from {}".format(path))
    for text in tqdm.tqdm(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length', truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape[0], token_ids.shape[1])
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i, :] = (tokens != mask_token_id)
    return token_ids, masks


def get_dataloader(path, emo_path, max_len, batch_size, shuffle, aug_prob, pretrain_name):
    data_list = json.load(open(path, 'r', encoding='utf-8'))
    df_data = pd.DataFrame(columns=('content', 'label'))
    print("\nData Processing: loading data from {}".format(path))
    for item in tqdm.tqdm(data_list):
        tmp_data = {}
        tmp_data['content'] = item['content']
        if 'chinese' in pretrain_name: tmp_data['label'] = label_dict[item['label']]
        else: tmp_data['label'] = item['label']  # real-0, fake-1
        tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
        df_data = df_data.append(tmp_data, ignore_index=True)
    emotion = np.load(emo_path).astype('float32')
    emotion = torch.tensor(emotion)
    content = df_data['content'].to_numpy()
    label = torch.tensor(df_data['label'].astype(int).to_numpy())
    year = torch.tensor(df_data['year'].apply(lambda c: category_dict[c]).astype(int).to_numpy())
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    content_token_ids, content_masks = word2input(content, max_len, tokenizer, path)

    dataset = TensorDataset(content_token_ids, content_masks, emotion, label, year)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader


def get_dataloader_noemo(path, emo_path, max_len, batch_size, shuffle, aug_prob, pretrain_name):
    data_list = json.load(open(path, 'r', encoding='utf-8'))
    df_data = pd.DataFrame(columns=('content', 'label'))
    print("\nData Processing: loading data from {}".format(path))
    for item in tqdm.tqdm(data_list):
        tmp_data = {}
        tmp_data['content'] = item['content']
        if 'chinese' in pretrain_name: tmp_data['label'] = label_dict[item['label']]
        else: tmp_data['label'] = item['label']  # real-0, fake-1
        tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
        df_data = df_data.append(tmp_data, ignore_index=True)
    content = df_data['content'].to_numpy()
    label = torch.tensor(df_data['label'].astype(int).to_numpy())
    tokenizer = AutoTokenizer.from_pretrained(pretrain_name)
    content_token_ids, content_masks = word2input(content, max_len, tokenizer, path)

    dataset = TensorDataset(content_token_ids, content_masks, label)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=shuffle
    )
    return dataloader


class DataProcess():
    def __init__(self, path, max_len, aug_prob, para_len, flag):
        self.data_list = json.load(open(path, 'r', encoding='utf-8'))
        # if flag == 'train':
        self.data_list.sort(key=lambda x: x['time'])  # sorted by time
        self.data_list = self.data_list
        df_data = pd.DataFrame(columns=('content', 'label'))
        for item in self.data_list:
            tmp_data = {}
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['year'] = item['time'].split(' ')[0].split('-')[0]
            df_data = df_data.append(tmp_data, ignore_index=True)
        self.emotion = np.load(path.replace('.json', '_emo.npy')).astype('float32')
        self.emotion = torch.tensor(self.emotion)
        content = df_data['content'].to_numpy()
        entity_content = df_data['entity'].to_numpy()
        self.label = torch.tensor(df_data['label'].astype(int).to_numpy())
        self.year = torch.tensor(df_data['year'].apply(lambda c: category_dict[c]).astype(int).to_numpy())
        self.content_token_ids, self.content_masks = word2input(content, max_len, para_len)
        self.entity_token_ids, self.entity_masks = word2input(entity_content, 50, para_len)

    def get_dataloader(self, batch_size, shuffle, segment_type, period=0):  # period=0 indicates the overall dataset
        if segment_type == 'year':
            if period > 10:
                raise ValueError('period must not bigger than 10! (2009 - 2018)')
            index = [17, 40, 83, 118, 168, 238, 326, 474, 784, 5537, 7061]  # start indices in the training set from 2009, by year
        else:
            if period > 37:
                raise ValueError('period must not bigger than 37! (2009 - 2018)')
            index = [17, 21, 24, 27, 37, 54, 57, 65, 76, 84, 89, 100, 107, 120, 128, 138, 153, 166, 184, 206, 219, 241, 261, 276, 303, 342, 372, 409, 447, 511, 584, 672, 753, 1053, 2397, 3823, 5502, 7025]
        if period == 0:
            dataset = TensorDataset(self.content_token_ids, self.content_masks, self.entity_token_ids, self.entity_masks,
                                    self.label, self.year, self.emotion)
        else:
            dataset = TensorDataset(self.content_token_ids[index[period-1]: index[period]],
                                    self.content_masks[index[period-1]: index[period]],
                                    self.entity_token_ids[index[period-1]: index[period]],
                                    self.entity_masks[index[period-1]: index[period]],
                                    self.label[index[period-1]: index[period]],
                                    self.year[index[period-1]: index[period]],
                                    self.emotion[index[period-1]: index[period]]
                                    )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=shuffle
        )
        return dataloader