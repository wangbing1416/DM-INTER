import os
import torch
import tqdm
import datetime
import copy
# from .layers import *
from sklearn.metrics import *
from transformers import BertModel, T5ForConditionalGeneration
# from opendelta import PrefixModel, LoraModel, SoftPromptModel
from utils.utils import data2gpu, Averager, metrics, Recorder
# from utils.dataloader import get_dataloader
from models.layers import *
# from .huggingface.modeling_bert import BertModel

class T5OUREANNModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, query, answer):
        super(T5OUREANNModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_name).requires_grad_(False)
        self.query = query
        self.answer = answer
        self.parent = {'popularize': ['public'], 'clout': ['public', 'emotion'], 'conflict': ['emotion', 'individual'],
                       'smear': ['individual'], 'bias': ['individual'], 'connect': ['individual']}

        for name, param in self.model.named_parameters():
            if name.startswith("encoder.block.11.layer"):
                param.requires_grad = True
            if name.startswith("decoder.block.11.layer"):
                param.requires_grad = True
        # self.cross_attention = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attention_head, batch_first=True)
        # self.fusion_mlp = torch.nn.Linear(emb_dim, emb_dim)
        domain_num = 3
        self.domain_classifier = nn.Sequential(MLP(emb_dim * 2, [mlp_dims], dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims, domain_num))
        self.mlp = MLP(emb_dim * 2, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        alpha = kwargs['alpha']
        answer_list = [self.answer['yes'], self.answer['no']]
        with torch.no_grad():
            answers = {}
            batch_size = content.shape[0]
            decoder_ids_seq = torch.tensor([0 for _ in range(batch_size)]).unsqueeze(dim=1).cuda()
            encoder_output = self.model.encoder(input_ids=content)
            for q1 in ['public', 'emotion', 'individual']:
                # generate answers for three second-layer intents
                decoder_ids_seq = torch.cat([decoder_ids_seq, self.query[q1].repeat(batch_size, 1)], dim=1)
                output = self.model(input_ids=None, encoder_outputs=encoder_output, decoder_input_ids=decoder_ids_seq, output_hidden_states=False)
                index = output.logits[:, -1, answer_list].max(dim=1)[1]
                result = [answer_list[i] for i in index]
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.tensor(result).unsqueeze(dim=1).cuda()], dim=1)
                answers[q1] = index
            # generate answers for leaf intents
            for q2 in ['popularize', 'clout', 'conflict', 'smear', 'bias', 'connect']:
                temp = []
                flag = [0 for _ in range(batch_size)]
                for i in range(batch_size):  # construct questions
                    for pare in self.parent[q2]:
                        if answers[pare][i] == 1: flag[i] = 1  # if its parents exist 1, this question should be queried
                    if flag[i] == 1: temp.append(self.query[q2])
                    else: temp.append(torch.zeros(self.query[q2].shape[0], dtype=torch.int64).cuda())  # padding with 0 (T5 pad)
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.stack(temp)], dim=1)
                output = self.model(input_ids=None, encoder_outputs=encoder_output, decoder_input_ids=decoder_ids_seq, output_hidden_states=False)
                index = output.logits[:, -1, answer_list].max(dim=1)[1]
                result = [answer_list[i] for i in index]
                temp = []
                for i in range(batch_size):  # construct answers
                    if flag[i] == 1: temp.append(result[i])
                    else: temp.append(torch.zeros(1, dtype=torch.int64).cuda())  # padding
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.stack(temp)], dim=1)
            labels = torch.cat([decoder_ids_seq, torch.ones([batch_size, 1], dtype=torch.int64).cuda()], dim=1)[:, 1:].contiguous()  # add </s>

        output = self.model(input_ids=content, labels=labels, output_hidden_states=True)
        # attention fusion
        # decoder_last_token_embedding = output.decoder_hidden_states[-1][:, -1, :].unsqueeze(dim=1)  # [batch size * 1 * hidden size]
        # encoder_token_embedding = output.encoder_last_hidden_state  # [bact size * length * hidden size]
        # attention_output, _ = self.cross_attention(query=decoder_last_token_embedding, key=encoder_token_embedding, value=encoder_token_embedding)
        # concatenation
        decoder_last_token_embedding = torch.sum(output.decoder_hidden_states[-1], dim=1) / content.shape[0]
        encoder_token_embedding = torch.sum(output.encoder_last_hidden_state, dim=1) / content.shape[0]
        attention_output = torch.cat([encoder_token_embedding, decoder_last_token_embedding], dim=1)
        # bert_feature_content, _ = self.attention(attention_output, content_masks)

        feature = attention_output.squeeze()
        pred = self.mlp(feature)
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(feature, alpha))

        return torch.sigmoid(pred.squeeze(1)), domain_pred, output.loss, labels


class T5OUREMOModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, query, answer):
        super(T5OUREMOModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_name).requires_grad_(False)

        self.query = query
        self.answer = answer
        self.parent = {'popularize': ['public'], 'clout': ['public', 'emotion'], 'conflict': ['emotion', 'individual'],
                       'smear': ['individual'], 'bias': ['individual'], 'connect': ['individual']}
        for name, param in self.model.named_parameters():
            if name.startswith("encoder.block.11.layer"):
                param.requires_grad = True
            if name.startswith("decoder.block.11.layer"):
                param.requires_grad = True
        # self.cross_attention = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attention_head, batch_first=True)
        # self.fusion_mlp = torch.nn.Linear(emb_dim, emb_dim)
        self.mlp = MLP(emb_dim * 2 + 38, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        answer_list = [self.answer['yes'], self.answer['no']]
        content, content_masks = kwargs['content'], kwargs['content_masks']
        emotion = kwargs['emotion']
        with torch.no_grad():
            answers = {}
            batch_size = content.shape[0]
            decoder_ids_seq = torch.tensor([0 for _ in range(batch_size)]).unsqueeze(dim=1).cuda()
            encoder_output = self.model.encoder(input_ids=content)
            for q1 in ['public', 'emotion', 'individual']:
                # generate answers for three second-layer intents
                decoder_ids_seq = torch.cat([decoder_ids_seq, self.query[q1].repeat(batch_size, 1)], dim=1)
                output = self.model(input_ids=None, encoder_outputs=encoder_output, decoder_input_ids=decoder_ids_seq, output_hidden_states=False)
                index = output.logits[:, -1, answer_list].max(dim=1)[1]
                result = [answer_list[i] for i in index]
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.tensor(result).unsqueeze(dim=1).cuda()], dim=1)
                answers[q1] = index
            # generate answers for leaf intents
            for q2 in ['popularize', 'clout', 'conflict', 'smear', 'bias', 'connect']:
                temp = []
                flag = [0 for _ in range(batch_size)]
                for i in range(batch_size):  # construct questions
                    for pare in self.parent[q2]:
                        if answers[pare][i] == 1: flag[i] = 1  # if its parents exist 1, this question should be queried
                    if flag[i] == 1: temp.append(self.query[q2])
                    else: temp.append(torch.zeros(self.query[q2].shape[0], dtype=torch.int64).cuda())  # padding with 0 (T5 pad)
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.stack(temp)], dim=1)
                output = self.model(input_ids=None, encoder_outputs=encoder_output, decoder_input_ids=decoder_ids_seq, output_hidden_states=False)
                index = output.logits[:, -1, answer_list].max(dim=1)[1]
                result = [answer_list[i] for i in index]
                temp = []
                for i in range(batch_size):  # construct answers
                    if flag[i] == 1: temp.append(result[i])
                    else: temp.append(torch.zeros(1, dtype=torch.int64).cuda())  # padding
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.stack(temp)], dim=1)
            labels = torch.cat([decoder_ids_seq, torch.ones([batch_size, 1], dtype=torch.int64).cuda()], dim=1)[:, 1:].contiguous()  # add </s>

        # output = self.model(input_ids=content, decoder_input_ids=decoder_ids_seq, output_hidden_states=True)
        output = self.model(input_ids=content, labels=labels, output_hidden_states=True)
        # fusion approach 1: attention fusion
        # decoder_last_token_embedding = output.decoder_hidden_states[-1][:, -1, :].unsqueeze(dim=1)  # [batch size * 1 * hidden size]
        # encoder_token_embedding = output.encoder_last_hidden_state  # [bact size * length * hidden size]
        # attention_output, _ = self.cross_attention(query=decoder_last_token_embedding, key=encoder_token_embedding, value=encoder_token_embedding)
        # fusion approach 2: concatenation
        decoder_last_token_embedding = torch.sum(output.decoder_hidden_states[-1], dim=1) / content.shape[0]
        encoder_token_embedding = torch.sum(output.encoder_last_hidden_state, dim=1) / content.shape[0]
        attention_output = torch.cat([encoder_token_embedding, decoder_last_token_embedding, emotion], dim=1)
        # bert_feature_content, _ = self.attention(attention_output, content_masks)

        feature = attention_output.squeeze()
        pred = self.mlp(feature)
        return torch.sigmoid(pred.squeeze(1)), feature, output.loss, labels


class T5OURModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, query, answer):
        super(T5OURModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_name).requires_grad_(False)

        self.query = query
        self.answer = answer
        # self.hierarchy = {'public': ['popularize', 'clout'], 'emotion': ['clout', 'conflict'], 'individual': ['smear', 'bias', 'connect']}
        self.parent = {'popularize': ['public'], 'clout': ['public', 'emotion'], 'conflict': ['emotion', 'individual'],
                       'smear': ['individual'], 'bias': ['individual'], 'connect': ['individual']}
        for name, param in self.model.named_parameters():
            if name.startswith("encoder.block.11.layer"):
                param.requires_grad = True
            if name.startswith("decoder.block.11.layer"):
                param.requires_grad = True
        # self.cross_attention = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attention_head, batch_first=True)
        # self.fusion_mlp = torch.nn.Linear(emb_dim, emb_dim)
        self.mlp = MLP(emb_dim * 2, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        answer_list = [self.answer['yes'], self.answer['no']]
        content, content_masks = kwargs['content'], kwargs['content_masks']
        # decoder_labels_seq = self.query['public'].repeat(content.shape[0], 1)  # initialize decoder sequence on the label side
        #
        # output = self.model(input_ids=content, labels=decoder_labels_seq, output_hidden_states=True)
        # index = output.logits[:, -1, answer_list].max(dim=1)[1]
        # result = [answer_list[i] for i in index]
        # decoder_labels_seq = torch.cat([decoder_labels_seq, torch.tensor(result).unsqueeze(dim=1).cuda()], dim=1)

        with torch.no_grad():
            answers = {}
            batch_size = content.shape[0]
            decoder_ids_seq = torch.tensor([0 for _ in range(batch_size)]).unsqueeze(dim=1).cuda()
            encoder_output = self.model.encoder(input_ids=content)
            # for q in ['public', 'emotion', 'individual', 'popularize', 'clout', 'conflict', 'smear', 'bias', 'connect']:
            for q1 in ['public', 'emotion', 'individual']:
                # generate answers for three second-layer intents
                decoder_ids_seq = torch.cat([decoder_ids_seq, self.query[q1].repeat(batch_size, 1)], dim=1)
                output = self.model(input_ids=None, encoder_outputs=encoder_output, decoder_input_ids=decoder_ids_seq, output_hidden_states=False)
                index = output.logits[:, -1, answer_list].max(dim=1)[1]
                result = [answer_list[i] for i in index]
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.tensor(result).unsqueeze(dim=1).cuda()], dim=1)
                answers[q1] = index
            # generate answers for leaf intents
            for q2 in ['popularize', 'clout', 'conflict', 'smear', 'bias', 'connect']:
                temp = []
                flag = [0 for _ in range(batch_size)]
                for i in range(batch_size):  # construct questions
                    for pare in self.parent[q2]:
                        if answers[pare][i] == 0: flag[i] = 1  # if its parents exist 0(yes), this question should be queried
                    if flag[i] == 1: temp.append(self.query[q2])
                    else: temp.append(torch.zeros(self.query[q2].shape[0], dtype=torch.int64).cuda())  # padding with 0 (T5 pad)
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.stack(temp)], dim=1)
                output = self.model(input_ids=None, encoder_outputs=encoder_output, decoder_input_ids=decoder_ids_seq, output_hidden_states=False)
                index = output.logits[:, -1, answer_list].max(dim=1)[1]
                result = [answer_list[i] for i in index]
                temp = []
                for i in range(batch_size):  # construct answers
                    if flag[i] == 1: temp.append(result[i])
                    else: temp.append(torch.zeros(1, dtype=torch.int64).cuda())  # padding
                decoder_ids_seq = torch.cat([decoder_ids_seq, torch.stack(temp)], dim=1)
            labels = torch.cat([decoder_ids_seq, torch.ones([batch_size, 1], dtype=torch.int64).cuda()], dim=1)[:, 1:].contiguous()  # add </s>

        # output = self.model(input_ids=content, decoder_input_ids=decoder_ids_seq, output_hidden_states=True)
        output = self.model(input_ids=content, labels=labels, output_hidden_states=True)
        # fusion approach 1: attention fusion
        # decoder_last_token_embedding = output.decoder_hidden_states[-1][:, -1, :].unsqueeze(dim=1)  # [batch size * 1 * hidden size]
        # encoder_token_embedding = output.encoder_last_hidden_state  # [bact size * length * hidden size]
        # attention_output, _ = self.cross_attention(query=decoder_last_token_embedding, key=encoder_token_embedding, value=encoder_token_embedding)
        # fusion approach 2: concatenation
        decoder_last_token_embedding = torch.sum(output.decoder_hidden_states[-1], dim=1) / content.shape[0]
        encoder_token_embedding = torch.sum(output.encoder_last_hidden_state, dim=1) / content.shape[0]
        attention_output = torch.cat([encoder_token_embedding, decoder_last_token_embedding], dim=1)
        # bert_feature_content, _ = self.attention(attention_output, content_masks)

        feature = attention_output.squeeze()
        pred = self.mlp(feature)
        return torch.sigmoid(pred.squeeze(1)), feature, output.loss, labels


class T5MDModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, attention_head):
        super(T5MDModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_name).requires_grad_(False)


        for name, param in self.model.named_parameters():
            if name.startswith("encoder.block.11.layer"):
                param.requires_grad = True
            if name.startswith("decoder.block.11.layer"):
                param.requires_grad = True
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attention_head, batch_first=True)
        self.fusion_mlp = torch.nn.Linear(emb_dim, emb_dim)
        self.mlp = MLP(emb_dim * 2, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']

        output = self.model(input_ids=content, decoder_input_ids=content, output_hidden_states=True)
        # attention fusion
        # decoder_last_token_embedding = output.decoder_hidden_states[-1][:, -1, :].unsqueeze(dim=1)  # [batch size * 1 * hidden size]
        # encoder_token_embedding = output.encoder_last_hidden_state  # [bact size * length * hidden size]
        # attention_output, _ = self.cross_attention(query=decoder_last_token_embedding, key=encoder_token_embedding, value=encoder_token_embedding)
        # concatenation
        decoder_last_token_embedding = torch.sum(output.decoder_hidden_states[-1], dim=1) / content.shape[0]
        encoder_token_embedding = torch.sum(output.encoder_last_hidden_state, dim=1) / content.shape[0]
        attention_output = torch.cat([encoder_token_embedding, decoder_last_token_embedding], dim=1)
        # bert_feature_content, _ = self.attention(attention_output, content_masks)

        feature = attention_output.squeeze()
        output = self.mlp(feature)
        return torch.sigmoid(output.squeeze(1)), feature


class T5EMOMDModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, attention_head):
        super(T5EMOMDModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_name).requires_grad_(False)


        for name, param in self.model.named_parameters():
            if name.startswith("encoder.block.11.layer"):
                param.requires_grad = True
            if name.startswith("decoder.block.11.layer"):
                param.requires_grad = True
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attention_head, batch_first=True)
        self.fusion_mlp = torch.nn.Linear(emb_dim, emb_dim)
        self.mlp = MLP(emb_dim * 2 + 38, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        emotion = kwargs['emotion']

        output = self.model(input_ids=content, decoder_input_ids=content, output_hidden_states=True)
        # attention fusion
        # decoder_last_token_embedding = output.decoder_hidden_states[-1][:, -1, :].unsqueeze(dim=1)  # [batch size * 1 * hidden size]
        # encoder_token_embedding = output.encoder_last_hidden_state  # [bact size * length * hidden size]
        # attention_output, _ = self.cross_attention(query=decoder_last_token_embedding, key=encoder_token_embedding, value=encoder_token_embedding)
        # concatenation
        decoder_last_token_embedding = torch.sum(output.decoder_hidden_states[-1], dim=1) / content.shape[0]
        encoder_token_embedding = torch.sum(output.encoder_last_hidden_state, dim=1) / content.shape[0]
        attention_output = torch.cat([encoder_token_embedding, decoder_last_token_embedding, emotion], dim=1)
        # bert_feature_content, _ = self.attention(attention_output, content_masks)

        feature = attention_output.squeeze()
        output = self.mlp(feature)
        return torch.sigmoid(output.squeeze(1)), feature


class T5EANNMDModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(T5EANNMDModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_name).requires_grad_(False)


        for name, param in self.model.named_parameters():
            if name.startswith("encoder.block.11.layer"):
                param.requires_grad = True
            if name.startswith("decoder.block.11.layer"):
                param.requires_grad = True
        # self.cross_attention = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attention_head, batch_first=True)
        # self.fusion_mlp = torch.nn.Linear(emb_dim, emb_dim)
        domain_num = 3
        self.domain_classifier = nn.Sequential(MLP(emb_dim * 2, [mlp_dims], dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims, domain_num))
        self.mlp = MLP(emb_dim * 2, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        alpha = kwargs['alpha']

        output = self.model(input_ids=content, decoder_input_ids=content, output_hidden_states=True)
        # attention fusion
        # decoder_last_token_embedding = output.decoder_hidden_states[-1][:, -1, :].unsqueeze(dim=1)  # [batch size * 1 * hidden size]
        # encoder_token_embedding = output.encoder_last_hidden_state  # [bact size * length * hidden size]
        # attention_output, _ = self.cross_attention(query=decoder_last_token_embedding, key=encoder_token_embedding, value=encoder_token_embedding)
        # concatenation
        decoder_last_token_embedding = torch.sum(output.decoder_hidden_states[-1], dim=1) / content.shape[0]
        encoder_token_embedding = torch.sum(output.encoder_last_hidden_state, dim=1) / content.shape[0]
        attention_output = torch.cat([encoder_token_embedding, decoder_last_token_embedding], dim=1)
        # bert_feature_content, _ = self.attention(attention_output, content_masks)

        feature = attention_output.squeeze()
        output = self.mlp(feature)
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(feature, alpha))

        return torch.sigmoid(output.squeeze(1)), domain_pred


class T5MDFENDMDModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout, attention_head):
        super(T5MDFENDMDModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrain_name).requires_grad_(False)
        self.emb_dim = emb_dim
        self.domain_num = 3
        self.num_expert = 5
        for name, param in self.model.named_parameters():
            if name.startswith("encoder.block.11.layer"):
                param.requires_grad = True
            if name.startswith("decoder.block.11.layer"):
                param.requires_grad = True
        # self.cross_attention = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=attention_head, batch_first=True)
        # self.fusion_mlp = torch.nn.Linear(emb_dim, emb_dim)
        self.mlp = MLP(320, [mlp_dims], dropout)
        # self.attention = MaskAttention(emb_dim * 2)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim * 2))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 3, mlp_dims),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims, self.num_expert),
                                  nn.Softmax(dim=1))

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        domain_labels = kwargs['year']

        output = self.model(input_ids=content, decoder_input_ids=content, output_hidden_states=True)
        # attention fusion
        # decoder_last_token_embedding = output.decoder_hidden_states[-1][:, -1, :].unsqueeze(dim=1)  # [batch size * 1 * hidden size]
        # encoder_token_embedding = output.encoder_last_hidden_state  # [bact size * length * hidden size]
        # attention_output, _ = self.cross_attention(query=decoder_last_token_embedding, key=encoder_token_embedding, value=encoder_token_embedding)
        # concatenation
        concatened = torch.cat([output.decoder_hidden_states[-1], output.encoder_last_hidden_state], dim=-1)
        attention_output = torch.sum(concatened, dim=1) / content.shape[0]

        # decoder_last_token_embedding = torch.sum(output.decoder_hidden_states[-1], dim=1) / content.shape[0]
        # encoder_token_embedding = torch.sum(output.encoder_last_hidden_state, dim=1) / content.shape[0]
        # attention_output = torch.cat([encoder_token_embedding, decoder_last_token_embedding], dim=1)
        # bert_feature_content, _ = self.attention(attention_output, content_masks)
        shared_feature = 0
        if self.training:
            idxs = torch.tensor([index for index in domain_labels]).view(-1, 1).cuda()
            domain_embedding = self.domain_embedder(idxs).squeeze(1)
        else:
            batchsize = content.size(0)
            domain_embedding = self.domain_embedder(torch.LongTensor(range(self.domain_num)).cuda()).squeeze(1).mean(dim=0, keepdim=True).expand(batchsize, self.emb_dim)

        gate_input = torch.cat([domain_embedding, attention_output], dim=-1)
        gate_value = self.gate(gate_input)
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](concatened)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        feature = attention_output.squeeze()
        output = self.mlp(shared_feature)
        return torch.sigmoid(output.squeeze(1)), feature


class BERTModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"): \
                    # or name.startswith('encoder.layer.10') \
                # or name.startswith('encoder.layer.9') \
                # or name.startswith('encoder.layer.8') \
                # or name.startswith('encoder.layer.7') \
                # or name.startswith('encoder.layer.6') \
                # or name.startswith('encoder.layer.5') \
                # or name.startswith('encoder.layer.4') \
                # or name.startswith('encoder.layer.3'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.mlp = MLP(emb_dim, [mlp_dims], dropout)
        self.attention = MaskAttention(emb_dim)

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']

        bert_feature_content = self.bert(content, attention_mask=content_masks)[0]  # opendelta is a in-place tool, so using forward function of self.bert
        bert_feature_content, _ = self.attention(bert_feature_content, content_masks)

        output = self.mlp(bert_feature_content)
        return torch.sigmoid(output.squeeze(1)), bert_feature_content


class BERTEmoModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(BERTEmoModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)

        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.fea_size = emb_dim
        if 'chinese' in pretrain_name: self.mlp = MLP(emb_dim * 2 + 47, [mlp_dims], dropout)
        else: self.mlp = MLP(emb_dim * 2 + 38, [mlp_dims], dropout)
        self.rnn = nn.GRU(input_size=emb_dim,
                          hidden_size=self.fea_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.attention = MaskAttention(emb_dim * 2)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        emotion = kwargs['emotion']

        bert_feature = self.bert(inputs, attention_mask=masks)[0]
        feature, _ = self.rnn(bert_feature)
        feature, _ = self.attention(feature, masks)
        output = self.mlp(torch.cat([feature, emotion], dim=1))
        return torch.sigmoid(output.squeeze(1)), feature


class EANNModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(EANNModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)
        self.embedding = self.bert.embeddings
        self.bertembedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder
        domain_num = 3

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.convs = cnn_extractor(feature_kernel, emb_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.classifier = MLP(mlp_input_shape, [mlp_dims], dropout)
        self.domain_classifier = nn.Sequential(MLP(mlp_input_shape, [mlp_dims], dropout, False), torch.nn.ReLU(),
                                               torch.nn.Linear(mlp_dims, domain_num))
        self.adapter = nn.ModuleList([self.convs, self.classifier, self.domain_classifier])

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        alpha = kwargs['alpha']

        bert_feature = self.bert(inputs, attention_mask=masks)[0]

        feature = self.convs(bert_feature)
        output = self.classifier(feature)
        reverse = ReverseLayerF.apply
        domain_pred = self.domain_classifier(reverse(feature, alpha))
        return torch.sigmoid(output.squeeze(1)), domain_pred


class MDFENDModel(torch.nn.Module):
    def __init__(self, pretrain_name, emb_dim, mlp_dims, dropout):
        super(MDFENDModel, self).__init__()
        self.domain_num = 3
        self.num_expert = 5
        self.emb_dim = emb_dim
        self.bert = BertModel.from_pretrained(pretrain_name).requires_grad_(False)
        self.embedding = self.bert.embeddings
        self.bertencoder = self.bert.encoder

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, mlp_dims),
                                  nn.ReLU(),
                                  nn.Linear(mlp_dims, self.num_expert),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)

        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.classifier = MLP(320, [mlp_dims], dropout)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        domain_labels = kwargs['year']

        init_feature = self.bert(inputs, attention_mask=masks)[0]
        gate_input_feature, _ = self.attention(init_feature, masks)
        shared_feature = 0
        if self.training:
            idxs = torch.tensor([index for index in domain_labels]).view(-1, 1).cuda()
            domain_embedding = self.domain_embedder(idxs).squeeze(1)
        else:
            batchsize = inputs.size(0)
            domain_embedding = self.domain_embedder(torch.LongTensor(range(self.domain_num)).cuda()).squeeze(1).mean(dim=0, keepdim=True).expand(batchsize, self.emb_dim)

        gate_input = torch.cat([domain_embedding, gate_input_feature], dim=-1)
        gate_value = self.gate(gate_input)
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1)), shared_feature
