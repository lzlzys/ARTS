# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import datetime
from  PEPLER_utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
import math
import numpy as np
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from tqdm import tqdm
import random
from PEPLER_module import ContinuousPromptLearning
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import RandomSampler
from transformers import GPT2Tokenizer, AdamW

from ICLR_models import KMeans

from ICLR_module import NCELoss, NTXent, SupConLoss, PCLoss
from ICLR_utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr
from PEPLER_utils import Batchify, DataLoader

d_model = 768  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 12  # number of heads in Multi-Head Attention


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '

class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        self.clusters = []
        for num_intent_cluster in self.num_intent_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size * self.args.max_seq_length,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        # projection head for contrastive learn task
        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True),
        )
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)

    def train(self, epoch, args, eval_dataloader):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader, args=args, eval_dataloader=eval_dataloader, train='train')

    def valid(self, epoch, args, full_sort=False, test_seq=None):
        return self.iteration(epoch, self.eval_dataloader, args=args, full_sort=full_sort, train='valid', test_seq=test_seq)

    def test(self, epoch, args, full_sort=False, test_seq=None, early_epoch=None):
        return self.iteration(epoch, self.test_dataloader, args=args, full_sort=full_sort, train='test', test_seq=test_seq, early_epoch=early_epoch)

    def iteration(self, epoch, dataloader,full_sort=False, train=True, A=None):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [1, 5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(recall[0]),
            "NDCG@1": "{:.4f}".format(ndcg[0]),
            "HIT@5": "{:.4f}".format(recall[1]),
            "NDCG@5": "{:.4f}".format(ndcg[1]),
            "HIT@10": "{:.4f}".format(recall[2]),
            "NDCG@10": "{:.4f}".format(ndcg[2]),
            "HIT@20": "{:.4f}".format(recall[4]),
            "NDCG@20": "{:.4f}".format(ndcg[4]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):

        pos_ids = pos_ids[:,0:pos_ids.size(1) - 1]#为了对齐
        neg_ids = neg_ids[:,0:neg_ids.size(1) - 1]#为了对齐
        seq_out = seq_out[:,1:seq_out.size(1),:]
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = seq_out.contiguous().view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        # istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()
        istarget = (pos_ids > 0).view(pos.size(0)).float()

        # [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=500):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(6285, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        self.cls_token = nn.Parameter(self.src_emb(torch.LongTensor(torch.arange(5000, 5000 + 1285))))
    def forward(self, enc_inputs, j):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # if self.cls_token.grad != None:
        #     print(self.cls_token.grad[10 * (3 - 1): 10 * 3,:])
        #     print(j)
        mask = torch.eq(enc_inputs, 0)
        pad_count = torch.sum(mask, dim=1)
        enc_outputs = self.src_emb(torch.LongTensor(enc_inputs.cpu()).cuda())  # [batch_size, src_len, d_model]
        for i in range(pad_count.size(0)):#加cls_token
            if pad_count[i] == 0:#如果没有pad，则加上cls_token
                left = self.cls_token[i + (j * 10),:].unsqueeze(0)
                right = self.src_emb(torch.LongTensor(enc_inputs[i,:].cpu()).cuda())
                new_enc_outputs = torch.cat((left, right),dim=0)
                enc_outputs[i] = new_enc_outputs[0:-1, :]
            else:#如果有pad，则将序列拆成两个序列，一个是pad，一个是实际的交互序列，然后在交互序列的最前面加上cls_token
                left = enc_inputs[i][:pad_count[i]]
                left = self.src_emb(torch.LongTensor(left.cpu()).cuda())
                right = enc_inputs[i][pad_count[i]:]
                right = self.src_emb(torch.LongTensor(right.cpu()).cuda())
                # w = torch.LongTensor(self.cls_token[int(user_id[i].item()) -1].cpu().long().unsqueeze(0))
                # a = self.src_emb(w.cuda(2))
                new_enc_outputs = torch.cat((left, self.cls_token[i + (j * 10),:].unsqueeze(0), right), dim=0)
                enc_outputs[i] = new_enc_outputs[0:-1, :]

        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        # self.decoder = Decoder().cuda(2)
        # self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda(2)

    def forward(self, enc_inputs, j):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''

        # tensor to store decoder outputs
        enc_outputs, enc_self_attns = self.encoder(enc_inputs, j)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        # dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return enc_outputs

import torch.nn.functional as F
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    def forward(self, q, k):
        # q和k是两个特征矩阵，每一行是一个样本的特征向量
        # 计算q和k的余弦相似度矩阵
        norm_item_emb1 = torch.flatten(q, start_dim=1, end_dim=2)
        norm_item_emb2 = torch.flatten(k, start_dim=1, end_dim=2)
        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_item_emb2.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.temperature)
        ttl_score_item = torch.exp(ttl_score_item / self.temperature).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = ssl_loss_item

        return ssl_loss


def contrastive_loss(output_vectors, label_vectors, margin=1.0):
    output_vectors = torch.flatten(output_vectors, start_dim=1, end_dim=2)
    label_vectors = torch.flatten(label_vectors, start_dim=1, end_dim=2)
    # 计算余弦相似度矩阵
    similarity_matrix = F.cosine_similarity(output_vectors.unsqueeze(1), label_vectors.unsqueeze(0), dim=2)

    # 构建标签矩阵，1表示正样本，0表示负样本
    labels = torch.eye(len(output_vectors)).to(output_vectors.device)

    # 计算对比损失
    positive_similarity = torch.diagonal(similarity_matrix)
    negative_similarity = similarity_matrix - labels * similarity_matrix  # 将正样本位置置为0
    negative_similarity = torch.max(negative_similarity, dim=1)[0]  # 取最大的负样本相似度

    loss = F.margin_ranking_loss(negative_similarity, positive_similarity, target=torch.ones_like(positive_similarity), margin=margin)

    return loss

# # 示例用法
# batch_size = 32
# dim = 64
#
# # 模型输出向量和标签向量示例，实际使用时需要替换成你的数据
# output_vectors = torch.rand((batch_size, dim))
# label_vectors = torch.rand((batch_size, dim))
#
# # 计算对比损失
# loss = contrastive_loss(output_vectors, label_vectors)
#
# print("Contrastive Loss:", loss.item())


from PG_model import SASRec
import torch.optim as optim
class ICLRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(ICLRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )

        # for param in model.parameters():
        #     param.requires_grad = False

        bos = '<bos>'
        self.eos = '<eos>'
        pad = '<pad>'
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=self.eos, pad_token=pad)
        corpus = DataLoader(args.data_path, args.index_dir, self.tokenizer, args.words, args.args_str)  # 和ICLR数据对应
        # self.train_data = Batchify(corpus.train, self.tokenizer, bos, self.eos, args.batch_size, shuffle=True)
        # self.val_data = Batchify(corpus.valid, self.tokenizer, bos, self.eos, args.batch_size)
        self.test_data = Batchify(corpus.test, self.tokenizer, bos, self.eos, args.batch_size)


        # nuser = len(corpus.user_dict)
        # nitem = len(corpus.item_dict)
        nuser = args.user_num
        nitem = args.item_num
        ntoken = len(self.tokenizer)
        gpt2_model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)
        gpt2_model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table

        gpt2_model.to(args.device)
        self.gpt2_model = gpt2_model
        self.dropout_rate = nn.Dropout(args.hidden_dropout_prob)
        self.prompt_generator_model = SASRec(args, args.all_size)
        # self.prompt_generator_model.load_state_dict(torch.load('/home/t618141/python_code/test_model/model/{}_PG_FMLP_data.pth'.format(args.args_str)))
        self.prompt_generator_model.to(args.device)
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim1 = Adam(self.prompt_generator_model.parameters(), lr=args.prompt_generator_lr, betas=betas, weight_decay=self.args.weight_decay)
        params = list(self.model.parameters()) + list(self.gpt2_model.parameters())
        self.optim = Adam(list(self.model.parameters()), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.info_nce_loss = InfoNCELoss(temperature=1)
        self.PG_optim = Adam(list(self.model.parameters()), lr=args.lr * 0.01, betas=betas, weight_decay=self.args.weight_decay)
        # self.optim1 = Adam(self.prompt_generator_model.parameters(), lr=args.prompt_generator_lr, betas=betas, weight_decay=self.args.weight_decay)#lr建议config
    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None,A=None, args=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        逻辑问题？？？？？？
        """

        cl_batch = torch.cat(inputs, dim=0)#20*20
        A = torch.cat((A,A), dim=0)#20*768
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model.forward2(cl_batch, A)#20*20*768

        cl_sequence_output = cl_sequence_output[:,1:cl_sequence_output.size(1),:]

        if self.args.seq_representation_instancecl_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        # cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_sequence_flatten = cl_sequence_output.view(cl_sequence_output.size(0), -1)
        # cf_output = self.projection(cf_sequence_flatten)
        # batch_size = cl_batch.shape[0] // 2
        batch_size = cl_sequence_output.size(0)// 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)

        # print(cl_output_slice[0].size())

        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)

        return cl_loss


    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids, A=None, args=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        intents: [num_clusters batch_size hidden_dims]
        """


        intent_ids[0] = intent_ids[0][1:,]
        intents[0] = intents[0][1:,:]
        A = torch.cat((A, A), dim=0)
        n_views, (bsz, seq_len) = len(inputs), inputs[0][0:args.batch_size - 1].shape
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model.forward2(cl_batch, A)

        cl_sequence_output = cl_sequence_output[:, 1:cl_sequence_output.size(1), :]
        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        # cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)
        cl_sequence_flatten = cl_sequence_output.view(cl_sequence_output.size(0), -1)

        bsz = int(cl_sequence_flatten.size(0) / 2)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)
        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)

        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)

        return cl_loss

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True, args=False, eval_dataloader=None, test_seq=None, early_epoch=None):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar


        if train == 'train':


            # ------ intentions clustering ----- #
            if self.args.contrast_type in ["IntentCL", "Hybrid"] and epoch >= self.args.warm_up_epoches:
                print("Preparing Clustering:")
                self.model.eval()
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
                for i, (rec_batch, _, _) in rec_cf_data_iter:#ICLR聚类
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    user_id_index, input_ids, target_pos, target_neg, _, seq, mask, noise_data = rec_batch

                    # A1 = A[i * input_ids.size(0):(i + 1) * input_ids.size(0), :]

                    A1, prompt_index = self.prompt_generator_model(input_ids, i, user_id_index)
                    A1 = A1.to(args.device)
                    b = torch.arange(A1.size(0))
                    i = torch.stack([b, prompt_index], dim=1)


                    i = torch.tensor(i).to(args.device)
                    A1 = torch.index_select(A1, dim=1, index=i)

                    sequence_output = self.model.forward2(input_ids, A1)

                    # average sum
                    if self.args.seq_representation_type == "mean":
                        sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                    sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                    sequence_output = sequence_output.detach().cpu().numpy()
                    kmeans_training_data.append(sequence_output)
                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)

                # train multiple clusters
                print("Training Clusters:")
                for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):
                    cluster.train(kmeans_training_data)
                    self.clusters[i] = cluster
                # clean memory
                del kmeans_training_data
                import gc

                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()
            cl_sum_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            joint_avg_loss = 0
            rec_avg_loss = 0
            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                """
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                """


                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                user_ids, input_ids, target_pos, target_neg, _ , seq, mask, noisy_data = rec_batch

                # ---------- recommendation task ---------------#

                # sequence_output = self.model(input_ids)
                # src = sequence_output


                # A1 = A[i * input_ids.size(0):(i + 1) * input_ids.size(0), :]

                A1, prompt_index = self.prompt_generator_model(input_ids, i, user_ids)
                A1 = A1.to(args.device)
                prompt_index = torch.tensor(prompt_index).to(args.device)
                A1 = torch.index_select(A1, dim=1, index=torch.tensor(prompt_index))
                # #數據增强(dropout
                # mask_aug = torch.empty_like(input_ids).bernoulli_(args.hidden_dropout_prob)
                # # 将输入张量与mask相乘，得到dropout后的张量
                # input_ids_aug = input_ids * mask_aug
                # aug_A1 = self.prompt_generator_model(input_ids_aug, i)
                # aug_A1 = aug_A1[:, 0, :]
                # noisy_A1 = self.prompt_generator_model(noisy_data, i)
                # noisy_A1 = noisy_A1[:, 0, :]
                #
                # sequence_output_aug = self.model.forward2(input_ids_aug, aug_A1)
                # sequence_output_noisy = self.model.forward2(noisy_data, noisy_A1)
                #
                # PG_cl_loss = contrastive_loss(sequence_output_aug, sequence_output_noisy)#暫時用這個


                sequence_output = self.model.forward2(input_ids, A1)

                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)


                B = sequence_output#生成的对应预测Btensor
                # ---------- contrastive learning task -------------#
                cl_losses = []
                for cl_batch in cl_batches:

                    if self.args.contrast_type == "InstanceCL":
                        cl_loss = self._instance_cl_one_pair_contrastive_learning(
                            cl_batch, intent_ids=seq_class_label_batches, A=A1, args=args
                        )
                        cl_losses.append(self.args.cf_weight * cl_loss)
                    elif self.args.contrast_type == "IntentCL":
                        # ------ performing clustering for getting users' intentions ----#
                        # average sum
                        if epoch >= self.args.warm_up_epoches:
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()

                            # query on multiple clusters
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids, A=A1, args=args
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss)
                        else:
                            continue
                    elif self.args.contrast_type == "Hybrid":
                        if epoch < self.args.warm_up_epoches:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches, A=A1, args=args
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                        else:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches, A=A1, args=args
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()
                            # query on multiple clusters
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss3 = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids, A=A1, args=args
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss3)


                ###############################################################################
                # Build the model
                ###############################################################################

                def train(seq, mask, A1):
                    # Turn on training mode which enables dropout.
                    self.gpt2_model.train()
                    text_loss = 0.
                    total_sample = 0
                    idss_predict = []
                    loss1 = 0
                    for i in range(seq.size(0)):

                        A = A1[i,:]
                        B1 = B[i,:]
                        seq1 = seq[i,:]
                        mask1 = mask[i,:]
                        A1 = A1.to(args.device)
                        B1 = B1.to(args.device)
                        seq1 = seq1.to(args.device)
                        mask1 = mask1.to(args.device)
                        # Starting each batch, we detach the hidden state from how it was previously produced.
                        # If we didn't, the model would try backpropagating all the way to start of the dataset.
                        # optimizer.zero_grad()
                        outputs = self.gpt2_model.forward3(A, B1, seq1, mask=mask1)

                        loss = outputs.loss
                        loss1 += loss
                        batch_size = A1.size(0)
                        text_loss += batch_size * loss.item()
                        total_sample += batch_size

                    # if data.step % args.log_interval == 0 or data.step == data.total_step:

                    if i == seq.size(0) - 1:
                        cur_t_loss = text_loss / total_sample
                        # print(f'text_loss = {text_loss}')
                        # print(f'total_sample = {total_sample}')
                        # print(now_time() + 'text ppl {:4.4f}'.format(math.exp(cur_t_loss)))
                        return loss1, cur_t_loss

                # 3将A和B输入到PEPLER产生对应

                for param in self.gpt2_model.parameters():
                    param.requires_grad = True

                loss_PEPLER, cur_t_loss = train(seq, mask, A1)


                for cl_loss in cl_losses:
                    rec_loss += cl_loss

                # joint_loss = rec_loss * self.args.rec_weight + loss_PEPLER * (1 - self.args.rec_weight)
                joint_loss = rec_loss + loss_PEPLER * self.args.rec_weight
                # PG_cl_loss = 0.0000001 * PG_cl_loss

                self.optim.zero_grad()
                self.optim1.zero_grad()
                # self.PG_optim.zero_grad()
                # joint_loss.backward(retain_graph=True)
                rec_loss.backward(retain_graph=True)
                loss_PEPLER.backward(retain_graph=True)
                # PG_cl_loss.backward(retain_graph=True)
                self.optim.step()
                self.optim1.step()
                # self.PG_optim.step()
                rec_avg_loss += rec_loss.item()
                # for name, param in self.model.named_parameters():
                #     if param.grad == None:
                #         print(name, param.grad)

                for i, cl_loss in enumerate(cl_losses):
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()

            post_fix = {
                        "epoch": epoch,
                        # 'cl_avg_loss':"{:.4f}".format(PG_cl_loss / len(rec_cf_data_iter)),
                        "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                        "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
                        "text ppl": "{:4.4f}".format(math.exp(cur_t_loss))
            }#加上loss_PEPLER
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")


        elif train == 'valid':
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            def evaluate(seq, mask, A1):  # 删除吧还是
                # Turn on training mode which enables dropout.
                self.gpt2_model.eval()
                text_loss = 0.
                total_sample = 0
                idss_predict = []
                loss1 = 0
                for i in range(seq.size(0)):

                    A = A1[i, :]
                    B1 = B[i, :]
                    seq1 = seq[i, :]
                    mask1 = mask[i, :]
                    A1 = A1.to(args.device)
                    B1 = B1.to(args.device)
                    seq1 = seq1.to(args.device)
                    mask1 = mask1.to(args.device)
                    # Starting each batch, we detach the hidden state from how it was previously produced.
                    # If we didn't, the model would try backpropagating all the way to start of the dataset.
                    # optimizer.zero_grad()
                    outputs = self.gpt2_model.forward3(A, B1, seq1, mask=mask1)

                    loss = outputs.loss
                    loss1 += loss
                    batch_size = seq.size(0)
                    text_loss += batch_size * loss.item()
                    total_sample += batch_size

                # if data.step % args.log_interval == 0 or data.step == data.total_step:

                if i == seq.size(0) - 1:
                    cur_t_loss = text_loss / total_sample
                    # print(f'text_loss = {text_loss}')
                    # print(f'total_sample = {total_sample}')
                    # print(now_time() + 'text ppl {:4.4f}'.format(math.exp(cur_t_loss)))
                    return cur_t_loss

            if full_sort:
                def generate(data, A, B, seq, mask):
                    # Turn on evaluation mode which disables dropout.
                    self.gpt2_model.eval()
                    idss_predict = []
                    with torch.no_grad():
                        for i in range(seq.size(0)):  # 对应最长评论长度
                            user, item, _, seq2, _ = data.next_batch()  # data.step += 1
                            # text = seq[:, :1].to(args.device)  # bos, (batch_size, 1)
                            text = seq2[:, :1]
                            text = text.to('cuda')
                            A1 = A[i, :]
                            B1 = B[i, :]
                            A1 = A1.to('cuda')
                            B1 = B1.to('cuda')
                            text = text[:1, :1].repeat(B.size(1), 1)
                            for idx in range(seq.size(1)):
                                # produce a word at each step

                                outputs = self.gpt2_model.forward3(A1, B1, text, mask=None)
                                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                                word_prob = torch.softmax(last_token, dim=-1)
                                token = torch.argmax(word_prob, dim=1,
                                                     keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                                text = torch.cat([text, token], 1)

                                if token == self.tokenizer.eos_token:
                                    break
                            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)

                            idss_predict.extend(ids)

                            if data.step == data.total_step:
                                break
                    return idss_predict
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, seq, mask, _ = batch
                    #user_ids, input_ids, target_pos, target_neg, answers = batch

                    A1, prompt_index = self.prompt_generator_model(input_ids, i, user_ids)
                    A1 = A1.to(args.device)
                    prompt_index = torch.tensor(prompt_index).to(args.device)
                    A1 = torch.index_select(A1, dim=1, index=torch.tensor(prompt_index))

                    recommend_output = self.model.forward2(input_ids, A1)
                    sequence_output = recommend_output
                    recommend_output = recommend_output[:, -1, :]

                    B = sequence_output
                    cur_t_loss = evaluate(seq, mask, A1)
                    print(now_time() + 'Generating text')
                    idss_predicted = generate(self.test_data, A1, B, seq, mask)
                    tokens_test = [ids2tokens(ids[1:], self.tokenizer, self.eos) for ids in test_seq.tolist()]#test_data, seq, mask匹配不上的错误？？？？？？？？？？
                    tokens_predict = [ids2tokens(ids, self.tokenizer, self.eos) for ids in idss_predicted]
                    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
                    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
                    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
                    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
                    text_test = [' '.join(tokens) for tokens in tokens_test]
                    text_predict = [' '.join(tokens) for tokens in tokens_predict]
                    ROUGE = rouge_score(text_test, text_predict)  # a dictionary

                    for (k, v) in ROUGE.items():
                        print(now_time() + '{} {:7.4f}'.format(k, v))
                    if i == 0:
                        dictionary = {}
                        dictionary['BLEU1'] = 0
                        dictionary['BLEU4'] = 0
                        for (k, v) in ROUGE.items():
                            dictionary[k] = 0

                    for (k, v) in ROUGE.items():
                        dictionary[k] += v

                    dictionary['BLEU1'] += BLEU1
                    dictionary['BLEU4'] += BLEU4
                    text_out = ''
                    for (real, fake) in zip(text_test, text_predict):
                        text_out += '{}\n{}\n\n'.format(real, fake)
                    with open(args.prediction_path, 'w', encoding='utf-8') as f:
                        f.write(text_out)
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

                # if epoch == 40:
                #     for (k, v) in dictionary.items():
                #         print(now_time() + '{} {:7.4f}'.format(k, v/(h + 1)))
                return self.get_full_sort_score(epoch, answer_list, pred_list), cur_t_loss

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids, train)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
        else:
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            def generate(data, A, B, seq, mask):
                # Turn on evaluation mode which disables dropout.
                self.gpt2_model.eval()
                idss_predict = []
                with torch.no_grad():
                    for i in range(seq.size(0)):#对应最长评论长度
                        user, item, _, seq2, _ = data.next_batch()  # data.step += 1
                        # text = seq[:, :1].to(args.device)  # bos, (batch_size, 1)
                        text = seq2[:, :1]
                        text = text.to('cuda')
                        A1 = A[i, :]
                        B1 = B[i, :]
                        A1 = A1.to('cuda')
                        B1 = B1.to('cuda')
                        text = text[:1, :1].repeat(B.size(1), 1)
                        for idx in range(seq.size(1)):
                            # produce a word at each step

                            outputs = self.gpt2_model.forward3(A1, B1, text, mask=None)
                            last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                            word_prob = torch.softmax(last_token, dim=-1)
                            token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                            text = torch.cat([text, token], 1)

                            if token == self.tokenizer.eos_token:
                                break
                        ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)

                        idss_predict.extend(ids)

                        if data.step == data.total_step:
                            break
                return idss_predict

            if full_sort:
                answer_list = None
                h = 0
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)

                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, seq, mask, _ = batch

                    A1, prompt_index = self.prompt_generator_model(input_ids, i, user_ids)
                    A1 = A1.to(args.device)
                    prompt_index = torch.tensor(prompt_index).to(args.device)
                    A1 = torch.index_select(A1, dim=1, index=prompt_index)

                    recommend_output = self.model.forward2(input_ids, A1)
                    sequence_output = recommend_output
                    recommend_output = recommend_output[:, -1, :]

                    B = sequence_output

                    #generate阶段

                    print(now_time() + 'Generating text')
                    idss_predicted = generate(self.test_data, A1, B, seq, mask)
                    tokens_test = [ids2tokens(ids[1:], self.tokenizer, self.eos) for ids in test_seq.tolist()]#test_data, seq, mask匹配不上的错误？？？？？？？？？？
                    tokens_predict = [ids2tokens(ids, self.tokenizer, self.eos) for ids in idss_predicted]
                    BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
                    print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
                    BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
                    print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
                    text_test = [' '.join(tokens) for tokens in tokens_test]
                    text_predict = [' '.join(tokens) for tokens in tokens_predict]
                    ROUGE = rouge_score(text_test, text_predict)  # a dictionary

                    for (k, v) in ROUGE.items():
                        print(now_time() + '{} {:7.4f}'.format(k, v))
                    if i == 0:
                        dictionary = {}
                        dictionary['BLEU1'] = 0
                        dictionary['BLEU4'] = 0
                        for (k, v) in ROUGE.items():
                            dictionary[k] = 0

                    for (k, v) in ROUGE.items():
                        dictionary[k] += v

                    dictionary['BLEU1'] += BLEU1
                    dictionary['BLEU4'] += BLEU4
                    text_out = ''
                    for (real, fake) in zip(text_test, text_predict):
                        text_out += '{}\n{}\n\n'.format(real, fake)
                    with open(args.prediction_path, 'w', encoding='utf-8') as f:
                        f.write(text_out)
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    # val_loss = evaluate(self.val_data, A , B, seq, mask)
                    # print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss),
                    #                                                                                 val_loss))

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                # if early_epoch == 40:
                #     for (k, v) in dictionary.items():
                #         print(now_time() + '{} {:7.4f}'.format(k, v/(h + 1)))
                return self.get_full_sort_score(epoch, answer_list, pred_list)
            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    # user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs, seq, mask = batch

                    A1 = self.prompt_generator_model(input_ids, i, user_ids)
                    A1 = A1[:, 0, :]

                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)