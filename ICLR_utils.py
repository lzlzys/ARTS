# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_gpt_socre = 1000000
    def compare(self, score, gpt_score):
        for i in range(len(score)):

            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i] + self.delta:

                self.best_gpt_socre = gpt_score
                return False
        return True

    def __call__(self, score, model, gpt_model,gpt_score):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_gpt_socre = gpt_score
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
            self.save_checkpoint2(score, gpt_model)
        elif self.compare(score, gpt_score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.save_checkpoint2(score, gpt_model)
            self.counter = 0

        return self.counter
    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f"Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score
    def save_checkpoint2(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f"Validation score increased.  Saving model ...")
        torch.save(model.state_dict(), 'output/ICLRec-Yelp-Toys_and_Games-gpt-new.pt')


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def get_user_seqs(args):
    seq2 = []
    seq = []
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
    user_seq = []
    item_set = set()
    if args.amazon == False:
        a = np.load('/home/t618141/python_code/new_lzl_model/data/10_core_{}_PEPLER.npy'.format(args.data_name), allow_pickle=True)  # len 1285
        dict1 = {}
        dict2 = {}
        max_num = 0
        for i in a:
            if max_num < i['user']:
                max_num = i['user']
            a, b, seq1, d = i['template']

            tokens = tokenizer(seq1)['input_ids']
            text = tokenizer.decode(tokens[:args.max_len])
            seq2.append('{} {} {}'.format(bos, text, eos))
            if i['user'] not in dict1:#user_id

                new_list_seq = []
                new_list = []
                new_list.append(i['item'])#item_id
                new_list_seq.append(text)
                dict1[i['user']] = new_list
                dict2[i['user']] = new_list_seq
            else:

                dict1[i['user']].append(i['item'])
                dict2[i['user']].append(text)
        for i in range(1, len(dict1) + 1):
            items = dict1[i]

            seq1 = dict2[i]
            seq.append(seq1)

            user_seq.append(items)
            item_set = item_set | set(items)
        encoded_inputs = tokenizer(seq2, padding=True, return_tensors='pt')
        text_seq = encoded_inputs['input_ids'].contiguous()
    else:
        data = np.load('/data/t618141/{}_10_1.npy'.format(args.data_name), allow_pickle=True)
        max_num = 0
        item_dict = {}
        j = 1
        for i in data:
            if i['reviewerID'] not in item_dict:
                item_dict[i['reviewerID']] = j
                j += 1
        # 创建一个字典，键为reviewerid，值为一个列表，存储该用户访问过的项目和时间
        user_items = {}
        for row in data:
            # 获取用户ID，项目ID和时间戳
            review = row['new_reviewText']

            tokens = tokenizer(review)['input_ids']
            text = tokenizer.decode(tokens[:20])
            review = ('{} {} {}'.format(bos, text, eos))
            if max_num < item_dict[row['reviewerID']]:
                max_num = item_dict[row['reviewerID']]
            user, item, time = row['reviewerID'], row['asin'], row['unixReviewTime']
            # 获取用户ID和项目ID
            # 如果用户ID不在字典中，则创建一个空列表
            if user not in user_items:
                user_items[user] = []
            # 将项目ID和时间戳作为一个元组添加到列表中
            user_items[user].append((item, time, review))


        # 创建一个字典，键为asin，值为一个整数，表示该项目的类型
        item_types = {}
        # 初始化类型编号为0
        type_id = 1
        for row in data:
            # 获取项目ID
            item = row['asin']
            # 如果项目ID不在字典中，则创建一个新的类型编号，并加1
            if item not in item_types:
                item_types[item] = type_id
                type_id += 1
        dict1 = {}
        dict2 = {}
        # 遍历每个用户
        for user, real_items in user_items.items():
            # 按照时间戳升序排序列表中的元素
            real_items.sort(key=lambda x: x[1])
            # 将项目ID替换为对应的类型编号，并只保留类型编号
            items = [item_types[item[0]] for item in real_items]
            review = [item[2] for item in real_items]
            # 打印用户ID和访问过的项目类型列表

            dict1[item_dict[user]] = items
            dict2[item_dict[user]] = review
            for j in review:
                seq2.append(j)

        for i in range(1, len(dict1) + 1):
            items = dict1[i]
            seq1 = dict2[i]
            seq.append(seq1)
            user_seq.append(items)
            item_set = item_set | set(items)
        encoded_inputs = tokenizer(seq2, padding=True, return_tensors='pt')
        text_seq = encoded_inputs['input_ids'].contiguous()

    max_item = max(item_set)

    num_users = max_num
    num_items = max_item + 3

    print(num_users)
    print(num_items)
    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, seq, max_item, valid_rating_matrix, test_rating_matrix, text_seq, max_num



def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        long_sequence.extend(items)  # 后面的都是采的负例
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(" ", 1)
        items = items.split(" ")
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)  # 331
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
