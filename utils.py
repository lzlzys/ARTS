import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# def evaluate(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     HT = 0.0
#     valid_user = 0.0

#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
#     for u in users:

#         if len(train[u]) < 1 or len(test[u]) < 1: continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        
#         # predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        
#         predictions = predictions[0] # - for 1st argsort DESC
    
#         rank = predictions.argsort().argsort()[0].item()
        
#         # rank = predictions.argsort()[0].item()
#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user



def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] =dataset.user_train,dataset.user_valid,dataset.user_test,dataset.usernum,dataset.itemnum                   
    print(len(train.keys()))
    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_5 = 0.0
    HT_5 = 0.0
    valid_user = 0.0
    # users = range(1, usernum + 1)
    for u in (train.keys()):
        if len(train[u]) < 1 or len(test[u]) < 1: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        # predictions = -model.predict(*[np.array(l) for l in [[seq], item_idx]])
        predictions = -model.predict(torch.tensor([u]).to(args.device),torch.tensor([seq]).to(args.device),np.array(item_idx))
        predictions = predictions[0] # - for 1st argsort DESC
        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG_5 / valid_user, HT_5 / valid_user , NDCG_10 / valid_user, HT_10 / valid_user



def get_eval(predlist, truelist, klist):#return recall@k and mrr@k
    recall = []
    mrr = []
    predlist = predlist.argsort()
    for k in klist:
        recall.append(0)
        mrr.append(0)
        templist = predlist[:,-k:]#the result of argsort is in ascending 
        i = 0
        while i < len(truelist):
            pos = torch.argwhere(templist[i]==(truelist[i]-1))#pos is a list of positions whose values are all truelist[i]
        
            if len(pos) > 0:
                recall[-1] += 1
                mrr[-1] += 1/(k-pos[0][0])
            else:
                recall[-1] += 0
                mrr[-1] += 0
            i += 1
    return recall, mrr