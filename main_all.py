from torch.utils.data.dataloader import DataLoader
from utils import evaluate , get_eval
from datasets_all import *
import os
import time
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import copy
import argparse

def train_model_all(model,seq_dataloader_train,seq_dataloader_test,optimizer_list,bce_criterion,num_epochs,args,len_):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            if phase == 'train':
                # user_list=[]
                for idx, (seq, target) in enumerate(seq_dataloader_train):
                    # with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        seq = seq.to(args.device)
                        target = target.to(args.device)
                    logits = model(idx,seq)
                    optimizer_list[0].zero_grad()
                
                    loss = bce_criterion(logits, (target-1))
                    # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                    loss.backward()
                    
                    optimizer.step() 
                    if len(optimizer_list) == 2:
                        lr_scheduler.step()    
            # if phase == 'val' and ((epoch+1 )% 20 == 0) :
            if phase == 'val' :
                with torch.no_grad():
                    r3_b=0
                    m3_b=0
                    r5_b = 0
                    m5_b = 0
                    r10_b = 0
                    m10_b = 0
                    r20_b = 0
                    m20_b = 0
                    for idx,(seq,target) in enumerate(seq_dataloader_test):
                        if torch.cuda.is_available():
                            seq=seq.to(args.device)
                            target=target.to(args.device) 
                     
                        logits_B=model(idx,seq)
                        recall,mrr = get_eval(logits_B, target, [3,5,10,20])
                        r3_b += recall[0]
                        m3_b += mrr[0]
                        r5_b += recall[1]
                        m5_b += mrr[1]
                        r10_b += recall[2]
                        m10_b += mrr[2]
                        r20_b += recall[3]
                        m20_b += mrr[3]
                    # if (r10_b+r20_b)/2>best_cirtion :
                    #     best_cirtion=(r10_b+r20_b)/2
                    #     torch.save(model.state_dict(),'model/{},{},{},qianyi={},model_best.pth'.format(args.alpha,args.Beta,args.Gamma,str(args.qianyi)))
                    print('Recall3_b: {:.5f}; Mrr3: {:.5f}'.format(r3_b/len_,m3_b/len_))
                    print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_,m5_b/len_))
                    print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_,m10_b/len_))
                    print('Recall20_b: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len_,m20_b/len_))
                # if (epoch+1 ) == 20:
                #     torch.save(model.state_dict(),'model/SAS.pth')
    torch.save(model.state_dict(),'model/SAS_all.pth')               


def train_model_one(model,seq_dataloader_train_A,data_loader_qianyi_train,seq_dataloader_test_A,data_loader_qianyi_test,optimizer_list,bce_criterion,num_epochs,args,len_):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            if phase == 'train':
                # user_list=[]
                for idx,((seq_a, target_a),(seq_b,target_b)) in enumerate(zip(seq_dataloader_train_A,data_loader_qianyi_train)):
                    # with torch.set_grad_enabled(phase == 'train'):
                    if torch.cuda.is_available():
                        seq_a=seq_a.to(args.device)
                        target_a=target_a.to(args.device)
                        seq_b=seq_b.to(args.device)
                    # for_weak  = model.log2feats(seq_b)
                    
                    logits = model(idx,seq_a)

                    optimizer_list[0].zero_grad()
                    # if phase == 'train':
                    loss = bce_criterion(logits, (target_a-1))
                    # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                    loss.backward()
                    optimizer.step() 
                    if len(optimizer_list) == 2:
                        lr_scheduler.step()    
            # if phase == 'val' and ((epoch+1 )% 20 == 0) :
            if phase == 'val' :
                with torch.no_grad():
                    r3_b=0
                    m3_b=0
                    r5_b = 0
                    m5_b = 0
                    r10_b = 0
                    m10_b = 0
                    r20_b = 0
                    m20_b = 0
                    for idx,((seq_a, target_a),(seq_b,target_b)) in enumerate(zip(seq_dataloader_test_A,data_loader_qianyi_test)):
                        if torch.cuda.is_available():
                            seq_a=seq_a.to(args.device)
                            target_a=target_a.to(args.device) 
                            seq_b=seq_b.to(args.device) 
                     
                        # for_weak  = model.log2feats(seq_b)
                        logits = model(idx,seq_a)

                        recall,mrr = get_eval(logits, target_a, [3,5,10,20])
                        r3_b += recall[0]
                        m3_b += mrr[0]
                        r5_b += recall[1]
                        m5_b += mrr[1]
                        r10_b += recall[2]
                        m10_b += mrr[2]
                        r20_b += recall[3]
                        m20_b += mrr[3]
                    # if (r10_b+r20_b)/2>best_cirtion :
                    #     best_cirtion=(r10_b+r20_b)/2
                    #     torch.save(model.state_dict(),'model/{},{},{},qianyi={},model_best.pth'.format(args.alpha,args.Beta,args.Gamma,str(args.qianyi)))
                    print('Recall3_b: {:.5f}; Mrr3: {:.5f}'.format(r3_b/len_,m3_b/len_))
                    print('Recall5_b: {:.5f}; Mrr5: {:.5f}'.format(r5_b/len_,m5_b/len_))
                    print('Recall10_b: {:.5f}; Mrr10: {:.5f}'.format(r10_b/len_,m10_b/len_))
                    print('Recall20_b: {:.5f}; Mrr20: {:.5f}'.format(r20_b/len_,m20_b/len_))
                # if (epoch+1 ) == 20:
                #     torch.save(model.state_dict(),'model/SAS.pth')
    torch.save(model.state_dict(),'model/SAS_all2.pth')               


    # return best_model_wts
if __name__=='__main__':
    # seed = 608
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    # parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--A_size', default=3389, type=int)
    parser.add_argument('--B_size', default=16431, type=int)
    parser.add_argument('--all_size', default=(16431+3389), type=int)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--optimizer_all', default=True, type=bool)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--maxlen', default=15, type=int)  #50
    parser.add_argument('--min_len', default=0, type=int)
    parser.add_argument('--lr_decline', default=False, type=bool)


    parser.add_argument('--d_k', default=100, type=int)
    parser.add_argument('--n_heads', default=1, type=int)
    parser.add_argument('--d_v', default=100, type=int)
    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--n_layers', default=1, type=int)


    parser.add_argument('--hidden_units', default=100, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=201, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--state_dict_path', default=None, type=str)
    args = parser.parse_args()
    dataset = TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak.txt','finalcontruth_info/B_strong.txt',args,domain='all',offsets=args.A_size)
    # usernum=dataset.usernum
    usernum=None

    dataset_test = TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak_test.txt','finalcontruth_info/B_strong_test.txt',args,domain='A',offsets=args.A_size)

    bce_criterion = torch.nn.CrossEntropyLoss().to(args.device)

    data_loader_train_A = DataLoader(dataset,batch_size=128,shuffle=True)
    data_loader_test_A = DataLoader(dataset_test,batch_size=128,shuffle=True)

    # from model_cross import SASRec
    # model= SASRec(usernum,args.A_size,args).to(args.device)
    # from model_cross import SequenceExcetor_E
    # model = SequenceExcetor_E(args).to(args.device)

    # from transformer_sas import SASRec
    # from transformer import SASRec
    from transformer2 import SASRec
    # from SASRec import SASRec
    # model= SASRec(args,args.A_size).to(args.device)
    model = SASRec(args, args.all_size).to(args.device)


    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers


    # for name, param in model.named_parameters():
    #         if "prompt" not in name:
    #             param.requires_grad_(False)

    # path='/home/LiuHao/SAS-prompt/model/SAS.pth'
    # model.load_state_dict(torch.load(path,map_location=torch.device(args.device)),strict=False)
    optimizer_list=[]
    lr = 0.001
    # # optimizer = optim.Adam(list(model.prompt_common.parameters())+list(model.prompt_user.parameters()), lr=lr,betas=(0.9, 0.98))  #betas=(0.9, 0.98)
    if args.lr_decline == False :
        optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.98))
        optimizer_list.append(optimizer)
    else:
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters()],
                    "weight_decay": 0.01,
                }
            ]
        optimizer = AdamW(optimizer_grouped_parameters,
                    lr=1e-3, eps=1e-6)
        t_total = (len(dataset) // args.batch_size + 1) * args.epoch
        warmup_ratio = 0.1
        warmup_iters = int(t_total * warmup_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, warmup_iters, t_total)
        optimizer_list.append(optimizer)
        optimizer_list.append(lr_scheduler)
    
    train_model_all(model,data_loader_train_A,data_loader_test_A,optimizer_list,bce_criterion,args.epoch,args,len(dataset_test))




    path='model/SAS_all.pth'
    model.load_state_dict(torch.load(path,map_location=torch.device(args.device)),strict=False)
    # # for name, param in model.named_parameters():
    # #         print(name)
    # #         if "prompt" not in name:
    # #             param.requires_grad_(False)

    # model.Freeze()
    dataset_a_train=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak.txt','finalcontruth_info/B_strong.txt',args,domain='A',offsets=args.A_size)

    dataset_qianyi_train=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak.txt','finalcontruth_info/B_strong.txt',args,domain='B',offsets=args.A_size)    

    dataset_qianyi_test=TVdatasets_all('finalcontruth_info/Elist.txt','finalcontruth_info/Vlist.txt','finalcontruth_info/A_weak_test.txt','finalcontruth_info/B_strong_test.txt',args,domain='B',offsets=args.A_size)

    bce_criterion = torch.nn.CrossEntropyLoss().to(args.device)

    data_loader_train_A = DataLoader(dataset_a_train,batch_size=128,shuffle=True)

    data_loader_qianyi_train = DataLoader(dataset_qianyi_train,batch_size=128,shuffle=True)
    data_loader_qianyi_test = DataLoader(dataset_qianyi_test,batch_size=128,shuffle=True)

    optimizer_list=[optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.98))]

    train_model_one(model,data_loader_train_A,data_loader_qianyi_train,data_loader_test_A,data_loader_qianyi_test,optimizer_list,bce_criterion,30,args,len(dataset_test))



