import numpy as np
import torch
from torch import nn
import math
from torch.nn import functional as F
import copy


# 王春晓建模

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


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

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = dk

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args, d_model, d_k, n_heads, d_v):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.d_model = d_model
        self.dk = d_k
        self.n_heads = n_heads
        self.dv = d_v
        self.W_Q = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.dk * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.dv * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dv, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dk).transpose(1,
                                                                                    2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dk).transpose(1,
                                                                                    2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dv).transpose(1,
                                                                                    2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dk)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, (self.n_heads) * (
            self.dv))  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):

    def __init__(self, args, d_model, d_k, n_heads, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args, d_model, d_k, n_heads, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(args, d_model, d_ff)

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


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x, mask=None):
        # out = self.linear(x)
        # weight = F.softmax(out.view(1, -1), dim=1)
        # return weight
        out = self.linear(x)
        if mask is not None:  # 这是针对item aggregation中的attention部分
            out = out.masked_fill(mask, -100000)  # 在mask值为1的位置处用value填充-np.inf
            weight = F.softmax(out, dim=1)
            return weight
        else:
            weight = F.softmax(out, dim=2)  # F.softmax(out.view(1, -1), dim=1)
        return weight  # 得到attention weight


class SASRec(nn.Module):
    def __init__(self, args, item_num):
        super(SASRec, self).__init__()
        self.args = args
        print(args.all_size)


        self.class_= nn.Linear(args.hidden_units, args.all_size)
        self.src_emb = nn.Embedding(args.item_num + 1, args.hidden_units)#item_num

        self.pos_emb = PositionalEncoding(args.hidden_units)
        self.layers = nn.ModuleList([EncoderLayer(self.args,args.hidden_units,args.d_k,args.n_heads,args.d_v,args.d_ff) for _ in range(args.n_layers)])
        # self.cls_token = nn.Parameter(self.user_cls_token_src_emb(torch.LongTensor(torch.arange(0, args.user_num + 1))))#user_num
        self.cls_token = nn.Parameter(torch.randn(1, args.hidden_units))
        # self.cls_token = nn.Parameter(torch.load('/home/t618141/python_code/test_model/model/{}_cls_token.pth'.format(args.args_str)))

    def phase_one(self, log_seqs, z, user_id_index):
        prompt_index = []
        enc_inputs = log_seqs
        a = None
        for h in enc_inputs:
            length = 0
            for j in h:
                if j.item() == 0:
                    length += 1
                else:
                    break
            if length != 0:
                b = torch.rand(1, enc_inputs.size(1) - length + 1).cuda()
                c = torch.zeros(1, length - 1).cuda()
                d = torch.cat((c, b), dim=1).cuda()
            else:
                d = h.unsqueeze(0).cuda()
            if a == None:
                a = d.cuda()
            else:
                a = torch.cat((a, d), dim=0).cuda()
            prompt_index.append(length)
        a = a.cuda()
        mask = torch.eq(enc_inputs, 0)
        pad_count = torch.sum(mask, dim=1)
        enc_outputs = self.src_emb(enc_inputs)
        # [batch_size, src_len, d_model]
        for i in range(enc_inputs.size(0)):  # 加cls_token
            if pad_count[i] == 0:  # 如果没有pad，则加上cls_token
                # left = self.cls_token[i + (z * enc_inputs.size(0)), :].unsqueeze(0)
                # left = self.cls_token[user_id_index[i], :].unsqueeze(0)
                left = self.cls_token
                right = self.src_emb(torch.LongTensor(enc_inputs[i, :].cpu().long()).cuda())
                new_enc_outputs = torch.cat((left, right), dim=0)
                enc_outputs[i] = new_enc_outputs[0:-1, :]
            else:  # 如果有pad，则将序列拆成两个序列，一个是pad，一个是实际的交互序列，然后在交互序列的最前面加上cls_token
                left = enc_inputs[i][:pad_count[i]]
                left = self.src_emb(torch.LongTensor(left.cpu().long()).cuda())
                right = enc_inputs[i][pad_count[i]:]
                right = self.src_emb(torch.LongTensor(right.cpu().long()).cuda())
                # w = torch.LongTensor(self.cls_token[int(user_id[i].item()) -1].cpu().long().unsqueeze(0))
                # a = self.src_emb(w.cuda(2))
                # new_enc_outputs = torch.cat((left, self.cls_token[i + (z * enc_inputs.size(0)), :].unsqueeze(0), right), dim=0)
                # new_enc_outputs = torch.cat((left, self.cls_token[user_id_index[i], :].unsqueeze(0), right), dim=0)
                new_enc_outputs = torch.cat((left, self.cls_token, right), dim=0)
                enc_outputs[i] = new_enc_outputs[0:-1, :]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(a, a)  # [batch_size, src_len, src_len]
        enc_attn_mask = get_attn_subsequence_mask(log_seqs).cuda()
        all_mask = torch.gt((enc_self_attn_mask + enc_attn_mask), 0).cuda()
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
            enc_self_attns.append(enc_self_attn)



        return enc_outputs, prompt_index

    def forward(self, log_seqs, z, user_id_index):
        # log_seqs = log_seqs.cuda()
        logits = self.phase_one(log_seqs, z, user_id_index)
        return logits

    def Freeze(self):  # tune prompt + head
        for param in self.parameters():
            param.requires_grad = False

        for name, param in self.named_parameters():
            # print(name)
            if "ctx" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True