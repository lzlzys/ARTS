import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from metrics4rec import mean_average_precision, mean_reciprocal_rank, average_precision, hit_at_k, ndcg_at_k, \
    recall_at_k, precision_at_k

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
# sentences = [
#     # enc_input           dec_input         dec_output
#     ['1 2 3 4', 'S 5 .', '5 . E'],
#     ['1 2 3 4', 'S 5 .', '5 . E']
# ]
#
# # Padding Should be Zero
# src_vocab = {'P': 0, '1': 1, '2': 2, '3': 3, '4': 4}
# src_vocab_size = len(src_vocab)
#
# tgt_vocab = {'P': 0, 'S': 1, '5': 2, 'E': 3, '.': 4}
# idx2word = {i: w for i, w in enumerate(tgt_vocab)}
# tgt_vocab_size = len(tgt_vocab)
#


def evaluate_once(topk_preds, groundtruth):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_id>. length of the list is topK.
        groundtruth: list of <item_id>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth)
    topk = len(topk_preds)
    rel = []
    for iid in topk_preds:
        if iid in gt_set:
            rel.append(1)
        else:
            rel.append(0)
    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "ndcg@k": ndcg_at_k(rel, topk, 1),
        "hit@k": hit_at_k(rel, topk),
        "ap": average_precision(rel),
        "rel": rel,
    }


def evaluate_all(topk=10, item=None):
    """Evaluate all user-items performance.
    Args:
        user_item_scores: dict with key = <item_id>, value = <user_item_score>.
                     Make sure larger score means better recommendation.
        groudtruth: dict with key = <user_id>, value = list of <item_id>.
        topk: int
    Returns:
    """
    avg_prec, avg_recall, avg_ndcg, avg_hit = 0.0, 0.0, 0.0, 0.0
    rs = []
    cnt = 0
    j = 0
    for enc_inputs, dec_inputs, dec_outputs, user_id in loader:
        batch_size = 0
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)

        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        groudtruth = {}
        groudtruth[batch_size] = dec_outputs.tolist()[batch_size]
        outputs = model(enc_inputs[:,:], item, j=j)
        values, indices = torch.topk(outputs, k=5, dim=1)#indices : 128 * 5
        topk_preds1 = indices.tolist()
        topk_preds = topk_preds1[batch_size]
        result = evaluate_once(topk_preds, groudtruth[batch_size])#topk_preds, groudtruth[uid] 为 list,
        avg_prec += result["precision@k"]
        avg_recall += result["recall@k"]
        avg_ndcg += result["ndcg@k"]
        avg_hit += result["hit@k"]
        rs.append(result["rel"])
        cnt += 1
        batch_size += 1
        j += 1
    avg_prec = avg_prec / cnt
    avg_recall = avg_recall / cnt
    avg_ndcg = avg_ndcg / cnt
    avg_hit = avg_hit / cnt
    map_ = mean_average_precision(rs)
    mrr = mean_reciprocal_rank(rs)
    msg = "\nNDCG@{}\tRec@{}\tHits@{}\tPrec@{}\tMAP@{}\tMRR@{}".format(topk, topk, topk, topk, topk, topk)
    msg += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, avg_hit, avg_prec, map_, mrr)
    # msg = "NDCG@{}\tRec@{}\tMAP@{}".format(topk, topk, topk)
    # msg += "\n{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, map)
    print(msg)
    res = {'@':topk,
        'ndcg': avg_ndcg,
        'map': map_,
        'recall': avg_recall,
        'precision': avg_prec,
        'mrr': mrr,
        'hit': avg_hit,
    }
    return msg, res




src_len = 99  # enc_input max sequence length
tgt_len = 1  # dec_input(=dec_output) max sequence length

def make_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    max_num = 0
    # for i in range(1285):
    #
    #     ini_dec_input = []
    #     ini_dec_output = []
    #     ini_input = torch.load('/home/t618141/python_code/new_lzl_model/input/{}_input.pt'.format(i))
    #
    #     # if sum_0 != 0:
    #     #     ini_input.insert(sum_0, i + 5000)#给予对应cls_token,加5000是因为防止产生与item相同的对应
    #     #     ini_input = ini_input[:100]
    #     if max_num < max(ini_input):
    #         max_num = max(ini_input)#2940
    #     tgt = ini_input[-1]
    #     enc_input = ini_input[:-1]
    #     ini_dec_input.append(tgt)
    #     ini_dec_output.append(tgt)
    #     enc_inputs.append(enc_input)
    #     dec_inputs.append(ini_dec_input)
    #     dec_outputs.append(ini_dec_output)

    a = np.load('/data/t618141/Toys_and_Games_10_new.npy', allow_pickle=True).item()
    for i, j in a.items():

        ini_dec_input = []
        ini_dec_output = []
        ini_input = j

        length = len(ini_input)

        # 判断列表长度是否小于100
        if length < 100:
            # 计算需要填充的0的个数
            padding = 100 - length
            # 在列表左侧加上padding个0
            ini_input = [0] * padding + ini_input
        else:
            ini_input = ini_input[:100]

        if max_num < max(ini_input):
            max_num = max(ini_input)#2940
        tgt = ini_input[-1]
        enc_input = ini_input[:-1]
        ini_dec_input.append(tgt)
        ini_dec_output.append(tgt)
        enc_inputs.append(enc_input)
        dec_inputs.append(ini_dec_input)
        dec_outputs.append(ini_dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs), max_num


enc_inputs, dec_inputs, dec_outputs, max_num = make_data()
# src_vocab_size = 6285
tgt_vocab_size = max_num
item = [i for i in range(1, tgt_vocab_size + 2)]
item = torch.LongTensor(item)

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx], idx + 1


loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 128, False)
d_model = 768  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 12  # number of heads in Multi-Head Attention


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
        return nn.LayerNorm(d_model).cuda(2)(output + residual), attn


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
        return nn.LayerNorm(d_model).cuda(2)(output + residual)  # [batch_size, seq_len, d_model]


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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(tgt_vocab_size + 1, d_model)
        #self.src_emb = nn.Embedding(6285, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.class_ = nn.Linear(768, tgt_vocab_size)
        self.item_embedding = nn.Embedding(tgt_vocab_size + 2, d_model)
        #self.cls_token = nn.Parameter(self.src_emb(torch.LongTensor(torch.arange(5000, 5000 + 1285))))
    def forward(self, enc_inputs, item, z):
        '''
        enc_inputs: [batch_size, src_len]
        '''

        item = self.item_embedding(item.cuda(2))
        # a = None
        # for h in enc_inputs:
        #     length = 0
        #     for j in h:
        #
        #         if j.item() == 0:
        #
        #             length += 1
        #         else:
        #             break
        #
        #     if length != 0:
        #         b = torch.rand(1, enc_inputs.size(1) - length + 1).cuda()
        #         c = torch.zeros(1, length - 1).cuda()
        #         d = torch.cat((c, b), dim=1).cuda()
        #     else:
        #         d = h.unsqueeze(0).cuda()
        #     if a == None:
        #         a = d.cuda()
        #     else:
        #         a = torch.cat((a, d),dim=0).cuda()
        # a = a.cuda(2)
        # mask = torch.eq(enc_inputs, 0)
        # pad_count = torch.sum(mask, dim=1)
        # enc_outputs = self.src_emb(enc_inputs)
        # # [batch_size, src_len, d_model]
        # for i in range(user_id.size(0)):#加cls_token
        #     if pad_count[i] == 0:#如果没有pad，则加上cls_token
        #         left = self.cls_token[i - 1 + (z * user_id.size(0)),:].unsqueeze(0)
        #         right = self.src_emb(torch.LongTensor(enc_inputs[i,:].cpu()).cuda(2))
        #         new_enc_outputs = torch.cat((left, right),dim=0)
        #         enc_outputs[i] = new_enc_outputs[0:-1, :]
        #     else:#如果有pad，则将序列拆成两个序列，一个是pad，一个是实际的交互序列，然后在交互序列的最前面加上cls_token
        #         left = enc_inputs[i][:pad_count[i]]
        #         left = self.src_emb(torch.LongTensor(left.cpu()).cuda(2))
        #         right = enc_inputs[i][pad_count[i]:]
        #         right = self.src_emb(torch.LongTensor(right.cpu()).cuda(2))
        #         # w = torch.LongTensor(self.cls_token[int(user_id[i].item()) -1].cpu().long().unsqueeze(0))
        #         # a = self.src_emb(w.cuda(2))
        #         new_enc_outputs = torch.cat((left, self.cls_token[i - 1 + (z * user_id.size(0)),:].unsqueeze(0), right), dim=0)
        #         enc_outputs[i] = new_enc_outputs[0:-1, :]
        # enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        #   # [batch_size, src_len, d_model]
        # enc_self_attn_mask = get_attn_pad_mask(a, a)  # [batch_size, src_len, src_len]
        # enc_self_attns = []
        # for layer in self.layers:
        #     # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
        #     enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
        #     enc_self_attns.append(enc_self_attn)
        # enc_outputs = enc_outputs[:,-1, :]
        # output = torch.matmul(enc_outputs.cpu(), np.transpose(item.detach().cpu(), (1, 0)))
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        #enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        dec_self_attn_pad_mask = get_attn_pad_mask(enc_inputs, enc_inputs).cuda(2)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(enc_inputs).cuda(2)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda(
            2)  # [batch_size, tgt_len, tgt_len]

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, dec_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        enc_outputs = enc_outputs[:, -1, :]
        #output = torch.matmul(enc_outputs.cuda(2), torch.transpose(item.cuda(2), 1, 0))
        # item_embs = self.src_emb(torch.LongTensor(range(1, 2940 + 1)).cuda(2))# (U, I, C)
        logits = self.class_(enc_outputs)
        # logits = item_embs.matmul(enc_outputs.unsqueeze(-1)).squeeze(-1)
        return logits



class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        '''
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda(
            2)  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda(2)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda(2)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda(
            2)  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns



class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda(2)
        self.decoder = Decoder().cuda(2)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda(2)

    def forward(self, enc_inputs, item, j):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''

        # tensor to store decoder outputs

        enc_outputs = self.encoder(enc_inputs, item, j)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        # dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        # return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
        return enc_outputs.cuda(2)


device = torch.device('cuda:2')
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.99)

for epoch in range(30):
    j = 0
    loss1 = 0
    h = 0
    for enc_inputs, dec_inputs, dec_outputs, user_id in loader:
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        h += 1
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)

        # outputs: [batch_size * tgt_len, tgt_vocab_size]

        # outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs, j)
        outputs = model(enc_inputs, item, j)
        # loss = criterion(outputs, dec_outputs.view(-1))
        loss = criterion(outputs, dec_outputs.view(-1))
        loss1 += loss
        # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        # 获取模型的所有参数
        params = dict(model.named_parameters())
        optimizer.zero_grad()
        loss.backward()
        # print('epoch{}'.format(epoch))
        # print(params)
        optimizer.step()
        j += 1
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss1/h))
    evaluate_all(topk=5, item=item)


