import numpy as np
import torch
from torch import nn
import math
from torch.nn import functional as F
import copy

#王春晓建模

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
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
# class TextEncoder(nn.Module):  # 文本编码
#     def __init__(self,args):
#         super().__init__()
#         self.dropout_rate =0 #0.2  # 可以传过来，传过来就是与cop中的dropout一样
#         self.hidden_units = args.hidden_units
#         self.num_heads =1  #
#         self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
#         self.attention_layers = torch.nn.ModuleList()#SAS_model.attention_layers
#         self.attention_layernorms =torch.nn.ModuleList() # SAS_model.attention_layernorms
#         self.forward_layernorms = torch.nn.ModuleList() #SAS_model.forward_layernorms
#         self.forward_layers = torch.nn.ModuleList() #SAS_model.forward_layers
#         self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8) # SAS_model.last_layernorm
#         self.dev = args.device
#         self.num_blocks = 2 #变成2更好一些
#         for _ in range(self.num_blocks):  # 循环两遍
#             new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
#             self.attention_layernorms.append(new_attn_layernorm)

#             new_attn_layer = torch.nn.MultiheadAttention(self.hidden_units,  # 多头的将总的hidden_unit除成num_heads份
#                                                             self.num_heads,
#                                                             self.dropout_rate,batch_first=True)
#             self.attention_layers.append(new_attn_layer)

#             new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
#             self.forward_layernorms.append(new_fwd_layernorm)

#             new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
#             self.forward_layers.append(new_fwd_layer)

#     def forward(self, prompts):  # prompt=(4000,77,512) tokenzied_prompt(4000,77)
#         x = prompts
#         x *= prompts.shape[3] ** 0.5
#         x = self.emb_dropout(x)  # (32,77,512)
#         tl = x.shape[2]  # tl=77 time dim len for enforce causality强化因果关系的时间维度

#         for i in range(len(self.attention_layers)):  # range(0,2),这里是多头注意力，只有两层。可以像常规Python列表一样索引，但是它包含的模块已正确注册，所有人都可以看到
#             # x = torch.transpose(x, 0, 1)  # 在seqs的（0，1）维进行转置
#             x = x.float()  # 20220916加
#             Q = self.attention_layernorms[i](x)  # 注意力前layernorm，Q是最终的x
#             mha_outputs, _ = self.attention_layers[i](Q, x, x) #, attn_mask=attention_mask)  # 注意力网络，多头注意力输出
#             x = Q + mha_outputs  # Q+output(Q)：浅绿色x+浅粉 色z
#             # x = torch.transpose(x, 0, 1)  # （32，77，512）12132,61,50
#             x = self.forward_layernorms[i](x)  # 将残差后的数值送入Laynorm，得到整个句子经过多头注意力的特征
#             x = self.forward_layers[i](x)  # 前馈神经网络

#         log_feats = self.last_layernorm(x)  # (U, T, C) -> (U, -1, C) 输入一个张量，得到(32,77,512)

#         return log_feats


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
   
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # [batch_size, 1, len_k], False is masked
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
    def __init__(self,dk):
        super(ScaledDotProductAttention, self).__init__()
        self.dk=dk
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self,args,d_model,d_k,n_heads,d_v):
        super(MultiHeadAttention, self).__init__()
        self.args=args
        self.d_model=d_model
        self.dk=d_k
        self.n_heads=n_heads
        self.dv=d_v
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
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.dk).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.dv).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.dk)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,(self.n_heads) * (self.dv)) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,args,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args=args
        self.d_model=d_model
        self.d_ff=d_ff
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
        return nn.LayerNorm(self.d_model).to(self.args.device)(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):

    def __init__(self,args,d_model,d_k,n_heads,d_v,d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args,d_model,d_k,n_heads,d_v)
        self.pos_ffn = PoswiseFeedForwardNet(args,d_model,d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
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
            weight = F.softmax(out, dim=2) #F.softmax(out.view(1, -1), dim=1)
        return weight  # 得到attention weight


class PromptLearner(nn.Module):
    def __init__(self, args,item_num):
        super().__init__()
        self.args=args
        emb_num = 2   
        emb_num_S =2  
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        # self.pos_emb = PositionalEncoding(args.hidden_units)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE

        self.context_embedding_E = nn.Embedding(emb_num, args.hidden_units)
        # self.context_embedding_V = nn.Embedding(emb_num, args.hidden_units)
        self.context_embedding_s_E = nn.Embedding(emb_num_S, args.hidden_units) #share
        # self.context_embedding_s_V = self.context_embedding_s_E # nn.Embedding(emb_num_S, cfg.TRANSFORMER.HIDDEN_UNIT)  # share
        drop_out = 0.2 
        # self.text_encoder_E = TextEncoder(args)  
        self.attention_E = AttentionLayer(2 * args.hidden_units, drop_out)
        # self.attention_E = AttentionLayer(args.hidden_units, drop_out)
        # self.attention_E = AttentionLayer(args)

        # self.attention_V = AttentionLayer(2 * args.hidden_units, drop_out)
  
        # self.text_encoder_E = TextEncoder(self.args)
        # self.text_encoder_V = TextEncoder(self.args)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight)
        #     if isinstance(m, nn.Embedding):
        #         nn.init.xavier_normal_(m.weight)
        #################################################
        embedding_E = self.context_embedding_E(torch.LongTensor(list(range(emb_num))))
        # embedding_V = self.context_embedding_V(torch.LongTensor(list(range(emb_num))))
        embedding_S_E = self.context_embedding_s_E(torch.LongTensor(list(range(emb_num_S))))
        # embedding_S_V = self.context_embedding_s_V(torch.LongTensor(list(range(emb_num_S))))

        ctx_vectors_E = embedding_E
        # ctx_vectors_V = embedding_V
        ctx_vectors_S_E = embedding_S_E
        # ctx_vectors_S_V = embedding_S_V

 

        #####所有的item将会共享self.ctx,根据不同的item数量调整不同的前缀和后缀个数
        #a photo of的向量表示,requires_grad变成true
        self.ctx_E = nn.Parameter(ctx_vectors_E)  # 4*50  to be optimized，被所有的item共享 待优化 参数层将ctx_vectors放进去，通过parameter将词向量带入进模型中，将此变量成为模型的一部分，根据训练就可以改动，想让这些参数在学习的过程中达到最优
        # self.ctx_V = nn.Parameter(ctx_vectors_V)
        self.ctx_S_E = nn.Parameter(ctx_vectors_S_E)
        # self.ctx_S_V = nn.Parameter(ctx_vectors_S_V)
        # with torch.no_grad():  # 不是参数，token_embedding是一样的，根据prompts E取出来
        #     embedding_class_E = self.src_emb(torch.LongTensor(list(range(0,args.A_size+1)))) # 不更新的用的预训练的  n_cls_E是加了30之后的类数量
        # with torch.no_grad():
        #     embedding_class_V = sasr_model_E.item_emb(torch.LongTensor(list(range(args.A_size+1, args.A_size+args.B_size)))).unsqueeze(1)
        # self.register_buffer("token_prefix_E", embedding_class_E)  # buffer中的内容不会被更新
        # self.register_buffer("token_prefix_V", embedding_class_V)  # SOS词前缀
        # 在内存中定一个不太被重视的模型参数常量，保证模型保存和加载时可以写入和读出

    def forward(self, seq ):
        # for_e=for_e.unsqueeze(1).unsqueeze(1).expand(-1,seq.shape[1],-1,-1)

        # seq_feat=self.token_prefix_E[seq.type(torch.long)]
        seq_feat=self.src_emb(seq.long())
        # seq_feat *= self.src_emb.embedding_dim ** 0.5
        # seq_feat = self.pos_emb(seq_feat.transpose(0, 1)).transpose(0, 1)
        positions = np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])
        seq_feat += self.pos_emb(torch.LongTensor(positions).to(self.args.device))

        seq_feat = self.emb_dropout(seq_feat)

        ctx_E = self.ctx_E # 30,50这里的ctx已经经过parameter了
        # ctx_V = self.ctx_V
        ctx_S_E = self.ctx_S_E #share
        # ctx_S_V = self.ctx_S_V  # share

        alpha = 0
        beta = 0 
        # ctx_E_1 = ctx_E + alpha * ctx_V
        ctx_E_1 = ctx_E 
        # ctx_V_1 = ctx_V + beta * ctx_E

        if ctx_S_E.dim() == 2:
            ctx_E = ctx_E_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  # 按不同的在ctx的第0维，进行扩充;将内部，第2维复制为n_cls=37倍，0维和1维不变。  12132,30,50
            # ctx_V = ctx_V_1.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1],-1, -1)  # 18388,30,50  等于V本身数量
            ctx_S_E = ctx_S_E.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  # 12132,30,50
            # ctx_S_V = ctx_S_V.unsqueeze(0).unsqueeze(0).expand(seq.shape[0], seq.shape[1] ,-1, -1)  # 18388,30,50


        # ctx_prefix_E = self.getPrompts(prefix_E, ctx_E, ctx_S_E ) # 128 15 8 100

        ctx_prefix_E = self.getPrompts(seq_feat.unsqueeze(2), ctx_E, ctx_S_E ) # 128 15 8 100
        # ctx_prefix_E= self.text_encoder_E(ctx_prefix_E)
        # prompts_E=ctx_prefix_E[:,:,-1,:]
        # ctx_prefix_V = self.getPrompts(prefix_V, ctx_V, ctx_S_V , for_e)

        #先将他们进行线性变换，用可学习的Transformer?,这个位置也可以提前，提交到concat之前
        # x_E = self.text_encoder_E(ctx_prefix_E) # 在这里进入text_encoder  12132,(1+30+30),50
        # x_V = self.text_encoder_V(ctx_prefix_V)

        # ctx_prefix_E = x_E
        # ctx_prefix_V = x_V

        #下面是求和
        item_embedding = seq_feat.unsqueeze(2).expand(-1, -1 ,ctx_prefix_E.shape[2],-1)
        prompt_item = torch.cat((ctx_prefix_E, item_embedding), dim=3)
        at_wt = self.attention_E(prompt_item)
        #把和item concat的结果送给它
        
        # at_wt = self.attention_E(ctx_prefix_E)  # 128 15 8 1  

        # prompts_E = torch.bmm(at_wt.permute(0, 1, 3, 2), ctx_prefix_E).squeeze()  # 对user加权和之后得到的group emb
        prompts_E = torch.matmul(at_wt.permute(0, 1, 3, 2), ctx_prefix_E).squeeze() 


        # prompts_E = self.attention_E(ctx_prefix_E)[:,:,-1,:]

        # prompts_E=ctx_prefix_E.sum(dim=2)

        # item_embedding = prefix_V.expand(-1, ctx_prefix_V.shape[1], -1)
        # prompt_item = torch.cat((ctx_prefix_V, item_embedding), dim=2)
       
        # at_wt = self.attention_V(prompt_item)  # 这里是计算weight, 先将prompt和item concat，然后经过linear计算它们的相似度
        # prompts_V = torch.bmm(at_wt.permute(0, 1 , 3, 2), ctx_prefix_V).squeeze()  # 对user加权和之后得到的group emb

        return prompts_E

    def getPrompts(self, prefix, ctx,ctx_S): #ctx_S, suffix=None):#
    
        prompts = torch.cat(
            [
                ctx_S, 
                ctx,  
                prefix 
            ],
            dim=2,
        )
        return prompts
class SASRec(nn.Module):
    def __init__(self,args,item_num):
        super(SASRec, self).__init__()
        self.args=args
        self.class_=nn.Linear(args.hidden_units,args.all_size)
        # self.src_emb = nn.Embedding(item_num+1, args.hidden_units)
        # self.pos_emb = PositionalEncoding(args.hidden_units)
        self.layers = nn.ModuleList([EncoderLayer(self.args,args.hidden_units,args.d_k,args.n_heads,args.d_v,args.d_ff) for _ in range(args.n_layers)])
        self.prompt=PromptLearner(args,item_num)

    def phase_one(self, user , log_seqs):

        enc_outputs = self.prompt(log_seqs)


        enc_self_attn_mask = get_attn_pad_mask(log_seqs, log_seqs) # [batch_size, src_len, src_len]

        enc_attn_mask=get_attn_subsequence_mask(log_seqs).to(self.args.device)
        all_mask=torch.gt((enc_self_attn_mask + enc_attn_mask), 0).to(self.args.device)
        
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, all_mask)
            enc_self_attns.append(enc_self_attn)
        a=enc_outputs[:,-1,:]

        logits=self.class_(enc_outputs[:,-1,:])
        return logits


    def forward(self,user,log_seqs):
        logits=self.phase_one(user,log_seqs)
        return logits
    
    def Freeze(self):#tune prompt + head
        for param in self.parameters():
            param.requires_grad = False
            
        for name, param in self.named_parameters():
            #print(name)
            if "ctx" in name:
                param.requires_grad = True
            if "class_" in name:
                param.requires_grad = True
        self.prompt.src_emb.requires_grad = True
        # self.prompt.pos_emb.requires_grad = True