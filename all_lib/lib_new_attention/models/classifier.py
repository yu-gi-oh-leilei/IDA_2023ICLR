from turtle import forward
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import copy

class Interventional_Classifier(nn.Module):
    def __init__(self, num_classes=80, feat_dim=2048, num_head=4, beta=0.03125, heavy=True, *args):
        super(Interventional_Classifier, self).__init__()
        self.norm_scale = beta       # 1.0 / 32.0      
        self.num_head = num_head
        self.num_classes = num_classes
        self.head_dim = feat_dim // num_head
        self.heavy = heavy
        self.head = nn.ModuleList(nn.Conv2d(self.head_dim, num_classes, 1, bias=False) for i in range(num_head))
        num_layers = 2 if heavy else 1
        self.att_layers = attention_layers(attention_layer(self.head_dim, ffn=heavy), num_layers)
        if heavy:
            print('use heavy')
            self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        self.feat_dim = feat_dim
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x):
        x_list = torch.split(x, self.head_dim, dim=1)
        y_list = []
        for i in range(self.num_head):
            
            normed_x  = x_list[i]
            
            y_ =  self.head[i](normed_x) / (torch.norm(self.head[i].weight, dim=1, keepdim=True).transpose(0,1) + self.norm_scale)   
            score = y_.flatten(2)
            score_soft = self.softmax(score)
            flat_x = normed_x.flatten(2) 
            flat_x  = flat_x.transpose(1, 2) # (B,49,512)
            class_feat = torch.matmul(score_soft, flat_x).unsqueeze(-1) #(B,80,512,1)
            y_list.append(class_feat)
        y = torch.cat(y_list,dim=-1) #(B,80,512,4)
        
        bsz = y.size(0)
        y_sequence = y.contiguous().view(bsz * self.num_classes, self.num_head, self.head_dim)
        attn_output = self.att_layers(y_sequence)
        attn_output = attn_output.contiguous().view(bsz, self.num_classes, self.head_dim * self.num_head)
        if self.heavy:
            attn_output = self.out_proj(attn_output)
        logit = torch.sum(attn_output * torch.cat([head.weight.squeeze() / (torch.norm(self.head[i].weight.squeeze(), dim=1, keepdim=True) + self.norm_scale)  for head in self.head], dim=-1), dim=2)
        #logit = torch.sum(attn_output * torch.cat([head.weight.squeeze() for head in self.head], dim=-1), dim=2)
        #return (attn_output_weights, attn_output_weights_do), logit
        return logit

class attention(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        bsz, tgt_len, _ = query.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling

        q = q.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
        v = v.contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2)) #(B*80*2, 4, 4)
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights_do = F.dropout(attn_output_weights, p=0., training=self.training)
        attn_output = torch.bmm(attn_output_weights_do, v) #(B*80*2, 4, 512)
        attn_output = attn_output.contiguous().view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class attention_layer(nn.Module):
    def __init__(self, embed_dim, num_heads=1, ffn = True, dim_feedforward = 2048):
        super(attention_layer, self).__init__()
        self.att = attention(embed_dim, num_heads)
        self.ffn = ffn
        if ffn:
            
            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, embed_dim)
            #self.dropout1 = nn.Dropout(0.1)
            #self.dropout2 = nn.Dropout(0.1)
            #self.dropout3 = nn.Dropout(0.1)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.activation = F.relu

    def forward(self, src):
        src2 = self.att(src, src, src)
        src = src + src2  # 残差连接
        
        if self.ffn: 
            src = self.norm1(src)
            src2 = self.activation(self.linear1(src))  # [src_len,batch_size,dim_feedforward]
            src2 = self.linear2(src2)                  # [src_len,batch_size,num_heads*kdim]
            #src2 = self.proj(src)
            src = src + src2
            src = self.norm2(src)
        return src  

class attention_layers(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(attention_layers, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)]) 
        self.num_layers = num_layers
    
    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)  
        return output  



class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())

