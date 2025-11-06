import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import math
from torch_geometric.nn import GCNConv, RGCNConv, TransformerConv
import pandas as pd
import pickle
from model.egnn_clean_deng import EGNN

class DDI_GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, num_layers):
        super().__init__()
        self.layer = num_layers
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads, dropout=0.1)
        self.conv2 = nn.ModuleList(
            [TransformerConv(hidden_channels * heads, hidden_channels, heads, dropout=0.1) for _ in
             range(2, num_layers)])

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index,edge_attr).relu()
        if self.layer > 1:
            for conv in self.conv2:
                x = conv(x, edge_index,edge_attr)  #
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):

        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output, attn


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output, attn = self.attn(X)
        X = self.AN1(output + X)

        output = self.l1(X)
        X = self.AN2(output + X)

        return X, attn


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AE1(torch.nn.Module):  # Joining together
    def __init__(self, vector_size, len_after_AE=128, bert_n_heads=4):
        super(AE1, self).__init__()

        self.vector_size = vector_size
        self.att2 = EncoderLayer(self.vector_size, bert_n_heads)

        self.l1 = torch.nn.Linear(self.vector_size, self.vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(self.vector_size // 2)

        # self.att2 = EncoderLayer((self.vector_size + len_after_AE) // 2, bert_n_heads)
        self.l2 = torch.nn.Linear(self.vector_size // 2, len_after_AE)

        self.l3 = torch.nn.Linear(len_after_AE, self.vector_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.vector_size // 2)

        self.l4 = torch.nn.Linear(self.vector_size // 2, self.vector_size)

        self.dr = torch.nn.Dropout(0.1)
        self.ac = gelu  # nn.ReLU()#

    def forward(self, X):
        X, attn = self.att2(X)
        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.l2(X)

        X_AE = self.dr(self.bn3(self.ac(self.l3(X))))

        X_AE = self.l4(X_AE)

        return X, attn

def to_edge_index(ei):
    ei = torch.as_tensor(ei)
    if ei.ndim != 2:
        raise ValueError(f"edges ndim 应为2，got {ei.ndim} with shape {ei.shape}")
    if ei.shape[0] == 2:
        return ei.long()                          # (2, E)
    if ei.shape[1] == 2:
        return ei.t().contiguous().long()         # (E,2) -> (2,E)
    raise ValueError(f"edges 形状需为(2,E)或(E,2)，got {ei.shape}")


def collate_graphs(batch_graphs, device="cpu", target_edge_feat_dim=None):
    X_list, X3_list, EI_list, EA_list, BATCH_list = [], [], [], [], []
    node_base = 0

    # 先遍历，规范 edge_index，并把 edge_attr 统一成 (E, Fe) 或 None
    for gi, g in enumerate(batch_graphs):
        h  = torch.as_tensor(g["h"], dtype=torch.float)     # [Ni, Fh]
        x  = torch.as_tensor(g["x"], dtype=torch.float)     # [Ni, 3]
        ei = to_edge_index(g["edges"])                      # [2, Ei]
        E  = ei.shape[1]

        # --- 规范 edge_attr 到 (E, Fe) 或 None ---
        ea_raw = g.get("edge_attr", None)
        if ea_raw is None:
            ea = None
        else:
            ea = torch.as_tensor(ea_raw, dtype=torch.float)
            if ea.ndim == 1:
                ea = ea.unsqueeze(-1)                       # (E,) -> (E,1)
            # 可能被存成 (Fe, E)，需要转置
            if ea.shape[0] != E and ea.shape[1] == E:
                ea = ea.t().contiguous()                    # -> (E, Fe)
            # 严格检查
            if ea.shape[0] != E:
                raise ValueError(
                    f"edge_attr 与 edges 不匹配: edges E={E}, edge_attr shape={ea.shape}"
                )

        Ni = h.size(0)
        X_list.append(h)
        X3_list.append(x)
        EI_list.append(ei + node_base)
        EA_list.append(ea)                                  # 先收集，后面统一维度
        BATCH_list.append(torch.full((Ni,), gi, dtype=torch.long))

        node_base += Ni

    # ---- 统一 edge_attr 维度 ----
    # 1) 决定最终的 Fe
    Fe = 0
    if target_edge_feat_dim is not None:
        Fe = int(target_edge_feat_dim)
    else:
        for ea in EA_list:
            if ea is not None:
                Fe = int(ea.shape[1])
                break
    # 2) 按 Fe 对齐（缺失补零，过大裁剪）
    EA_fixed = []
    for ei, ea in zip(EI_list, EA_list):
        E = ei.shape[1]
        if Fe == 0:
            EA_fixed.append(None)
        else:
            if ea is None:
                EA_fixed.append(torch.zeros(E, Fe, dtype=torch.float))
            else:
                if ea.shape[1] < Fe:
                    pad = torch.zeros(ea.shape[0], Fe - ea.shape[1], dtype=ea.dtype)
                    EA_fixed.append(torch.cat([ea, pad], dim=1))
                else:
                    EA_fixed.append(ea[:, :Fe])

    # ---- 拼接 ----
    H   = torch.cat(X_list,  dim=0).to(device)              # (ΣN, Fh)
    X   = torch.cat(X3_list, dim=0).to(device)              # (ΣN, 3)
    EI  = torch.cat(EI_list, dim=1).to(device)              # (2, ΣE)
    BTH = torch.cat(BATCH_list, dim=0).to(device)           # (ΣN,)

    if Fe == 0:
        EA = None
    else:
        EA = torch.cat(EA_fixed, dim=0).to(device)          # (ΣE, Fe)

    return H, X, EI, EA, BTH

class EGNNEncoder(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf=64, out_node_nf=64, n_layers=3, pooling='mean'):
        super().__init__()
        self.layers = nn.ModuleList([
            EGNN(
                in_node_nf = in_node_nf if i==0 else out_node_nf,
                hidden_nf  = hidden_nf,
                out_node_nf= out_node_nf,
                in_edge_nf = in_edge_nf
            ) for i in range(n_layers)
        ])
        self.pooling = pooling

    def forward(self, h, x, edges, edge_attr):
        # h: (N,F_h), x: (N,3), edges: (2,E), edge_attr: (E,F_e)
        for layer in self.layers:
            h, x = layer(h, x, edges, edge_attr)  # 保持与 egnn_clean 接口一致
        # readout
        if self.pooling == 'mean':
            mol_emb = h.mean(dim=0, keepdim=True)   # 简单单分子示例
        elif self.pooling == 'sum':
            mol_emb = h.sum(dim=0, keepdim=True)
        else:  # max
            mol_emb, _ = h.max(dim=0, keepdim=True)
        return mol_emb, h, x  # 分子向量、节点向量、更新后的坐标

def mean_pool_nodes(node_feats, batch, num_graphs=None):
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1

    out = torch.zeros(num_graphs, node_feats.size(1), device=node_feats.device)
    out.index_add_(0, batch, node_feats)  # 按图求和
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1).unsqueeze(1)
    out = out / counts                    # 求平均
    return out

class model_S0(torch.nn.Module):
    def __init__(self, device, index, dim, l, head):  # in_dim,hidden_dim,out_dim,
        super(model_S0, self).__init__()
        self.device = device
        self.dim = dim
        self.l = l
        self.head = head

        self.trans2 = DDI_GraphTransformer(128, self.dim, self.head, self.l)

        self.mlp = nn.ModuleList([nn.Linear(3072, 3072),  # 3072,2560
                                  nn.ELU(),
                                  nn.Linear(3072, 65)
                                  ])

        morgen_path = '../data/drugfeatures/morgen_fingerprint_DDIMDL.npy'
        np_morgen = np.load(morgen_path)
        self.morgen = torch.tensor(np_morgen).to(device).float()

        alignn_path = '../data/drugfeatures/molformer_DDIMDL.npy'  #
        alignn = np.load(alignn_path)
        self.alignn = torch.tensor(alignn).to(device).float()

        with open("../data/drugfeatures/DDIMDL_drug_3dproperty_569.pkl", "rb") as f:
            self.all_graphs = pickle.load(f)
        g0 = self.all_graphs[0]
        sample_h = torch.as_tensor(g0["h"], dtype=torch.float)
        in_node_nf = sample_h.size(1)
        if g0.get("edge_attr", None) is None:
            in_edge_nf = 0
        else:
            sample_ea = torch.as_tensor(g0["edge_attr"], dtype=torch.float)
            if sample_ea.ndim == 1:
                sample_ea = sample_ea.unsqueeze(-1)
            # 若存成(Fe,E)就转为(E,Fe)再取维度
            if sample_ea.shape[0] != torch.as_tensor(g0["edges"]).shape[0] and sample_ea.shape[1] == \
                    torch.as_tensor(g0["edges"]).shape[0]:
                sample_ea = sample_ea.t().contiguous()
            in_edge_nf = sample_ea.size(1)

        self.in_node_nf = in_node_nf
        self.in_edge_nf = in_edge_nf

        self.egnn = EGNN(
            in_node_nf=self.in_node_nf,
            in_edge_nf=self.in_edge_nf,
            hidden_nf=128,
            out_node_nf=256,
            n_layers=1,
            device=self.device
        )
        self.ae1 = AE1(1024, self.dim* head , self.head)#

        self.cl_proj_dim = 128  # 投影维度，可调 64/128/256
        self.cl_temp = 0.07 # 温度，可调 0.05–0.2

        self.proj_morgen = nn.Sequential(
            nn.LazyLinear(self.cl_proj_dim), nn.ReLU(inplace=True),
            nn.Linear(self.cl_proj_dim, self.cl_proj_dim)
        )
        self.proj_graph = nn.Sequential(
            nn.LazyLinear(self.cl_proj_dim), nn.ReLU(inplace=True),
            nn.Linear(self.cl_proj_dim, self.cl_proj_dim)
        )
        self.proj_3d = nn.Sequential(
            nn.LazyLinear(self.cl_proj_dim), nn.ReLU(inplace=True),
            nn.Linear(self.cl_proj_dim, self.cl_proj_dim)
        )
    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
        """
        标准对称 InfoNCE（NT-Xent）：z1 与 z2 为一一对应的正样本，其它为负样本。
        z*: [B, D]；返回标量 loss。
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = z1 @ z2.t() / temp  # [B, B]
        labels = torch.arange(z1.size(0), device=z1.device)
        loss12 = F.cross_entropy(logits, labels)
        loss21 = F.cross_entropy(logits.t(), labels)
        return 0.5* (loss12 + loss21)

    def MLP(self, vectors, layer):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def fuse_features(self, modal_feat, graph_feat):
        combined = torch.cat([modal_feat, graph_feat], dim=1)  # [batch, 2*feat_dim]
        gate = torch.sigmoid(self.fc_gate(combined))  # [batch, feat_dim]
        fused = gate * modal_feat + (1 - gate) * graph_feat  # [batch, feat_dim]
        return fused

    def forward(self, drug1s, drug2s):
        modalfeatures1, atten = self.ae1(self.morgen)

        drug1_emb_morgen = modalfeatures1[drug1s.long()]
        drug2_emb_morgen = modalfeatures1[drug2s.long()]

        drug1_emb_graph = self.alignn[drug1s.long()]
        drug2_emb_graph = self.alignn[drug2s.long()]

        batch_graphs1 = [self.all_graphs[i] for i in drug1s.long().tolist()]
        h, x, edge_index, edge_attr, batch = collate_graphs(
            batch_graphs1, device=self.device,
            # 确保边特征维度与初始化一致；若 self.in_edge_nf==0，collate 返回 None
            target_edge_feat_dim=(self.in_edge_nf if self.in_edge_nf > 0 else None)
        )
        h1, x1 = self.egnn(h, x, edge_index, edge_attr)
        drug1_3d_emb = mean_pool_nodes(h1, batch, num_graphs=drug1s.size(0))

        batch_graphs2 = [self.all_graphs[i] for i in drug2s.long().tolist()]
        h2, x2, edge_index2, edge_attr2, batch2 = collate_graphs(
            batch_graphs2, device=self.device,
            # 确保边特征维度与初始化一致；若 self.in_edge_nf==0，collate 返回 None
            target_edge_feat_dim=(self.in_edge_nf if self.in_edge_nf > 0 else None)
        )
        h2, x2 = self.egnn(h2, x2, edge_index2, edge_attr2)
        drug2_3d_emb = mean_pool_nodes(h2, batch2, num_graphs=drug2s.size(0))

        all = torch.cat((drug1_emb_morgen,drug1_emb_graph,drug1_3d_emb, drug2_emb_morgen,drug2_emb_graph,drug2_3d_emb), 1)  #
        #print(all.shape)
        logits = self.MLP(all, 3)
        z1_m = self.proj_morgen(drug1_emb_morgen)  # [B, P]
        z1_g = self.proj_graph(drug1_emb_graph)  # [B, P]
        z1_3 = self.proj_3d(drug1_3d_emb)  # [B, P]

        z2_m = self.proj_morgen(drug2_emb_morgen)  # [B, P]
        z2_g = self.proj_graph(drug2_emb_graph)  # [B, P]
        z2_3 = self.proj_3d(drug2_3d_emb) # [B, P]

        cl_loss_d1 = (
                             self._info_nce(z1_m, z1_g, self.cl_temp) +
                             self._info_nce(z1_m, z1_3, self.cl_temp) +
                             self._info_nce(z1_g, z1_3, self.cl_temp)
                     ) / 3.0

        cl_loss_d2 = (
                             self._info_nce(z2_m, z2_g, self.cl_temp) +
                             self._info_nce(z2_m, z2_3, self.cl_temp) +
                             self._info_nce(z2_g, z2_3, self.cl_temp)
                     )/ 3.0

        #cl_loss_d1 = self._info_nce(z1_g, z1_3, self.cl_temp)
        #cl_loss_d2 =  self._info_nce(z2_g, z2_3, self.cl_temp)
        cl_loss = 0.5* (cl_loss_d1 + cl_loss_d2)

        return logits, cl_loss
