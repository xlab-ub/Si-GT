import torch
import os
import dgl
import dgl.nn as dglnn
import torch.optim as optim
import dgl.sparse as dglsp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
import pytorch_lightning as pl
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm
from dgl.nn.pytorch.gt import path_encoder
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder, GINConv, GraphConv, EGATConv
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, GINConv, PairNorm
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)
from dgl.nn import GraphConv, GATConv


class CustomGINLayer(nn.Module):
    def __init__(self, emb_dim, edge_encoder_cls=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_encoder = edge_encoder_cls(emb_dim) if edge_encoder_cls else None

    def message_func(self, edges):
        if self.edge_encoder:
            e = self.edge_encoder(edges.data["edge_attr"])
        else:
            e = edges.data["edge_attr"]  
        return {"m": edges.src["h"] + e}

    def reduce_func(self, nodes):
        return {"aggr": torch.sum(nodes.mailbox["m"], dim=1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.update_all(self.message_func, self.reduce_func)
            h_aggr = g.ndata["aggr"]
            # GIN formula: (1 + eps)*h + MLP(aggregated)
            out = (1.0 + self.eps) * h + self.mlp(h_aggr)
            return out


class CustomLayer(nn.Module):
    """
    Wraps one layer of GIN/GCN/GAT for consistency in the main GNN.
    """
    def __init__(self, emb_dim, gnn_type="gin", edge_encoder_cls=None, num_heads=1):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.num_heads = num_heads

        if self.gnn_type == "gin":
            self.layer = CustomGINLayer(emb_dim, edge_encoder_cls=edge_encoder_cls)
            self.out_dim = emb_dim
        elif self.gnn_type == "gcn":
            self.layer = GraphConv(in_feats=emb_dim, out_feats=emb_dim, norm="none")
            self.out_dim = emb_dim
        elif self.gnn_type == "gat":
            out_feats = emb_dim // num_heads
            self.layer = GATConv(in_feats=emb_dim, 
                                 out_feats=out_feats, 
                                 num_heads=num_heads)
            self.out_dim = emb_dim
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

    def forward(self, g, h):
        out = self.layer(g, h)
        if self.gnn_type == "gat":
            if self.num_heads > 1:
                out = out.mean(dim=1)  
        return out


class SimpleNodeEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
    def forward(self, x):
        return self.linear(x)


class SimpleEdgeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(1, emb_dim)  # if edge_attr is 1D
    def forward(self, e):
        return self.linear(e)


class DeepGCNLayer(nn.Module):
    def __init__(self, emb_dim, residual=True, dropout=0.5):
        super(DeepGCNLayer, self).__init__()
        self.residual = residual
        self.conv = dglnn.GraphConv(emb_dim, emb_dim)
        self.norm = nn.BatchNorm1d(emb_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = self.conv(g, x)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.residual:
            h = h + x
        return h


class GNN_node_encoder(nn.Module):
    def __init__(
        self, 
        num_layer, 
        emb_dim, 
        node_encoder, 
        edge_encoder, 
        drop_ratio=0.5, 
        JK="last", 
        residual=False, 
        gnn_type="gin"
    ):
        super(GNN_node_encoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.edge_encoder = edge_encoder
        self.node_encoder = node_encoder

        # Define GNN layers
        self.convs = nn.ModuleList()
        # Define separate BatchNorm layers for node and edge if both are needed
        self.node_batch_norms = nn.ModuleList()
        self.edge_batch_norms = nn.ModuleList()
        self.gnn_type = gnn_type

        for _ in range(num_layer):
            if self.gnn_type == "gin":
                self.convs.append(
                    dglnn.GINConv(
                        nn.Linear(emb_dim, emb_dim),
                        aggregator_type='sum'
                    )
                )
            elif self.gnn_type == "gcn":
                self.convs.append(
                    dglnn.GraphConv(
                        emb_dim, emb_dim
                    )
                )
            elif self.gnn_type == "gat":
                # Example: EGATConv returns updated node and edge features
                self.convs.append(
                    dglnn.EGATConv(emb_dim, emb_dim, emb_dim, emb_dim, num_heads=1)
                )
            elif self.gnn_type == "sage":
                self.convs.append(
                    dglnn.SAGEConv(emb_dim, emb_dim, "mean")
                )
            elif self.gnn_type == "deepgcn":
                self.convs.append(DeepGCNLayer(emb_dim, residual=True, dropout=0.2))
            else:
                raise ValueError(f"Undefined GNN type: {self.gnn_type}")


    def forward(self, graph, edge_feat=None, perturb=None):
        node_feat = graph.ndata['feat']
        edge_feat = graph.edata['edge_attr']
        if self.node_encoder is not None:
            node_feat = self.node_encoder(node_feat) 
        if self.edge_encoder is not None:
            edge_feat = self.edge_encoder(edge_feat)

        h_list = []
        node_h = node_feat
        edge_h = edge_feat

        for layer in range(self.num_layer):
            if self.gnn_type in ['gat']:
                node_h, edge_h = self.convs[layer](graph, node_h, edge_h)
            elif self.gnn_type in ['gcn']:
                node_h = self.convs[layer](graph, node_h, edge_weight=edge_h)
            elif self.gnn_type == "deepgcn":
                node_h = self.convs[layer](graph, node_h)
            else:
                node_h = self.convs[layer](graph, node_h, edge_h)

            # Residual connection (if enabled) uses the *previous* node features
            if self.residual and len(h_list) > 0:
                node_h = node_h + h_list[-1]

            if layer == self.num_layer - 1:
                node_h = F.dropout(node_h, p=self.drop_ratio, training=self.training)
            else:
                node_h = F.dropout(F.relu(node_h), p=self.drop_ratio, training=self.training)

            h_list.append(node_h)

        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = torch.sum(torch.stack(h_list), dim=0)
        elif self.JK == "cat":
            node_representation = torch.cat(h_list, dim=-1)
        else:
            raise ValueError(f"Unknown JK method: {self.JK}")

        return node_representation