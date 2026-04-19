import torch
import os
import dgl
import dgl.nn as dglnn
import torch.optim as optim
import dgl.sparse as dglsp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ogb.graphproppred.mol_encoder import AtomEncoder
from tqdm import tqdm
from typing import Optional
from dgl.nn.pytorch.gt import path_encoder
from graphtransformer import GraphormerLayer
from dgl.nn import DegreeEncoder, PathEncoder, SpatialEncoder, GINConv, GraphConv
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, GINConv, PairNorm, GINEConv, GPSConv, GatedGraphConv, DeepGCNLayer, LayerNorm, SGFormer
from torch_geometric.nn.attention import PerformerAttention
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from dgl.nn import GraphConv, GATConv
from node_encoder import GNN_node_encoder
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, PairNorm

class SiGTDelay(pl.LightningModule):
    def __init__(self, out_dim=1, edge_dim=1, node_encoding='gnn', max_degree=512, num_spatial=511, multi_hop_max_dist=5, num_encoder_layers=6, embedding_dim=256, ffn_embedding_dim=128, num_attention_heads=4, dropout=0.1, pre_layernorm=True, activation_fn=nn.GELU(), total_updates: int = 0,
        warmup_updates: int = 0, lr: float = 8e-4, weight_decay: float = 0.0, eps: float = 1e-8, batch_size: int = 32, mode: str = 'segment', vic_only = False):
        super(SiGTDelay, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.mode = mode
        self.vic_only = vic_only
        self.eps = eps
        self.weight_decay = weight_decay
        self.total_updates = total_updates
        self.warmup_updates = warmup_updates
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.node_encoding = node_encoding
        self.graph_token = nn.Parameter(torch.zeros(1, 3, embedding_dim))
        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=embedding_dim)
        self.path_encoder = PathEncoder(max_len=multi_hop_max_dist, feat_dim=edge_dim, num_heads=num_attention_heads,)
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=num_attention_heads)
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.linear_atten_bias_couple = nn.Linear(1, num_attention_heads)
        self.linear_atten_bias_net = nn.Linear(1, num_attention_heads)
        self.linear_node_encoder = nn.Linear(1, embedding_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))
        self.criterion = nn.L1Loss()
        if self.node_encoding == 'gnn':
            node_encoder = torch.nn.Linear(1, embedding_dim)
            num_layers = 3
            drop_ratio = 0.2
            edge_encoder = torch.nn.Linear(2, embedding_dim)
            self.node_encoder = GNN_node_encoder(num_layers, embedding_dim, node_encoder, edge_encoder, drop_ratio, JK="last", residual=True, gnn_type="gat")
        elif self.node_encoding == 'sgnn':
            num_layers = 2
            drop_ratio = 0.2
            node_encoder = torch.nn.Linear(1, embedding_dim)
            edge_encoder = torch.nn.Linear(2, embedding_dim)
            self.node_encoder = GNN_node_encoder(num_layers, embedding_dim, node_encoder, edge_encoder, drop_ratio, JK="last", residual=True, gnn_type="gcn")

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(self, bg, bsg, node_feat, path_data, dist, attn_net_A, attn_couple_A, attn_mask=None, net_attn_mask=None):
        node_feat_emb = self.linear_node_encoder(node_feat)
        attn_mask += net_attn_mask
        num_graphs, max_num_nodes, _ = node_feat.shape
        graph_token_feat = self.graph_token.repeat(num_graphs, 1, 1)
        if self.node_encoding == 'gnn':
            node_feat = self.node_encoder(bg)
            num_nodes_per_graph = bg.batch_num_nodes()
            batchsize = len(num_nodes_per_graph)
            max_nodes = max(num_nodes_per_graph).item()
            padded_node_embeddings = torch.zeros((batchsize, max_nodes, node_feat.shape[-1]), dtype=node_feat.dtype, device=node_feat.device)
            start_idx = 0
            for i, num_nodes in enumerate(num_nodes_per_graph):
                padded_node_embeddings[i, :num_nodes, :] = node_feat[start_idx : start_idx + num_nodes]
                start_idx += num_nodes  
            x = torch.cat([graph_token_feat, padded_node_embeddings], dim=1)
        elif self.node_encoding == 'sgnn':
            node_feat = self.node_encoder(bsg).squeeze()
            target_mask = bsg.ndata['target_mask']
            target_node_embeddings = node_feat[target_mask.bool()]
            num_nodes_per_graph = bg.batch_num_nodes()
            batchsize = len(num_nodes_per_graph)
            max_nodes = max(num_nodes_per_graph).item()
            padded_node_embeddings = torch.zeros((batchsize, max_nodes, node_feat.shape[-1]), dtype=node_feat.dtype, device=node_feat.device)
            start_idx = 0
            for i, num_nodes in enumerate(num_nodes_per_graph):
                padded_node_embeddings[i, 3:num_nodes, :] = target_node_embeddings[start_idx:start_idx+(num_nodes-3)]
                start_idx += num_nodes-3  
            padded_node_embeddings += node_feat_emb
            x = torch.cat([graph_token_feat, padded_node_embeddings], dim=1)
            
        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 3,          
            max_num_nodes + 3,
            self.num_heads,
            device=attn_net_A.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 3:, 3:, :] = path_encoding + spatial_encoding

        atten_bias_couple = self.linear_atten_bias_couple(attn_couple_A)
        atten_bias_net = self.linear_atten_bias_net(attn_net_A)
        extra_attn_bias = atten_bias_net + atten_bias_couple
        attn_bias = attn_bias + atten_bias_couple
        
        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )


        graph_rep = x[:, 3:, :] 
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep)

        return graph_rep
    
    def training_step(self, batch, batch_idx):
        (
            bg,
            bsg,
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            attn_mask,
            net_token_attn_mask,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A
        ) = batch
        out = self(
            bg,
            bsg,
            node_feat,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A,
            attn_mask=attn_mask,
            net_attn_mask = net_token_attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'sink':
            if self.vic_only:
                sink_vic_mask[sink_vic_mask == -1] = 0
                mask = sink_vic_mask.bool()
            else:
                sink_mask[sink_mask == -1] = 0
                mask = sink_mask.bool()
        if self.mode == 'segment':
            if self.vic_only:
                node_vic_mask[node_vic_mask == -1] = 0
                mask = node_vic_mask.bool()
            else:
                node_mask[node_mask == -1] = 0
                mask = node_mask.bool()
        out = out[mask]
        label = node_label[mask]
        label = label.squeeze(dim=-1)
        loss = self.criterion(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            bg,
            bsg,
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            attn_mask,
            net_token_attn_mask,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A
        ) = batch
        out = self(
            bg,
            bsg,
            node_feat,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A,
            attn_mask=attn_mask,
            net_attn_mask = net_token_attn_mask,
        )
        out = out.squeeze(dim=-1)
        # 2A1V-sink
        if self.mode == 'sink':
            if self.vic_only:
                sink_vic_mask[sink_vic_mask == -1] = 0
                mask = sink_vic_mask.bool()
                vic_mask = sink_vic_mask.bool()
            else:
                sink_mask[sink_mask == -1] = 0
                sink_vic_mask[sink_vic_mask == -1] = 0
                sink_agg1_mask[sink_agg1_mask == -1] = 0
                sink_agg2_mask[sink_agg2_mask == -1] = 0
                
                mask = sink_mask.bool()
                agg1_mask = sink_agg1_mask.bool()
                agg2_mask = sink_agg2_mask.bool()
                vic_mask = sink_vic_mask.bool()
        # 2A1V-segment
        if self.mode == 'segment':
            if self.vic_only:
                node_vic_mask[node_vic_mask == -1] = 0
                mask = node_vic_mask.bool()
                vic_mask = node_vic_mask.bool()
            else:
                node_mask[node_mask == -1] = 0
                node_vic_mask[node_vic_mask == -1] = 0
                node_agg1_mask[node_agg1_mask == -1] = 0
                node_agg2_mask[node_agg2_mask == -1] = 0
                
                mask = node_mask.bool()
                agg1_mask = node_agg1_mask.bool()
                agg2_mask = node_agg2_mask.bool()
                vic_mask = node_vic_mask.bool()
        
        mask = mask
        mask_out = out[mask]
        mask_label = node_label[mask]
        mask_label = mask_label.squeeze(dim=-1)
        loss = self.criterion(mask_out, mask_label)
        zeromask = mask_label != 0
        zeromask = zeromask.squeeze(-1)
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        masked_label = masked_label.squeeze(-1)
        acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        if not self.vic_only:
            mask_out = out[agg1_mask]
            mask_label = node_label[agg1_mask]
            mask_label = mask_label.squeeze(dim=-1)
            zeromask = mask_label != 0
            zeromask = zeromask.squeeze(-1)
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            masked_label = masked_label.squeeze(-1)
            aggt1_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
            
            mask_out = out[agg2_mask]
            mask_label = node_label[agg2_mask]
            zeromask = mask_label != 0
            zeromask = zeromask.squeeze(-1)
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            masked_label = masked_label.squeeze(-1)
            aggt2_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        mask_out = out[vic_mask] 
        mask_label = node_label[vic_mask]
        zeromask = mask_label != 0
        zeromask = zeromask.squeeze(-1)
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        masked_label = masked_label.squeeze(-1)
        vict_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)
        if not self.vic_only:
            self.log('aggt1_acc', aggt1_acc, prog_bar=True, batch_size=self.batch_size)
            self.log('aggt2_acc', aggt2_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vict_acc', vict_acc, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def configure_optimizers(self): 
        """
        Returns a dict with 'optimizer' and 'lr_scheduler'.
        PyTorch Lightning will handle calling the scheduler per step.
        """
        optimizer = AdamW(
            self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
        # Polynomial decay with warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_updates),
            num_training_steps=int(self.total_updates),
            lr_end=1e-9,
            power=1.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # call step() after every batch
            },
        }


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
    @staticmethod
    def load_from_checkpoint(checkpoint_path, *args, **kwargs):
        model = SiGTDelay(*args, **kwargs)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model

class SiGTGlitch_Ablation(pl.LightningModule):
    def __init__(self, out_dim=4, edge_dim=1, node_encoding='gnn', max_degree=512, num_spatial=511, multi_hop_max_dist=5, num_encoder_layers=6, embedding_dim=32, ffn_embedding_dim=128, num_attention_heads=4, dropout=0.1, pre_layernorm=True, activation_fn=nn.GELU(), total_updates: int = 0,
        warmup_updates: int = 0, lr: float = 4e-4, weight_decay: float = 0.0, eps: float = 1e-8, batch_size: int = 32, mode: str = 'segment'):
        super(SiGTGlitch_Ablation, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.mode = mode
        self.weight_decay = weight_decay
        self.total_updates = total_updates
        self.warmup_updates = warmup_updates
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.node_encoding = node_encoding
        self.graph_token = nn.Parameter(torch.zeros(1, 3, embedding_dim))
        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=embedding_dim)
        self.path_encoder = PathEncoder(max_len=multi_hop_max_dist, feat_dim=edge_dim, num_heads=num_attention_heads,)
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=num_attention_heads)
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.linear_atten_bias_couple = nn.Linear(1, num_attention_heads)
        self.linear_atten_bias_net = nn.Linear(1, num_attention_heads)
        self.linear_node_encoder = nn.Linear(1, embedding_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))
        self.criterion = nn.L1Loss()
        if self.node_encoding == 'gnn':
            node_encoder = torch.nn.Linear(1, embedding_dim)
            num_layers = 3
            drop_ratio = 0.2
            edge_encoder = torch.nn.Linear(2, embedding_dim)
            self.node_encoder = GNN_node_encoder(num_layers, embedding_dim, node_encoder, edge_encoder, drop_ratio, JK="last", residual=True, gnn_type="gat")
        elif self.node_encoding == 'sgnn':
            # num_layers = 1
            # drop_ratio = 0.2
            # node_encoder = torch.nn.Linear(1, embedding_dim)
            # edge_encoder = torch.nn.Linear(2, embedding_dim)
            # self.node_encoder = GNN_node_encoder(num_layers, embedding_dim, node_encoder, edge_encoder, drop_ratio, JK="last", residual=True, gnn_type="gcn")
            # disable mpe
            self.node_encoder = nn.Linear(1, embedding_dim)
            self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=embedding_dim)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(self, bg, bsg, node_feat, in_degree, out_degree, path_data, dist, attn_net_A, attn_couple_A, attn_mask=None, net_attn_mask=None):
        node_feat_emb = self.linear_node_encoder(node_feat)
        # attn_mask += net_attn_mask
        num_graphs, max_num_nodes, _ = node_feat.shape
        graph_token_feat = self.graph_token.repeat(num_graphs, 1, 1)
        if self.node_encoding == 'gnn':
            node_feat = self.node_encoder(bg)
            num_nodes_per_graph = bg.batch_num_nodes()
            batchsize = len(num_nodes_per_graph)
            max_nodes = max(num_nodes_per_graph).item()
            padded_node_embeddings = torch.zeros((batchsize, max_nodes, node_feat.shape[-1]), dtype=node_feat.dtype, device=node_feat.device)
            start_idx = 0
            for i, num_nodes in enumerate(num_nodes_per_graph):
                padded_node_embeddings[i, :num_nodes, :] = node_feat[start_idx : start_idx + num_nodes]
                start_idx += num_nodes  
            x = torch.cat([graph_token_feat, padded_node_embeddings], dim=1)
        elif self.node_encoding == 'sgnn':
            # node_feat = self.node_encoder(bsg).squeeze()
            # target_mask = bsg.ndata['target_mask']
            # target_node_embeddings = node_feat[target_mask.bool()]
            # num_nodes_per_graph = bg.batch_num_nodes()
            # batchsize = len(num_nodes_per_graph)
            # max_nodes = max(num_nodes_per_graph).item()
            # padded_node_embeddings = torch.zeros((batchsize, max_nodes, node_feat.shape[-1]), dtype=node_feat.dtype, device=node_feat.device)
            # start_idx = 0
            # for i, num_nodes in enumerate(num_nodes_per_graph):
            #     padded_node_embeddings[i, 3:num_nodes, :] = target_node_embeddings[start_idx:start_idx+(num_nodes-3)]
            #     start_idx += num_nodes-3  
            # padded_node_embeddings += node_feat_emb
            # x = torch.cat([graph_token_feat, padded_node_embeddings], dim=1)
            # disable mpe
            num_graphs, max_num_nodes, _ = node_feat.shape
            deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))
            graph_token_feat = self.graph_token.repeat(num_graphs, 1, 1)

            node_embeddings = self.node_encoder(node_feat)
            deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))
            node_embeddings = node_embeddings + deg_emb
            x = torch.cat([graph_token_feat, node_embeddings], dim=1)
            
        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 3,          
            max_num_nodes + 3,
            self.num_heads,
            device=attn_net_A.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 3:, 3:, :] = path_encoding + spatial_encoding
        # disable IIN
        atten_bias_couple = self.linear_atten_bias_couple(attn_couple_A)
        atten_bias_net = self.linear_atten_bias_net(attn_net_A)
        extra_attn_bias = atten_bias_net + atten_bias_couple
        attn_bias = attn_bias + extra_attn_bias
        
        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )


        graph_rep = x[:, 3:, :] 
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep)

        return graph_rep
    
    def training_step(self, batch, batch_idx):
        (
            bg,
            bsg,
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            in_degree,
            out_degree,
            attn_mask,
            net_token_attn_mask,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A
        ) = batch
        out = self(
            bg,
            bsg,
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A,
            attn_mask=attn_mask,
            net_attn_mask = net_token_attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'segment':
            node_vic_mask[node_vic_mask == -1] = 0
            vic_mask = node_vic_mask.bool()
        if self.mode == 'sink':
            sink_vic_mask[sink_vic_mask == -1] = 0
            vic_mask = sink_vic_mask.bool()
        out = out[vic_mask]
        label = node_label[vic_mask]
        label = label.squeeze(dim=-1)
        loss = self.criterion(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            bg,
            bsg,
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            in_degree,
            out_degree,
            attn_mask,
            net_token_attn_mask,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A
        ) = batch
        out = self(
            bg,
            bsg,
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A,
            attn_mask=attn_mask,
            net_attn_mask = net_token_attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'segment':
            node_vic_mask[node_vic_mask == -1] = 0
            vic_mask = node_vic_mask.bool()
        if self.mode == 'sink':
            sink_vic_mask[sink_vic_mask == -1] = 0
            vic_mask = sink_vic_mask.bool()
        mask_out = out[vic_mask] 
        mask_label = node_label[vic_mask]
        loss = self.criterion(mask_out, mask_label)
        def compute_accuracy(pred, label):
            mask = label != 0
            pred_masked = pred[mask]
            label_masked = label[mask]
            acc = 100 - torch.mean(torch.abs(pred_masked - label_masked) / torch.abs(label_masked)) * 100
            return acc
        vmax_p_acc = compute_accuracy(mask_out[:, 0], mask_label[:, 0])
        tw_p_acc   = compute_accuracy(mask_out[:, 1], mask_label[:, 1])
        vmax_n_acc = compute_accuracy(mask_out[:, 2], mask_label[:, 2])
        tw_n_acc   = compute_accuracy(mask_out[:, 3], mask_label[:, 3])
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_p_acc', vmax_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_p_acc', tw_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_n_acc', vmax_n_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_n_acc', tw_n_acc, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def configure_optimizers(self): 
        """
        Returns a dict with 'optimizer' and 'lr_scheduler'.
        PyTorch Lightning will handle calling the scheduler per step.
        """
        optimizer = AdamW(
            self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
        # Polynomial decay with warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_updates),
            num_training_steps=int(self.total_updates),
            lr_end=1e-9,
            power=1.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # call step() after every batch
            },
        }


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0004, weight_decay=1e-4)
    @staticmethod
    def load_from_checkpoint(checkpoint_path, *args, **kwargs):
        model = SiGTGlitch_Ablation(*args, **kwargs)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model
    
    
class SiGTGlitch(pl.LightningModule):
    def __init__(self, out_dim=4, edge_dim=1, node_encoding='gnn', max_degree=512, num_spatial=511, multi_hop_max_dist=5, num_encoder_layers=6, embedding_dim=32, ffn_embedding_dim=128, num_attention_heads=4, dropout=0.1, pre_layernorm=True, activation_fn=nn.GELU(), total_updates: int = 0,
        warmup_updates: int = 0, lr: float = 4e-4, weight_decay: float = 0.0, eps: float = 1e-8, batch_size: int = 32, mode: str = 'segment'):
        super(SiGTGlitch, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.mode = mode
        self.weight_decay = weight_decay
        self.total_updates = total_updates
        self.warmup_updates = warmup_updates
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.node_encoding = node_encoding
        self.graph_token = nn.Parameter(torch.zeros(1, 3, embedding_dim))
        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=embedding_dim)
        self.path_encoder = PathEncoder(max_len=multi_hop_max_dist, feat_dim=edge_dim, num_heads=num_attention_heads,)
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=num_attention_heads)
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.linear_atten_bias_couple = nn.Linear(1, num_attention_heads)
        self.linear_atten_bias_net = nn.Linear(1, num_attention_heads)
        self.linear_node_encoder = nn.Linear(1, embedding_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))
        self.criterion = nn.L1Loss()
        if self.node_encoding == 'gnn':
            node_encoder = torch.nn.Linear(1, embedding_dim)
            num_layers = 4
            drop_ratio = 0.2
            edge_encoder = torch.nn.Linear(2, embedding_dim)
            self.node_encoder = GNN_node_encoder(num_layers, embedding_dim, node_encoder, edge_encoder, drop_ratio, JK="last", residual=True, gnn_type="deepgcn")
        elif self.node_encoding == 'sgnn':
            num_layers = 2
            drop_ratio = 0.2
            node_encoder = torch.nn.Linear(1, embedding_dim)
            edge_encoder = torch.nn.Linear(2, embedding_dim)
            self.node_encoder = GNN_node_encoder(num_layers, embedding_dim, node_encoder, edge_encoder, drop_ratio, JK="last", residual=True, gnn_type="gin")

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(self, bg, bsg, node_feat, path_data, dist, attn_net_A, attn_couple_A, attn_mask=None, net_attn_mask=None):
        node_feat_emb = self.linear_node_encoder(node_feat)
        attn_mask += net_attn_mask
        num_graphs, max_num_nodes, _ = node_feat.shape
        graph_token_feat = self.graph_token.repeat(num_graphs, 1, 1)
        if self.node_encoding == 'gnn':
            node_feat = self.node_encoder(bg)
            num_nodes_per_graph = bg.batch_num_nodes()
            batchsize = len(num_nodes_per_graph)
            max_nodes = max(num_nodes_per_graph).item()
            padded_node_embeddings = torch.zeros((batchsize, max_nodes, node_feat.shape[-1]), dtype=node_feat.dtype, device=node_feat.device)
            start_idx = 0
            for i, num_nodes in enumerate(num_nodes_per_graph):
                padded_node_embeddings[i, :num_nodes, :] = node_feat[start_idx : start_idx + num_nodes]
                start_idx += num_nodes  
            x = torch.cat([graph_token_feat, padded_node_embeddings], dim=1)
        elif self.node_encoding == 'sgnn':
            node_feat = self.node_encoder(bsg).squeeze()
            target_mask = bsg.ndata['target_mask']
            target_node_embeddings = node_feat[target_mask.bool()]
            num_nodes_per_graph = bg.batch_num_nodes()
            batchsize = len(num_nodes_per_graph)
            max_nodes = max(num_nodes_per_graph).item()
            padded_node_embeddings = torch.zeros((batchsize, max_nodes, node_feat.shape[-1]), dtype=node_feat.dtype, device=node_feat.device)
            start_idx = 0
            for i, num_nodes in enumerate(num_nodes_per_graph):
                padded_node_embeddings[i, 3:num_nodes, :] = target_node_embeddings[start_idx:start_idx+(num_nodes-3)]
                start_idx += num_nodes-3  
            padded_node_embeddings += node_feat_emb
            x = torch.cat([graph_token_feat, padded_node_embeddings], dim=1)
            
        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 3,          
            max_num_nodes + 3,
            self.num_heads,
            device=attn_net_A.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 3:, 3:, :] = path_encoding + spatial_encoding

        atten_bias_couple = self.linear_atten_bias_couple(attn_couple_A)
        atten_bias_net = self.linear_atten_bias_net(attn_net_A)
        extra_attn_bias = atten_bias_net + atten_bias_couple
        attn_bias = attn_bias + extra_attn_bias
        
        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )


        graph_rep = x[:, 3:, :] 
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep)

        return graph_rep
    
    def training_step(self, batch, batch_idx):
        (
            bg,
            bsg,
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            attn_mask,
            net_token_attn_mask,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A
        ) = batch
        out = self(
            bg,
            bsg,
            node_feat,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A,
            attn_mask=attn_mask,
            net_attn_mask = net_token_attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'segment':
            node_vic_mask[node_vic_mask == -1] = 0
            vic_mask = node_vic_mask.bool()
        if self.mode == 'sink':
            sink_vic_mask[sink_vic_mask == -1] = 0
            vic_mask = sink_vic_mask.bool()
        out = out[vic_mask]
        label = node_label[vic_mask]
        label = label.squeeze(dim=-1)
        loss = self.criterion(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            bg,
            bsg,
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            attn_mask,
            net_token_attn_mask,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A
        ) = batch
        out = self(
            bg,
            bsg,
            node_feat,
            path_data,
            dist,
            attn_net_A, 
            attn_couple_A,
            attn_mask=attn_mask,
            net_attn_mask = net_token_attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'segment':
            node_vic_mask[node_vic_mask == -1] = 0
            vic_mask = node_vic_mask.bool()
        if self.mode == 'sink':
            sink_vic_mask[sink_vic_mask == -1] = 0
            vic_mask = sink_vic_mask.bool()
        mask_out = out[vic_mask] 
        mask_label = node_label[vic_mask]
        loss = self.criterion(mask_out, mask_label)
        def compute_accuracy(pred, label):
            mask = label != 0
            pred_masked = pred[mask]
            label_masked = label[mask]
            acc = 100 - torch.mean(torch.abs(pred_masked - label_masked) / torch.abs(label_masked)) * 100
            return acc
        vmax_p_acc = compute_accuracy(mask_out[:, 0], mask_label[:, 0])
        tw_p_acc   = compute_accuracy(mask_out[:, 1], mask_label[:, 1])
        vmax_n_acc = compute_accuracy(mask_out[:, 2], mask_label[:, 2])
        tw_n_acc   = compute_accuracy(mask_out[:, 3], mask_label[:, 3])
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_p_acc', vmax_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_p_acc', tw_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_n_acc', vmax_n_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_n_acc', tw_n_acc, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def configure_optimizers(self): 
        """
        Returns a dict with 'optimizer' and 'lr_scheduler'.
        PyTorch Lightning will handle calling the scheduler per step.
        """
        optimizer = AdamW(
            self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
       
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_updates),
            num_training_steps=int(self.total_updates),
            lr_end=1e-9,
            power=1.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
            },
        }


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0004, weight_decay=1e-4)
    @staticmethod
    def load_from_checkpoint(checkpoint_path, *args, **kwargs):
        model = SiGTGlitch(*args, **kwargs)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model
    

class GraphomerGlitch(pl.LightningModule):
    def __init__(self, out_dim=4, edge_dim=1, max_degree=512, num_spatial=511, multi_hop_max_dist=5, num_encoder_layers=6, embedding_dim=32, ffn_embedding_dim=128, num_attention_heads=4, dropout=0.1, pre_layernorm=True, activation_fn=nn.GELU(), total_updates: int = 0,
        warmup_updates: int = 0, lr: float = 4e-4, weight_decay: float = 0.0, eps: float = 1e-8, batch_size: int = 32, mode: str = 'segment'):
        super(GraphomerGlitch, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.eps = eps
        self.mode = mode
        self.weight_decay = weight_decay
        self.total_updates = total_updates
        self.warmup_updates = warmup_updates
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.graph_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=embedding_dim)
        self.path_encoder = PathEncoder(max_len=multi_hop_max_dist, feat_dim=edge_dim, num_heads=num_attention_heads,)
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=num_attention_heads)
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.linear_atten_bias_couple = nn.Linear(1, num_attention_heads)
        self.linear_atten_bias_net = nn.Linear(1, num_attention_heads)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))
        self.criterion = nn.L1Loss()
        self.node_encoder = nn.Linear(1, embedding_dim)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(self, node_feat, in_degree, out_degree, path_data, dist, attn_mask=None,):
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))
        graph_token_feat = self.graph_token.repeat(num_graphs, 1, 1)

        node_embeddings = self.node_encoder(node_feat)
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))
        node_embeddings = node_embeddings + deg_emb
        x = torch.cat([graph_token_feat, node_embeddings], dim=1)
        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 1,          
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        t = self.graph_token_virtual_distance.weight.reshape(1, 1, self.num_heads)
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t
        
        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        graph_rep = x[:, 1:, :] 
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep)

        return graph_rep
    
    def training_step(self, batch, batch_idx):
        (
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            in_degree,
            out_degree,
            attn_mask,
            path_data,
            dist
        ) = batch
        out = self(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_mask=attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'segment':
            node_vic_mask[node_vic_mask == -1] = 0
            vic_mask = node_vic_mask.bool()
        if self.mode == 'sink':
            sink_vic_mask[sink_vic_mask == -1] = 0
            vic_mask = sink_vic_mask.bool()
        out = out[vic_mask]
        label = node_label[vic_mask]
        label = label.squeeze(dim=-1)
        loss = self.criterion(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            in_degree,
            out_degree,
            attn_mask,
            path_data,
            dist
        ) = batch
        out = self(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_mask=attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'segment':
            node_vic_mask[node_vic_mask == -1] = 0
            vic_mask = node_vic_mask.bool()
        if self.mode == 'sink':
            sink_vic_mask[sink_vic_mask == -1] = 0
            vic_mask = sink_vic_mask.bool()
        mask_out = out[vic_mask] 
        mask_label = node_label[vic_mask]
        loss = self.criterion(mask_out, mask_label)
        def compute_accuracy(pred, label):
            mask = label != 0
            pred_masked = pred[mask]
            label_masked = label[mask]
            acc = 100 - torch.mean(torch.abs(pred_masked - label_masked) / torch.abs(label_masked)) * 100
            return acc
        vmax_p_acc = compute_accuracy(mask_out[:, 0], mask_label[:, 0])
        tw_p_acc   = compute_accuracy(mask_out[:, 1], mask_label[:, 1])
        vmax_n_acc = compute_accuracy(mask_out[:, 2], mask_label[:, 2])
        tw_n_acc   = compute_accuracy(mask_out[:, 3], mask_label[:, 3])
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_p_acc', vmax_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_p_acc', tw_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_n_acc', vmax_n_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_n_acc', tw_n_acc, prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def configure_optimizers(self): 
        """
        Returns a dict with 'optimizer' and 'lr_scheduler'.
        PyTorch Lightning will handle calling the scheduler per step.
        """
        optimizer = AdamW(
            self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
        # Polynomial decay with warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_updates),
            num_training_steps=int(self.total_updates),
            lr_end=1e-9,
            power=1.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  
            },
        }


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=6e-4)
    @staticmethod
    def load_from_checkpoint(checkpoint_path, *args, **kwargs):
        model = GraphomerGlitch(*args, **kwargs)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model
    

class GraphomerDelay(pl.LightningModule):
    def __init__(self, out_dim=1, edge_dim=1, max_degree=512, num_spatial=511, multi_hop_max_dist=5, num_encoder_layers=6, embedding_dim=32, ffn_embedding_dim=128, num_attention_heads=4, dropout=0.1, pre_layernorm=True, activation_fn=nn.GELU(), total_updates: int = 0,
        warmup_updates: int = 0, lr: float = 4e-4, weight_decay: float = 0.0, eps: float = 1e-8, batch_size: int = 32, mode: str = 'segment', vic_only: bool = False):
        super(GraphomerDelay, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.mode = mode
        self.vic_only = vic_only
        self.eps = eps
        self.weight_decay = weight_decay
        self.total_updates = total_updates
        self.warmup_updates = warmup_updates
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.graph_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.degree_encoder = DegreeEncoder(max_degree=max_degree, embedding_dim=embedding_dim)
        self.path_encoder = PathEncoder(max_len=multi_hop_max_dist, feat_dim=edge_dim, num_heads=num_attention_heads,)
        self.spatial_encoder = SpatialEncoder(max_dist=num_spatial, num_heads=num_attention_heads)
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.linear_atten_bias_couple = nn.Linear(1, num_attention_heads)
        self.linear_atten_bias_net = nn.Linear(1, num_attention_heads)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=activation_fn,
                    norm_first=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))
        self.criterion = nn.L1Loss()
        self.node_encoder = nn.Linear(1, embedding_dim)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(self, node_feat, in_degree, out_degree, path_data, dist, attn_mask=None,):
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))
        graph_token_feat = self.graph_token.repeat(num_graphs, 1, 1)

        node_embeddings = self.node_encoder(node_feat)
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))
        node_embeddings = node_embeddings + deg_emb
        x = torch.cat([graph_token_feat, node_embeddings], dim=1)
        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 1,          
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        t = self.graph_token_virtual_distance.weight.reshape(1, 1, self.num_heads)
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t
        
        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        graph_rep = x[:, 1:, :] 
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep)

        return graph_rep
    
    def training_step(self, batch, batch_idx):
        (
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            in_degree,
            out_degree,
            attn_mask,
            path_data,
            dist
        ) = batch
        out = self(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_mask=attn_mask,
        )
        out = out.squeeze(dim=-1)
        if self.mode == 'sink':
            if self.vic_only:
                sink_vic_mask[sink_vic_mask == -1] = 0
                mask = sink_vic_mask.bool()
            else:
                sink_mask[sink_mask == -1] = 0
                mask = sink_mask.bool()
        if self.mode == 'segment':
            if self.vic_only:
                node_vic_mask[node_vic_mask == -1] = 0
                mask = node_vic_mask.bool()
            else:
                node_mask[node_mask == -1] = 0
                mask = node_mask.bool()
        out = out[mask]
        label = node_label[mask]
        label = label.squeeze(dim=-1)
        loss = self.criterion(out, label)
        self.log('train_loss', loss)
        return loss

        

    def validation_step(self, batch, batch_idx):
        (
            node_feat,
            node_label,
            node_mask,
            node_vic_mask,
            node_agg1_mask,
            node_agg2_mask,
            sink_mask,
            sink_vic_mask,
            sink_agg1_mask,
            sink_agg2_mask,
            in_degree,
            out_degree,
            attn_mask,
            path_data,
            dist
        ) = batch
        out = self(
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
            attn_mask=attn_mask,
        )
        out = out.squeeze(dim=-1)
        
        # 2A1V-sink
        if self.mode == 'sink':
            if self.vic_only:
                sink_vic_mask[sink_vic_mask == -1] = 0
                mask = sink_vic_mask.bool()
                vic_mask = sink_vic_mask.bool()
            else:
                sink_mask[sink_mask == -1] = 0
                sink_vic_mask[sink_vic_mask == -1] = 0
                sink_agg1_mask[sink_agg1_mask == -1] = 0
                sink_agg2_mask[sink_agg2_mask == -1] = 0
                
                mask = sink_mask.bool()
                agg1_mask = sink_agg1_mask.bool()
                agg2_mask = sink_agg2_mask.bool()
                vic_mask = sink_vic_mask.bool()
        # 2A1V-segment
        if self.mode == 'segment':
            if self.vic_only:
                node_vic_mask[node_vic_mask == -1] = 0
                mask = node_vic_mask.bool()
                vic_mask = node_vic_mask.bool()
            else:
                node_mask[node_mask == -1] = 0
                node_vic_mask[node_vic_mask == -1] = 0
                node_agg1_mask[node_agg1_mask == -1] = 0
                node_agg2_mask[node_agg2_mask == -1] = 0
                
                mask = node_mask.bool()
                agg1_mask = node_agg1_mask.bool()
                agg2_mask = node_agg2_mask.bool()
                vic_mask = node_vic_mask.bool()
        
        mask = mask
        mask_out = out[mask]
        mask_label = node_label[mask]
        mask_label = mask_label.squeeze(dim=-1)
        loss = self.criterion(mask_out, mask_label)
        zeromask = mask_label != 0
        zeromask = zeromask.squeeze(-1)
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        masked_label = masked_label.squeeze(-1)
        acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        if not self.vic_only:
            mask_out = out[agg1_mask]
            mask_label = node_label[agg1_mask]
            mask_label = mask_label.squeeze(dim=-1)
            zeromask = mask_label != 0
            zeromask = zeromask.squeeze(-1)
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            masked_label = masked_label.squeeze(-1)
            aggt1_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
            
            mask_out = out[agg2_mask]
            mask_label = node_label[agg2_mask]
            zeromask = mask_label != 0
            zeromask = zeromask.squeeze(-1)
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            masked_label = masked_label.squeeze(-1)
            aggt2_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        mask_out = out[vic_mask] 
        mask_label = node_label[vic_mask]
        zeromask = mask_label != 0
        zeromask = zeromask.squeeze(-1)
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        masked_label = masked_label.squeeze(-1)
        vict_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)
        if not self.vic_only:
            self.log('aggt1_acc', aggt1_acc, prog_bar=True, batch_size=self.batch_size)
            self.log('aggt2_acc', aggt2_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vict_acc', vict_acc, prog_bar=True, batch_size=self.batch_size)
        return loss
        
    
    def configure_optimizers(self): 
        """
        Returns a dict with 'optimizer' and 'lr_scheduler'.
        PyTorch Lightning will handle calling the scheduler per step.
        """
        optimizer = AdamW(
            self.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay
        )
        # Polynomial decay with warmup
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.warmup_updates),
            num_training_steps=int(self.total_updates),
            lr_end=1e-9,
            power=1.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # call step() after every batch
            },
        }


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=6e-4)
    @staticmethod
    def load_from_checkpoint(checkpoint_path, *args, **kwargs):
        model = GraphomerDelay(*args, **kwargs)
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        return model


class GatedGraphLayer(nn.Module):
    """
    Wraps a single-step GatedGraphConv with input/output projections.
    in_ch  → hidden_channels → GatedGraphConv → hidden_channels → out_ch
    """
    def __init__(self, in_ch, out_ch, hidden_channels, num_steps=1):
        super().__init__()
        self.lin_in  = nn.Linear(in_ch, hidden_channels)
        self.ggc     = GatedGraphConv(hidden_channels, num_layers=num_steps)
        self.lin_out = nn.Linear(hidden_channels, out_ch)

    def forward(self, x, edge_index):
        x = self.lin_in(x)
        x = self.ggc(x, edge_index)
        x = self.lin_out(x)
        return x


class GNNDelay(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        hidden_channels=32,
        batch_size=32,
        conv_layer_num=5,
        gnn_type='gat',
        lr=0.001,
        vic_only = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.gnn_type   = gnn_type.lower()
        self.loss_fn    = nn.L1Loss()
        self.lr         = lr
        self.vic_only   = vic_only
        # pass hidden_channels into the factory so it can build GatedGraphLayer
        conv_factory = self._get_conv_layer_factory(self.gnn_type, hidden_channels)
        self.layers = nn.ModuleList()
        for i in range(conv_layer_num):
            in_c  = in_ch  if i == 0 else hidden_channels
            out_c = out_ch if i == conv_layer_num - 1 else hidden_channels
            self.layers.append(conv_factory(in_c, out_c))

        self.norm = PairNorm(scale=1.0)

    def _get_conv_layer_factory(self, gnn_type, hidden_channels):
        if gnn_type == 'gat':
            return lambda in_c, out_c: GATConv(in_c, out_c, edge_dim=2)
        elif gnn_type == 'gcn':
            return lambda in_c, out_c: GCNConv(in_c, out_c)
        elif gnn_type == 'sage':
            return lambda in_c, out_c: SAGEConv(in_c, out_c)
        elif gnn_type == 'gin':
            def make_gin(in_c, out_c):
                mlp = nn.Sequential(
                    nn.Linear(in_c, out_c),
                    nn.ReLU(),
                    nn.Linear(out_c, out_c),
                )
                return GINConv(mlp)
            return make_gin
        elif gnn_type == 'gatedgraph':
            # Wrap GatedGraphConv with in/out projections
            return lambda in_c, out_c: GatedGraphLayer(
                in_ch=in_c,
                out_ch=out_c,
                hidden_channels=hidden_channels,
                num_steps=1
            )
        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")

    def forward(self, x, edge_index, edge_attr=None):
        edge_index = edge_index.long()

        for conv in self.layers[:-1]:
            if self.gnn_type == 'gat':
                x = F.relu(conv(x, edge_index, edge_attr))
            else:
                x = F.relu(conv(x, edge_index))
            x = self.norm(x)

        last = self.layers[-1]
        if self.gnn_type == 'gat':
            return last(x, edge_index, edge_attr)
        else:
            return last(x, edge_index)

    def training_step(self, batch, batch_idx):
        # only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        out = out[mask]
        label = batch.label[mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        # only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
            vic_mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
            agg1_mask = batch.agg1_mask.bool()
            agg2_mask = batch.agg2_mask.bool()
            vic_mask = batch.vic_mask.bool()
        
        mask_out = out[mask]
        mask_label = batch.label[mask]
        loss = self.loss_fn(mask_out, mask_label)
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        if not self.vic_only:
            mask_out = out[agg1_mask]
            mask_label = batch.label[agg1_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt1_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
            
            mask_out = out[agg2_mask]
            mask_label = batch.label[agg2_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt2_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        mask_out = out[vic_mask] 
        mask_label = batch.label[vic_mask]
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        vict_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)
        if not self.vic_only:
            self.log('aggt1_acc', aggt1_acc, prog_bar=True, batch_size=self.batch_size)
            self.log('aggt2_acc', aggt2_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vict_acc', vict_acc, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=6e-4)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model


class GNNGlitch(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        hidden_channels=32,
        batch_size=32,
        conv_layer_num=5,
        gnn_type='gat',
        lr=0.001,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.gnn_type = gnn_type.lower()
        self.loss_fn = torch.nn.L1Loss()

        conv_layer = self._get_conv_layer(gnn_type)

        self.layers = torch.nn.ModuleList()
        for i in range(conv_layer_num):
            if i == 0:
                self.layers.append(conv_layer(in_ch, hidden_channels))
            elif i == conv_layer_num - 1:
                self.layers.append(conv_layer(hidden_channels, out_ch))
            else:
                self.layers.append(conv_layer(hidden_channels, hidden_channels))

        self.norm = PairNorm(scale=1.0)
        self.lr = lr

    def _get_conv_layer(self, gnn_type):
        if gnn_type == 'gat':
            return lambda in_ch, out_ch: GATConv(in_ch, out_ch, edge_dim=2)
        elif gnn_type == 'gcn':
            return lambda in_ch, out_ch: GCNConv(in_ch, out_ch)
        elif gnn_type == 'sage':
            return lambda in_ch, out_ch: SAGEConv(in_ch, out_ch)
        elif gnn_type == 'gin':
            def gin_nn(in_ch, out_ch):
                return torch.nn.Sequential(
                    torch.nn.Linear(in_ch, out_ch),
                    torch.nn.ReLU(),
                    torch.nn.Linear(out_ch, out_ch)
                )
            return lambda in_ch, out_ch: GINConv(gin_nn(in_ch, out_ch))
        else:
            raise ValueError(f'Unsupported gnn_type: {gnn_type}')

    def forward(self, x, edge_index, edge_attr=None):
        edge_index = edge_index.long()

        for conv in self.layers[:-1]:
            if self.gnn_type == 'gat':
                x = F.relu(conv(x, edge_index, edge_attr))
            else:
                x = F.relu(conv(x, edge_index))
            x = self.norm(x)

        if self.gnn_type == 'gat':
            x = self.layers[-1](x, edge_index, edge_attr)
        else:
            x = self.layers[-1](x, edge_index)

        return x

    def training_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        out = out[vic_mask]
        label = batch.label[vic_mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        mask_out = out[vic_mask]
        mask_label = batch.label[vic_mask]
        loss = self.loss_fn(mask_out, mask_label)

        def compute_accuracy(pred, label):
            mask = label != 0
            pred_masked = pred[mask]
            label_masked = label[mask]
            acc = 100 - torch.mean(torch.abs(pred_masked - label_masked) / torch.abs(label_masked)) * 100
            return acc

        vmax_p_acc = compute_accuracy(mask_out[:, 0], mask_label[:, 0])
        tw_p_acc = compute_accuracy(mask_out[:, 1], mask_label[:, 1])
        vmax_n_acc = compute_accuracy(mask_out[:, 2], mask_label[:, 2])
        tw_n_acc = compute_accuracy(mask_out[:, 3], mask_label[:, 3])

        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_p_acc', vmax_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_p_acc', tw_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_n_acc', vmax_n_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_n_acc', tw_n_acc, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=6e-4)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model


class GraphGPSGlitch(pl.LightningModule):
    def __init__(
        self, channels: int, input_dim: int, pe_dim: int, num_layers: int, attn_type: str, batchsize: int, lr=0.001):
        super().__init__()
        self.node_emb = Linear(1, channels - pe_dim)
        self.pe_lin = Linear(input_dim, pe_dim)
        self.pe_norm = BatchNorm1d(input_dim)
        self.edge_emb = Linear(2, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4, attn_type=attn_type)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 4),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        self.lr =lr
        self.loss_fn = torch.nn.L1Loss()
        self.batch_size = batchsize

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x, _ = conv(x, edge_index, batch, edge_attr=edge_attr)
        return self.mlp(x)      

    def training_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.EigVecs, batch.edge_index, batch.edge_attr, batch.batch)
        out = out[vic_mask]
        label = batch.label[vic_mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.EigVecs, batch.edge_index, batch.edge_attr, batch.batch)
        mask_out = out[vic_mask]
        mask_label = batch.label[vic_mask]
        loss = self.loss_fn(mask_out, mask_label)

        def compute_accuracy(pred, label):
            mask = label != 0
            pred_masked = pred[mask]
            label_masked = label[mask]
            acc = 100 - torch.mean(torch.abs(pred_masked - label_masked) / torch.abs(label_masked)) * 100
            return acc

        vmax_p_acc = compute_accuracy(mask_out[:, 0], mask_label[:, 0])
        tw_p_acc = compute_accuracy(mask_out[:, 1], mask_label[:, 1])
        vmax_n_acc = compute_accuracy(mask_out[:, 2], mask_label[:, 2])
        tw_n_acc = compute_accuracy(mask_out[:, 3], mask_label[:, 3])

        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_p_acc', vmax_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_p_acc', tw_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_n_acc', vmax_n_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_n_acc', tw_n_acc, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
class GraphGPSDelay(pl.LightningModule):
    def __init__(
        self, channels: int, input_dim: int,  pe_dim: int, num_layers: int, attn_type: str, batchsize: int, lr=0.0006, vic_only=False):
        super().__init__()
        self.node_emb = Linear(1, channels - pe_dim)
        self.pe_lin = Linear(input_dim, pe_dim)
        self.pe_norm = BatchNorm1d(input_dim)
        self.edge_emb = Linear(2, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn), heads=4, attn_type=attn_type)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        self.lr =lr
        self.vic_only = vic_only
        self.loss_fn = torch.nn.L1Loss()
        self.batch_size = batchsize

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)
        for conv in self.convs:
            x, _ = conv(x, edge_index, batch, edge_attr=edge_attr)
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
    	# only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
        pe = torch.cat((batch.EigVecs, batch.pestat_RWSE), dim=1)
        out = self(batch.x, pe, batch.edge_index, batch.edge_attr, batch.batch)
        out = out[mask]
        label = batch.label[mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        pe = torch.cat((batch.EigVecs, batch.pestat_RWSE), dim=1)
        out = self(batch.x, pe, batch.edge_index, batch.edge_attr, batch.batch)
        # only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
            vic_mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
            agg1_mask = batch.agg1_mask.bool()
            agg2_mask = batch.agg2_mask.bool()
            vic_mask = batch.vic_mask.bool()
        
        mask_out = out[mask]
        mask_label = batch.label[mask]
        loss = self.loss_fn(mask_out, mask_label)
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        if not self.vic_only:
            mask_out = out[agg1_mask]
            mask_label = batch.label[agg1_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt1_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
            
            mask_out = out[agg2_mask]
            mask_label = batch.label[agg2_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt2_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        mask_out = out[vic_mask] 
        mask_label = batch.label[vic_mask]
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        vict_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)
        if not self.vic_only:
            self.log('aggt1_acc', aggt1_acc, prog_bar=True, batch_size=self.batch_size)
            self.log('aggt2_acc', aggt2_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vict_acc', vict_acc, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    
class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1


class SGFormerGlitch(pl.LightningModule):
    def __init__(
        self, channels: int, pe_dim: int, num_layers: int, batchsize: int, lr=0.00005
    ):
        super().__init__()
        self.sgformer = SGFormer(
            in_channels=1,
            hidden_channels=channels,
            out_channels=4,  # keep feature dim for MLP
            trans_num_layers=num_layers,
            trans_num_heads=4,
            trans_dropout=0.5,
            gnn_num_layers=3,
            gnn_dropout=0.5,
            graph_weight=0.5,
            aggregate='add'
        )

        self.lr = lr
        self.loss_fn = torch.nn.L1Loss()
        self.batch_size = batchsize

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x = self.sgformer(x, edge_index, batch)
        return x  

    def training_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.pestat_RWSE, batch.edge_index, batch.edge_attr, batch.batch)
        out = out[vic_mask]
        label = batch.label[vic_mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.pestat_RWSE, batch.edge_index, batch.edge_attr, batch.batch)
        mask_out = out[vic_mask]
        mask_label = batch.label[vic_mask]
        loss = self.loss_fn(mask_out, mask_label)

        def compute_accuracy(pred, label):
            mask = label != 0
            pred_masked = pred[mask]
            label_masked = label[mask]
            acc = 100 - torch.mean(torch.abs(pred_masked - label_masked) / torch.abs(label_masked)) * 100
            return acc

        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_p_acc', compute_accuracy(mask_out[:, 0], mask_label[:, 0]), prog_bar=True)
        self.log('tw_p_acc', compute_accuracy(mask_out[:, 1], mask_label[:, 1]), prog_bar=True)
        self.log('vmax_n_acc', compute_accuracy(mask_out[:, 2], mask_label[:, 2]), prog_bar=True)
        self.log('tw_n_acc', compute_accuracy(mask_out[:, 3], mask_label[:, 3]), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model


class SGFormerDelay(pl.LightningModule):
    def __init__(
        self, channels: int, pe_dim: int, num_layers: int, batchsize: int, lr=0.00005, vic_only=False
    ):
        super().__init__()
        self.sgformer = SGFormer(
            in_channels=1,
            hidden_channels=channels,
            out_channels=1,  # keep feature dim for MLP
            trans_num_layers=num_layers,
            trans_num_heads=4,
            trans_dropout=0.5,
            gnn_num_layers=3,
            gnn_dropout=0.5,
            graph_weight=0.5,
            aggregate='add'
        )

        self.lr = lr
        self.loss_fn = torch.nn.L1Loss()
        self.batch_size = batchsize
        self.vic_only = vic_only

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x = self.sgformer(x, edge_index, batch)
        return x      

    def training_step(self, batch, batch_idx):
        # only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
        out = self(batch.x, batch.pestat_RWSE, batch.edge_index, batch.edge_attr, batch.batch)
        out = out[mask]
        label = batch.label[mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.pestat_RWSE, batch.edge_index, batch.edge_attr, batch.batch)
        # only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
            vic_mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
            agg1_mask = batch.agg1_mask.bool()
            agg2_mask = batch.agg2_mask.bool()
            vic_mask = batch.vic_mask.bool()
        
        mask_out = out[mask]
        mask_label = batch.label[mask]
        loss = self.loss_fn(mask_out, mask_label)
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        if not self.vic_only:
            mask_out = out[agg1_mask]
            mask_label = batch.label[agg1_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt1_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
            
            mask_out = out[agg2_mask]
            mask_label = batch.label[agg2_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt2_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        mask_out = out[vic_mask] 
        mask_label = batch.label[vic_mask]
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        vict_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)
        if not self.vic_only:
            self.log('aggt1_acc', aggt1_acc, prog_bar=True, batch_size=self.batch_size)
            self.log('aggt2_acc', aggt2_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vict_acc', vict_acc, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model


class DeepGCNDelay(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        hidden_channels=64,
        conv_layer_num=20,
        batch_size=32,
        lr=0.001,
        dropout=0.1,
        block_type='res+',
        vic_only = False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.loss_fn = nn.L1Loss()
        self.batch_size = batch_size
        self.vic_only = vic_only
        # Input projection to hidden space
        self.input_proj = nn.Linear(in_ch, hidden_channels)

        # Build DeepGCN layers
        self.layers = nn.ModuleList()
        for _ in range(conv_layer_num):
            conv = GCNConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, affine=True)
            act = nn.ReLU()
            self.layers.append(DeepGCNLayer(
                conv=conv,
                norm=norm,
                act=act,
                block=block_type,
                dropout=dropout,
                ckpt_grad=False
            ))

        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_ch)

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_proj(x)

    def training_step(self, batch, batch_idx):
         # only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        out = out[mask]
        label = batch.label[mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        # only v
        if self.vic_only:
            mask = batch.vic_mask.bool()
            vic_mask = batch.vic_mask.bool()
        # 2a1v
        else:
            mask = batch.mask.bool()
            agg1_mask = batch.agg1_mask.bool()
            agg2_mask = batch.agg2_mask.bool()
            vic_mask = batch.vic_mask.bool()
    
        mask_out = out[mask]
        mask_label = batch.label[mask]
        loss = self.loss_fn(mask_out, mask_label)
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        if not self.vic_only:
            mask_out = out[agg1_mask]
            mask_label = batch.label[agg1_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt1_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
            
            mask_out = out[agg2_mask]
            mask_label = batch.label[agg2_mask]
            zeromask = mask_label != 0
            masked_out = mask_out[zeromask]
            masked_label = mask_label[zeromask]
            aggt2_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        mask_out = out[vic_mask] 
        mask_label = batch.label[vic_mask]
        zeromask = mask_label != 0
        masked_out = mask_out[zeromask]
        masked_label = mask_label[zeromask]
        vict_acc = 100 - (torch.mean(torch.abs(masked_out - masked_label) / torch.abs(masked_label)) * 100)
        
        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, prog_bar=True, batch_size=self.batch_size)
        if not self.vic_only:
            self.log('aggt1_acc', aggt1_acc, prog_bar=True, batch_size=self.batch_size)
            self.log('aggt2_acc', aggt2_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vict_acc', vict_acc, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=6e-4)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model

class DeepGCNGlitch(pl.LightningModule):
    def __init__(
        self,
        in_ch,
        out_ch,
        hidden_channels=64,
        conv_layer_num=20,
        batch_size=32,
        lr=0.001,
        dropout=0.1,
        block_type='res+',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.loss_fn = nn.L1Loss()
        self.batch_size = batch_size
        self.input_proj = nn.Linear(in_ch, hidden_channels)

        self.layers = nn.ModuleList()
        for _ in range(conv_layer_num):
            conv = GCNConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, affine=True)
            act = nn.ReLU()
            self.layers.append(DeepGCNLayer(
                conv=conv,
                norm=norm,
                act=act,
                block=block_type,
                dropout=dropout,
                ckpt_grad=False
            ))

        self.output_proj = nn.Linear(hidden_channels, out_ch)

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_proj(x)

    def training_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        out = out[vic_mask]
        label = batch.label[vic_mask]
        loss = self.loss_fn(out, label)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        vic_mask = batch.vic_mask.bool()
        out = self(batch.x, batch.edge_index, batch.edge_attr)
        mask_out = out[vic_mask]
        mask_label = batch.label[vic_mask]
        loss = self.loss_fn(mask_out, mask_label)

        def compute_accuracy(pred, label):
            mask = label != 0
            pred_masked = pred[mask]
            label_masked = label[mask]
            acc = 100 - torch.mean(torch.abs(pred_masked - label_masked) / torch.abs(label_masked)) * 100
            return acc

        vmax_p_acc = compute_accuracy(mask_out[:, 0], mask_label[:, 0])
        tw_p_acc = compute_accuracy(mask_out[:, 1], mask_label[:, 1])
        vmax_n_acc = compute_accuracy(mask_out[:, 2], mask_label[:, 2])
        tw_n_acc = compute_accuracy(mask_out[:, 3], mask_label[:, 3])

        self.log('val_loss', loss, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_p_acc', vmax_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_p_acc', tw_p_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('vmax_n_acc', vmax_n_acc, prog_bar=True, batch_size=self.batch_size)
        self.log('tw_n_acc', tw_n_acc, prog_bar=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=6e-4)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        model = cls(*args, **kwargs)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        return model