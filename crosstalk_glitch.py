import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataprocess_dgl import Mesh2a1vDGL, create_dataloaders, collate_cc_as_node, collate_cc_as_edge, collate_graphomer, collate_cc_as_edge_pe
from dataprocess_pyg import Mesh2a1vPYG
from utils import plot_acc_w_segment
from ogb.graphproppred import Evaluator
from model import GNNGlitch, DeepGCNGlitch, GraphomerGlitch, SiGTGlitch, SiGTGlitchPE, GraphGPSGlitch, SGFormerGlitch

# Glitch only for victim nodes.
model_name = 'Graphomer'
task = 'glitch'
mode = 'segment'  # 'segment' or 'sink'
raw_data = f'./{task}'
exp_name = f'{mode}-2A1V-{model_name}'
batch_size=256
num_epochs = 100

if model_name == 'NormGNN':
    root = f'./{task}_dataset/DGL/{mode}/left'
    os.makedirs(root, exist_ok=True)
    dataset = Mesh2a1vPYG(root=root, raw_data=raw_data, directed=False, accumulation=False, task=task, timing=mode, cc_pattern='as_edge')
    train_loader, valid_loader, test_loader = create_dataloaders(dataset, batch_size, 'PYG')
    model = GNNGlitch(in_ch=1, out_ch=4, hidden_channels=128, batch_size=batch_size, gnn_type='gcn', lr=0.002)

elif model_name == 'DeepGCN':
    root = f'./{task}_dataset/DGL/{mode}/left'
    os.makedirs(root, exist_ok=True)
    dataset = Mesh2a1vPYG(root=root, raw_data=raw_data, directed=False, accumulation=False, task=task, timing=mode, cc_pattern='as_edge')
    train_loader, valid_loader, test_loader = create_dataloaders(dataset, batch_size, 'PYG')
    model = DeepGCNGlitch(in_ch=1,out_ch=4,hidden_channels=64,conv_layer_num=20,batch_size=batch_size,lr=0.001,dropout=0.1,block_type='res+',)

elif model_name == 'Graphomer':
    root = f'./{task}_dataset/DGL/segment/left'
    os.makedirs(root, exist_ok=True)
    dataset = Mesh2a1vDGL(root=root, raw_data=raw_data, directed=False, accumulation=False, task=task, timing='segment', cc_pattern='as_edge')
    train_loader, valid_loader, test_loader = create_dataloaders(dataset, batch_size, 'DGL', collate_fn=collate_graphomer)
    total_updates = 33000 * num_epochs / batch_size
    warmup_updates = total_updates * 0.16
    model = GraphomerGlitch(total_updates=total_updates, warmup_updates=warmup_updates, lr=1e-4, weight_decay=0.0, eps=1e-8, batch_size=batch_size, mode=mode)

elif model_name == "SGFormer":
    root = f'./{task}_dataset/DGL/{mode}/left_pe_5'
    os.makedirs(root, exist_ok=True)
    cfg = {
        'posenc_RWSE': {
            'enable': True,
            'kernel': {
                'times_func': range(1, 17),
                'times': list(range(1, 17)),  
            },
            'model': 'Linear',
            'dim_pe': 16,
            'raw_norm_type': 'BatchNorm'
        }
    }
    dataset = Mesh2a1vPYG(root=root, raw_data=raw_data, directed=False, accumulation=False, task=task, timing=mode, cc_pattern='as_edge', pe=True, pe_type=["RWSE"], cfg=cfg)
    train_loader, valid_loader, test_loader = create_dataloaders(dataset, batch_size, 'PYG')
    model = SGFormerGlitch(channels=64, pe_dim=16, num_layers=4, batchsize=batch_size)

elif model_name == 'GraphGPS':
    #Support PE: 'LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE'
    # pe_type can be a combination, e.g. 'eigen+rw_landing'
    root = f'./{task}_dataset/DGL/{mode}/left_EqlapPE8'
    os.makedirs(root, exist_ok=True)
    cfg = {
        "posenc_EquivStableLapPE": {
            "enable": True,
            "eigen": {
                "laplacian_norm": "none",
                "eigvec_norm": "L2",
                "max_freqs": 8
            },
            "raw_norm_type": "none"
        }
    }
    dataset = Mesh2a1vPYG(root=root, raw_data=raw_data, directed=False, accumulation=False, task=task, timing={mode}, cc_pattern='as_edge', pe=True, pe_type=["EquivStableLapPE"], cfg=cfg)
    train_loader, valid_loader, test_loader = create_dataloaders(dataset, batch_size, 'PYG')
    model = GraphGPSGlitch(channels=64, input_dim=8, pe_dim=16, num_layers=10, attn_type='multihead', batchsize=batch_size)

elif model_name == 'SiGT':
    root = f'./{task}_dataset/DGL/segment/left'
    os.makedirs(root, exist_ok=True)
    dataset = Mesh2a1vDGL(root=root, raw_data=raw_data, directed=False, accumulation=False, task=task, timing='segment', cc_pattern='as_edge')
    train_loader, valid_loader, test_loader = create_dataloaders(dataset, batch_size, 'DGL', collate_fn=collate_cc_as_edge)
    total_updates = 33000 * num_epochs / batch_size
    warmup_updates = total_updates * 0.16
    model = SiGTGlitch(node_encoding='gnn', total_updates=total_updates, warmup_updates=warmup_updates, lr=8e-4, weight_decay=0.0, eps=1e-8, batch_size=batch_size, mode=mode)



logger = TensorBoardLogger(f'{task}_logs/left', name=exp_name)

trainer = Trainer(
    max_epochs=num_epochs,
    logger=logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=-1 if torch.cuda.is_available() else None,
    strategy=DDPStrategy(find_unused_parameters=True),
    precision="16-mixed",
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=3,
            mode="min"
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ]
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

