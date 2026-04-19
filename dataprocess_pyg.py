import re
import os
import torch
import numpy as  np
import random
import dgl
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import to_dgl
from sklearn.model_selection import train_test_split
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from dgl.dataloading import GraphDataLoader
from utils import process_glitch_wave, extract_glitch_metric
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
import torch_geometric.transforms as T
from position_encoder import compute_posenc_stats
transform = T.AddRandomWalkPE(walk_length=16, attr_name='pe')

# extract all circuits in a segemnt files
def extract_timing(file_path):
    matches = re.findall(r"\d+", file_path)
    segment = int(matches[0])
    timing_pattern = re.compile(
        r'^\s*'                      
        r'[+\-]?\d+(\.\d+)?([eE][+\-]?\d+)?'  
        r'(?:\s*,\s*[+\-]?\d+(\.\d+)?([eE][+\-]?\d+)?)*'  
        r'\s*$'                      
    )
    def extract_cc(meta_str):
        pattern = r'([-+]?\d*\.\d+)(?=f)'
        match = re.search(pattern, meta_str)
        cc = float(match.group())
        return cc
    def convert_seg_2_node(seg_timing):
        def cumulative_sum(values):
            result = []
            current_sum = 0
            for value in values:
                current_sum += value
                result.append(current_sum)
            return result
        timing = cumulative_sum(seg_timing)
        value = 0
        timing.insert(0, value)  # insert timing value 0 at starting node.
        return timing 
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for idx, line in enumerate(lines):
        if timing_pattern.match(line):
            values = [float(token.strip())* 1e+12 for token in line.split(",")]
            timing = {
                    'agg1_node': convert_seg_2_node(values[0: segment]),   #parse out timing wo total segment, will automatically adds up together, convert to L:segment to L:segment+1(num of nodes).
                    'vic_node': convert_seg_2_node(values[segment + 1: 2 * segment + 1]),
                    'agg2_node': convert_seg_2_node(values[2 * (segment + 1): 3 * segment + 2])
                }
            meta_strs = [token.strip() for token in lines[idx - 1].split(",")]
            cc = {
                    'agg1v_seg': [extract_cc(meta_str) for meta_str in meta_strs[0: segment]],
                    'agg2v_seg': [extract_cc(meta_str) for meta_str in meta_strs[2 * segment + 2: 3 * segment + 2]]
                }
    
            data.append((segment, timing, cc))
    return data


def extract_glitch(config, glitch_measurement):
    try:
        match1 = re.search(r"/(\d+)_seg_vic_0_fall_csv/", config)
        match2 = re.search(r"/(\d+)_seg_vic_1_fall_csv/", config)
        try:
            segment = int(match1.group(1))
        except:
            segment = int(match2.group(1))
        pattern = re.compile(r'coupling_capacitance_\d+(?:\.\d+)?_microm_([\d.]+)f')
        ccs = []
        metric = {}
        with open(config, "r") as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    ccs.append(float(match.group(1)))
        cc = {
                'agg1v_seg': [ccs[2*i] for i in range(segment)],
                'agg2v_seg': [ccs[2*i+1] for i in range(segment)]
            }
        data = process_glitch_wave(glitch_measurement)
        victims = [str(204+i*4) for i in range(segment)]
        for victim in victims:
            metric[victim] = extract_glitch_metric(data[victim])
        glitch = {
                    'v_max_p':[metric[victim][0][0] for victim in victims],
                    'tw_p': [metric[victim][0][1] for victim in victims],
                    'v_max_n':[metric[victim][1][0] for victim in victims],
                    'tw_n': [metric[victim][1][1] for victim in victims],
                }
        return (segment, glitch, cc)
    except:
        print(config)
        print('fail extract glitch!')
        return None
    

def mesh_segment_2a1v_pyg(circuit, directed, task):
    segment, label, cc = circuit
    nodes = torch.empty((0, 1), dtype=torch.float)
    edge_attr = torch.empty((0, 2), dtype=torch.float)
    edge_index = torch.empty((2, 0), dtype=torch.int64)
    if task == 'delay':
        labels = torch.empty((0, 1), dtype=torch.float)
    if task == 'glitch':
        labels = torch.empty((0, 4), dtype=torch.float)
    mask = torch.empty(0, dtype=torch.int64)
    agg1_mask = torch.empty(0, dtype=torch.int64)
    agg2_mask = torch.empty(0, dtype=torch.int64)
    vic_mask = torch.empty(0, dtype=torch.int64) 
    wr = 13.5
    wc = 0.15
    # using ground capacitance as node attribute.
    nv = torch.tensor([wc], dtype=torch.float)
    na1 = torch.tensor([wc], dtype=torch.float)
    na2 = torch.tensor([wc], dtype=torch.float)
    units = []
    for i in range(segment+1): # n(nodes) equals to n(segemnts) + 1
        if i == 0 or i == segment:
            unitNodes = torch.stack([nv/2, na1/2, na2/2], dim=0)
        else:
            unitNodes = torch.stack([nv, na1, na2], dim=0)
        if task == 'delay':
            unitLabels =  torch.tensor([[label['vic_node'][i]], [label['agg1_node'][i]], [label['agg2_node'][i]]], dtype=torch.float)
        else:
            if i == 0:
                unitLabels =  torch.tensor([[0,0,0,0], [0,0,0,0], [0,0,0,0]], dtype=torch.float)
            else:
                unitLabels =  torch.tensor([[label['v_max_p'][i-1],label['tw_p'][i-1],label['v_max_n'][i-1],label['tw_n'][i-1]], [0,0,0,0], [0,0,0,0]], dtype=torch.float)
        unitMask = torch.tensor([1, 1, 1], dtype=torch.int64)
        unitVicMask = torch.tensor([1, 0, 0], dtype=torch.int64)
        unitAgg1Mask = torch.tensor([0, 1, 0], dtype=torch.int64)
        unitAgg2Mask = torch.tensor([0, 0, 1], dtype=torch.int64)
        if i != segment:
            if directed:
                unitEdges = torch.tensor([[0, 1, 2, 0, 1, 0, 2],[3, 4, 5, 1, 0, 2, 0]], dtype=torch.int64) + 3*i
                unitEgdeAttrs = torch.cat([torch.cat([torch.ones(3, 1) * wr, torch.zeros(3, 1)], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg1v_seg'][i]], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg2v_seg'][i]], dim=1)], dim=0)
            else:
                unitEdges = torch.tensor([[0, 3, 1, 4, 2, 5, 0, 1, 0, 2],[3, 0, 4, 1, 5, 2, 1, 0, 2, 0]], dtype=torch.int64) + 3*i
                unitEgdeAttrs = torch.cat([torch.cat([torch.ones(6, 1) * wr, torch.zeros(6, 1)], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg1v_seg'][i]], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg2v_seg'][i]], dim=1)], dim=0)
        unit = {
            'nodes': unitNodes,
            'edges': unitEdges,
            'edge_attrs': unitEgdeAttrs,
            'label': unitLabels,
            'mask': unitMask,
            'vmask': unitVicMask,
            'a1mask': unitAgg1Mask,
            'a2mask': unitAgg2Mask
        }
        units.append(unit)
    for unit in units:
        nodes = torch.cat([nodes, unit['nodes']], dim=0)
        edge_index = torch.cat([edge_index, unit['edges']], dim=1)
        edge_attr = torch.cat([edge_attr, unit['edge_attrs']], dim=0)
        labels = torch.cat([labels, unit['label']], dim=0)
        mask = torch.cat([mask, unit['mask']], dim=0)
        agg1_mask = torch.cat([agg1_mask, unit['a1mask']], dim=0)
        agg2_mask = torch.cat([agg2_mask, unit['a2mask']], dim=0)
        vic_mask = torch.cat([vic_mask, unit['vmask']], dim=0)
    return nodes, edge_index, edge_attr, labels, mask, agg1_mask, agg2_mask, vic_mask, segment


def mesh_sink_2a1v_pyg(circuit, directed, task):
    segment, label, cc = circuit
    nodes = torch.empty((0, 1), dtype=torch.float)
    edge_attr = torch.empty((0, 2), dtype=torch.float)
    edge_index = torch.empty((2, 0), dtype=torch.int64)
    if task == 'delay':
        labels = torch.empty((0, 1), dtype=torch.float)
    if task == 'glitch':
        labels = torch.empty((0, 4), dtype=torch.float)
    wr = 13.5
    wc = 0.15
    # using ground capacitance as node attribute.
    nv = torch.tensor([wc], dtype=torch.float)
    na1 = torch.tensor([wc], dtype=torch.float)
    na2 = torch.tensor([wc], dtype=torch.float)
    units = []
    for i in range(segment+1): # n(nodes) equals to n(segemnts) + 1
        if i == 0 or i == segment:
            unitNodes = torch.stack([nv/2, na1/2, na2/2], dim=0)
        else:
            unitNodes = torch.stack([nv, na1, na2], dim=0)
        if task == 'delay':
            unitLabels =  torch.tensor([[label['vic_node'][i]], [label['agg1_node'][i]], [label['agg2_node'][i]]], dtype=torch.float)
        else:
            if i == 0:
                unitLabels =  torch.tensor([[0,0,0,0], [0,0,0,0], [0,0,0,0]], dtype=torch.float)
            else:
                unitLabels =  torch.tensor([[label['v_max_p'][i-1],label['tw_p'][i-1],label['v_max_n'][i-1],label['tw_n'][i-1]], [0,0,0,0], [0,0,0,0]], dtype=torch.float)
        if i != segment:
            if directed:
                unitEdges = torch.tensor([[0, 1, 2, 0, 1, 0, 2],[3, 4, 5, 1, 0, 2, 0]], dtype=torch.int64) + 3*i
                unitEgdeAttrs = torch.cat([torch.cat([torch.ones(3, 1) * wr, torch.zeros(3, 1)], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg1v_seg'][i]], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg2v_seg'][i]], dim=1)], dim=0)
            else:
                unitEdges = torch.tensor([[0, 3, 1, 4, 2, 5, 0, 1, 0, 2],[3, 0, 4, 1, 5, 2, 1, 0, 2, 0]], dtype=torch.int64) + 3*i
                unitEgdeAttrs = torch.cat([torch.cat([torch.ones(6, 1) * wr, torch.zeros(6, 1)], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg1v_seg'][i]], dim=1), torch.cat([torch.zeros(2, 1), torch.ones(2, 1)* cc['agg2v_seg'][i]], dim=1)], dim=0)
        unit = {
            'nodes': unitNodes,
            'edges': unitEdges,
            'edge_attrs': unitEgdeAttrs,
            'label': unitLabels
        }
        units.append(unit)
    for unit in units:
        nodes = torch.cat([nodes, unit['nodes']], dim=0)
        edge_index = torch.cat([edge_index, unit['edges']], dim=1)
        edge_attr = torch.cat([edge_attr, unit['edge_attrs']], dim=0)
        labels = torch.cat([labels, unit['label']], dim=0)
    mask = torch.zeros(nodes.size()[0], dtype=torch.int64)
    agg1_mask = torch.zeros(nodes.size()[0], dtype=torch.int64)
    agg2_mask = torch.zeros(nodes.size()[0], dtype=torch.int64)
    vic_mask = torch.zeros(nodes.size()[0], dtype=torch.int64)
    mask[-3:] = 1 # set to 1 for sink nodes
    agg2_mask[-1] = 1 # set to 1 for sink agg2
    agg1_mask[-2] = 1 # set to 1 for sink agg1
    vic_mask[-3] = 1 # set to 1 for sink agg2
    return nodes, edge_index, edge_attr, labels, mask, agg1_mask, agg2_mask, vic_mask, segment


def compose_mesh_graph_2a1v_pyg(raw_data, directed=False, accumulation=False, timing='segment', task='delay', cc_pattern="as_node", pe=True, pe_type=["RWSE"], cfg= None):
    if task == 'delay':
        files = os.listdir(raw_data) 
        data_list = []
        for file in files:
            file_path = os.path.join(raw_data, file)
            data = extract_timing(file_path)
            for circuit in data:
                if timing == 'segment':
                    nodes, edge_index, edge_attr, labels, mask, agg1_mask, agg2_mask, vic_mask, segment = mesh_segment_2a1v_pyg(circuit, directed, task)
                elif timing == 'sink':
                    nodes, edge_index, edge_attr, labels, mask, agg1_mask, agg2_mask, vic_mask, segment = mesh_sink_2a1v_pyg(circuit, directed, task)
                data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, label=labels, mask=mask, agg1_mask=agg1_mask, agg2_mask=agg2_mask, vic_mask=vic_mask, segment=segment)
                if pe:
                    data = compute_posenc_stats(data, pe_types=pe_type, is_undirected=not directed, cfg=cfg)
                data_list.append(data)
    if task == 'glitch':
        data_list = []
        for dirpath, _, filenames in os.walk(raw_data):
            if filenames:  
                for file in filenames:
                    if "config" in file.lower():  
                        config = os.path.join(dirpath, file)
                        glitch_measurement = os.path.join(dirpath, file).replace("_config","")
                        circuit = extract_glitch(config, glitch_measurement)
                        if circuit is not None:
                            if timing == 'segment':
                                nodes, edge_index, edge_attr, labels, mask, agg1_mask, agg2_mask, vic_mask, segment = mesh_segment_2a1v_pyg(circuit, directed, task)
                            elif timing == 'sink':
                                nodes, edge_index, edge_attr, labels, mask, agg1_mask, agg2_mask, vic_mask, segment = mesh_sink_2a1v_pyg(circuit, directed, task)
                            data = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, label=labels, mask=mask, agg1_mask=agg1_mask, agg2_mask=agg2_mask, vic_mask=vic_mask, segment=segment)
                            if pe:
                                data = compute_posenc_stats(data, pe_types=pe_type, is_undirected=not directed, cfg=cfg)
                            data_list.append(data)
                        
    return data_list


class Mesh2a1vPYG(InMemoryDataset):
    def __init__(self, root, raw_data=None, directed=False, accumulation=False, task='delay', timing='segment', cc_pattern='as_edge', pe=True, pe_type=["RWSE"], cfg=None, transform=None, pre_transform=transform, pre_filter=None):
        root = os.path.join(root, f'Mesh2a1v_cc_{cc_pattern}_{directed}_PYG')
        self.root = root
        self.raw_data = raw_data
        self.directed = directed
        self.accumulation = accumulation
        self.timing = timing
        self.task = task
        self.pe = pe
        self.pe_type = pe_type
        self.cfg = cfg
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.csv')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = compose_mesh_graph_2a1v_pyg(self.raw_data, self.directed, self.accumulation, self.timing, self.task, pe = self.pe, pe_type=self.pe_type, cfg=self.cfg)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]