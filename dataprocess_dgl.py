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
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from utils import process_glitch_wave, extract_glitch_metric
import torch.nn.functional as F
from torch.utils.data import Dataset
import gc

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
        print('fail extract glitch!')
        return None
    

def mesh_segment_2a1v_cc_on_edge_dgl(circuit, directed, task):
    segment, label, cc = circuit
    wr = 13.5
    wc = 0.15

    all_node_feats = []    
    all_node_labels = []  
    src_list = []
    dst_list = []
    edge_feats = []

    mask_list = []
    vic_mask_list = []
    agg1_mask_list = []
    agg2_mask_list = []

    for i in range(segment + 1):
        if i == 0 or i == segment:
            node_vic  = wc / 2
            node_agg1 = wc / 2
            node_agg2 = wc / 2
        else:
            node_vic  = wc
            node_agg1 = wc
            node_agg2 = wc
        all_node_feats.append([node_vic])
        all_node_feats.append([node_agg1])
        all_node_feats.append([node_agg2])
        if task == 'delay':
            all_node_labels.append([label['vic_node'][i]])
            all_node_labels.append([label['agg1_node'][i]])
            all_node_labels.append([label['agg2_node'][i]])
        else:
            if i == 0:
                all_node_labels.append([0,0,0,0])
                all_node_labels.append([0,0,0,0])
                all_node_labels.append([0,0,0,0])
            else:
                all_node_labels.append([label['v_max_p'][i-1],label['tw_p'][i-1],label['v_max_n'][i-1],label['tw_n'][i-1]])
                all_node_labels.append([0,0,0,0])
                all_node_labels.append([0,0,0,0])
        vic_mask_list.extend([1, 0, 0])
        agg1_mask_list.extend([0, 1, 0])
        agg2_mask_list.extend([0, 0, 1])
        mask_list.extend([1, 1, 1])

        if i != segment:
            if directed:
                raw_src = [0, 1, 2, 0, 1, 0, 2]
                raw_dst = [3, 4, 5, 1, 0, 2, 0]
                edge_feats += [[wr, 0.0]] * 3
            else:
                raw_src = [0, 3, 1, 4, 2, 5, 0, 1, 0, 2]
                raw_dst = [3, 0, 4, 1, 5, 2, 1, 0, 2, 0]
                edge_feats += [[wr, 0.0]] * 6
            src_list.extend(list(map(lambda x: x + 3*i, raw_src)))
            dst_list.extend(list(map(lambda x: x + 3*i, raw_dst)))
            edge_feats += [[0.0, cc['agg1v_seg'][i]]] * 2
            edge_feats += [[0.0, cc['agg2v_seg'][i]]] * 2
    node_feats_tensor  = torch.tensor(all_node_feats, dtype=torch.float)
    node_label_tensor  = torch.tensor(all_node_labels, dtype=torch.float)
    src_tensor         = torch.tensor(src_list, dtype=torch.int64)
    dst_tensor         = torch.tensor(dst_list, dtype=torch.int64)
    edge_feats_tensor  = torch.tensor(edge_feats, dtype=torch.float)

    mask_tensor        = torch.tensor(mask_list, dtype=torch.int64)
    vic_mask_tensor    = torch.tensor(vic_mask_list, dtype=torch.int64)
    agg1_mask_tensor   = torch.tensor(agg1_mask_list, dtype=torch.int64)
    agg2_mask_tensor   = torch.tensor(agg2_mask_list, dtype=torch.int64)

    g = dgl.graph((src_tensor, dst_tensor))
    g.ndata['feat'] = node_feats_tensor    
    g.ndata['label'] = node_label_tensor  
    g.ndata['mask'] = mask_tensor
    g.ndata['vic_mask'] = vic_mask_tensor
    g.ndata['agg1_mask'] = agg1_mask_tensor
    g.ndata['agg2_mask'] = agg2_mask_tensor
    g.edata['edge_attr'] = edge_feats_tensor
    spd, path = dgl.shortest_dist(g, root=None, return_paths=True) # shortest path distance and path
    g.ndata["spd"] = spd
    g.ndata["path"] = path
    # pack graph-level attributes into labels.
    # 1.mask for sinks.
    # max_node_num = 64
    max_node_num = 80 # For OOD
    net_num = 3
    node_num = g.number_of_nodes()
    sink = torch.zeros(node_num+net_num, dtype=torch.int64)
    sink_agg1 = torch.zeros(node_num+net_num, dtype=torch.int64)
    sink_agg2 = torch.zeros(node_num+net_num, dtype=torch.int64)
    sink_vic = torch.zeros(node_num+net_num, dtype=torch.int64)
    sink[-3:] = 1
    sink_agg2[-1] = 1
    sink_agg1[-2] = 1
    sink_vic[-3] = 1 
    sink = F.pad(sink, (0, max_node_num-node_num), mode='constant', value=-1)
    sink_agg2 = F.pad(sink_agg2, (0, max_node_num-node_num), mode='constant', value=-1)
    sink_agg1 = F.pad(sink_agg1, (0, max_node_num-node_num), mode='constant', value=-1)
    sink_vic = F.pad(sink_vic, (0, max_node_num-node_num), mode='constant', value=-1)

    # 2.InterNet A, IntraNet A
    agg1_A = torch.zeros(node_num + net_num, node_num + net_num) # add for <net> tokens
    agg2_A = torch.zeros(node_num + net_num, node_num + net_num)
    vic_A = torch.zeros(node_num + net_num, node_num + net_num)
    agg1_net = []
    agg2_net = []
    vic_net = []
    for i in range(segment+1):
        m = 3*i + net_num # add for <net> tokens
        vic_net.append(m)
        agg1_net.append(m+1)
        agg2_net.append(m+2)
    vic_A = compose_net_A(g, vic_net, vic_A, net_num)
    agg1_A = compose_net_A(g, agg1_net, agg1_A, net_num)
    agg2_A = compose_net_A(g, agg2_net, agg2_A, net_num)
    net_A = vic_A + agg1_A + agg2_A
    padding = (0, max_node_num - node_num, 0, max_node_num - node_num)
    net_A = F.pad(net_A, padding, mode='constant', value=0)
    net_A = net_A.unsqueeze(-1)

    couple_A = torch.zeros(node_num + net_num, node_num + net_num)
    for i in range(segment):
        m = 3*i + net_num # add for <net> tokens
        vic = m
        agg1 = m+1
        agg2 = m+2
        couple_A[vic, agg1] = 1
        couple_A[agg1, vic] = 1
        couple_A[vic, agg2] = 1
        couple_A[agg2, vic] = 1
    couple_A = replace_1s_as_edge(g, couple_A, net_num)
    couple_A = F.pad(couple_A, padding, mode='constant', value=0)
    couple_A = couple_A.unsqueeze(-1)

    # 3. Attn_mask for <net> tokens. In default, <Net1> is for victim, <Net2> ... starts for agg1
    Net_Attn_mask = torch.zeros(node_num + net_num, node_num + net_num)
    Net_Attn_mask[:net_num, :] = 1
    for i in range(segment+1):
        vic = 3*i + net_num # add for <net> tokens
        agg1 = vic + 1
        agg2 = vic + 2
        Net_Attn_mask[0, vic] = 0
        Net_Attn_mask[1, agg1] = 0
        Net_Attn_mask[2, agg2] = 0
    Net_Attn_mask = F.pad(Net_Attn_mask, padding, mode='constant', value=0)
    label = {}
    label['sink'] = sink
    label['sink_agg2'] = sink_agg2
    label['sink_agg1'] = sink_agg1
    label['sink_vic'] = sink_vic
    label['net_A'] = net_A
    label['couple_A'] = couple_A
    label['net_attn_mask'] = Net_Attn_mask
        
    return g, label


def mesh_segment_2a1v_cc_as_node_dgl(circuit, directed, task):
    segment, label, cc = circuit
    wr = 13.5
    wc = 0.15

    all_node_feats = []    
    all_node_labels = []  
    src_list = []
    dst_list = []
    edge_feats = []

    mask_list = []
    vic_mask_list = []
    agg1_mask_list = []
    agg2_mask_list = []
    sink_list = []
    vic_sink_mask_list = []
    agg1_sink_mask_list = []
    agg2_sink_mask_list = []

    # Add n nodes, and n-n edges
    for i in range(segment + 1):
        if i == 0 or i == segment:
            node_vic  = wc / 2
            node_agg1 = wc / 2
            node_agg2 = wc / 2
        else:
            node_vic  = wc
            node_agg1 = wc
            node_agg2 = wc
        all_node_feats.append([node_vic])
        all_node_feats.append([node_agg1])
        all_node_feats.append([node_agg2])
        if task == 'delay':
            all_node_labels.append([label['vic_node'][i]])
            all_node_labels.append([label['agg1_node'][i]])
            all_node_labels.append([label['agg2_node'][i]])
        else:
            if i == 0:
                all_node_labels.append([0,0,0,0])
                all_node_labels.append([0,0,0,0])
                all_node_labels.append([0,0,0,0])
            else:
                all_node_labels.append([label['v_max_p'][i-1],label['tw_p'][i-1],label['v_max_n'][i-1],label['tw_n'][i-1]])
                all_node_labels.append([0,0,0,0])
                all_node_labels.append([0,0,0,0])
        vic_mask_list.extend([1, 0, 0])
        agg1_mask_list.extend([0, 1, 0])
        agg2_mask_list.extend([0, 0, 1])
        mask_list.extend([1, 1, 1])
        if i != segment:
            sink_list.extend([0, 0, 0])
            vic_sink_mask_list.extend([0, 0, 0])
            agg1_sink_mask_list.extend([0, 0, 0])
            agg2_sink_mask_list.extend([0, 0, 0])
        else:
            sink_list.extend([1, 1, 1])
            vic_sink_mask_list.extend([1, 0, 0])
            agg1_sink_mask_list.extend([0, 1, 0])
            agg2_sink_mask_list.extend([0, 0, 1])

        if i != segment:
            if directed:
                raw_src = [0, 1, 2]
                raw_dst = [3, 4, 5]
                edge_feats += [[wr]] * 3
            else:
                raw_src = [0, 3, 1, 4, 2, 5]
                raw_dst = [3, 0, 4, 1, 5, 2]
                edge_feats += [[wr]] * 6
            src_list.extend(list(map(lambda x: x + 3*i, raw_src)))
            dst_list.extend(list(map(lambda x: x + 3*i, raw_dst)))
            

    # Add c nodes, and n-c edges
    for i in range(segment):
        n_va1 = torch.tensor([0,1], dtype=torch.int64) + 3 * i
        n_va2 = torch.tensor([0,2], dtype=torch.int64) + 3 * i
        cc_va1_idx = torch.ones(2, dtype=torch.int64) *  len(all_node_feats)
        cc_va2_idx = torch.ones(2, dtype=torch.int64) * (len(all_node_feats)+1)
        raw_src = torch.cat([n_va1, cc_va1_idx, n_va2, cc_va2_idx], dim=0)
        raw_dst = torch.cat([cc_va1_idx, n_va1, cc_va2_idx, n_va2], dim=0)

        src_list.extend(raw_src.tolist())
        dst_list.extend(raw_dst.tolist())
        edge_feats += [[0]] * 8 # zero-resistance in e-c edges

        node_cc_agg1v = cc['agg1v_seg'][i]
        node_cc_agg2v = cc['agg2v_seg'][i]

        all_node_feats.append([node_cc_agg1v])
        all_node_feats.append([node_cc_agg2v])
        if task == 'delay':
            all_node_labels.append([0])
            all_node_labels.append([0])
        else:
            all_node_labels.append([0,0,0,0])
            all_node_labels.append([0,0,0,0])
        vic_mask_list.extend([0, 0])
        agg1_mask_list.extend([0, 0])
        agg2_mask_list.extend([0, 0])
        mask_list.extend([0, 0])

        vic_sink_mask_list.extend([0, 0])
        agg1_sink_mask_list.extend([0, 0])
        agg2_sink_mask_list.extend([0, 0])
        sink_list.extend([0, 0])

    node_feats_tensor  = torch.tensor(all_node_feats, dtype=torch.float)
    node_label_tensor  = torch.tensor(all_node_labels, dtype=torch.float)
    src_tensor         = torch.tensor(src_list, dtype=torch.int64)
    dst_tensor         = torch.tensor(dst_list, dtype=torch.int64)
    edge_feats_tensor  = torch.tensor(edge_feats, dtype=torch.float)

    mask_tensor        = torch.tensor(mask_list, dtype=torch.int64)
    vic_mask_tensor    = torch.tensor(vic_mask_list, dtype=torch.int64)
    agg1_mask_tensor   = torch.tensor(agg1_mask_list, dtype=torch.int64)
    agg2_mask_tensor   = torch.tensor(agg2_mask_list, dtype=torch.int64)

    sink_tensor        = torch.tensor(sink_list, dtype=torch.int64)
    vic_sink_mask_tensor    = torch.tensor(vic_sink_mask_list, dtype=torch.int64)
    agg1_sink_mask_tensor   = torch.tensor(agg1_sink_mask_list, dtype=torch.int64)
    agg2_sink_mask_tensor   = torch.tensor(agg2_sink_mask_list, dtype=torch.int64)

    g = dgl.graph((src_tensor, dst_tensor))
    g.ndata['feat'] = node_feats_tensor    
    g.ndata['label'] = node_label_tensor  
    g.ndata['mask'] = mask_tensor
    g.ndata['vic_mask'] = vic_mask_tensor
    g.ndata['agg1_mask'] = agg1_mask_tensor
    g.ndata['agg2_mask'] = agg2_mask_tensor
    g.ndata['sink'] = sink_tensor
    g.ndata['vic_sink_mask'] = vic_sink_mask_tensor
    g.ndata['agg1_sink_mask'] = agg1_sink_mask_tensor
    g.ndata['agg2_sink_mask'] = agg2_sink_mask_tensor
    g.edata['edge_attr'] = edge_feats_tensor
    g.ndata['segment'] = torch.full((g.num_nodes(),), fill_value=segment, dtype=torch.long)
    return g


def compose_mesh_graph_2a1v_dgl(raw_data, directed=False, accumulation=False, task='delay', timing='segment', cc_pattern='as_node'):
    if task == 'delay':
        files = os.listdir(raw_data)
        graph_list = []
        graph_labels = {}
        for file in files:
            file_path = os.path.join(raw_data, file)
            data = extract_timing(file_path) 
            for circuit in data:
                if timing == 'segment':
                    if cc_pattern == 'as_node':
                        g = mesh_segment_2a1v_cc_as_node_dgl(circuit, directed, task)
                    else:
                        g, label = mesh_segment_2a1v_cc_on_edge_dgl(circuit, directed, task)
                else:
                    raise ValueError(f"Unknown timing method: {timing}")
                graph_list.append(g)
                for key, value in label.item():
                    if key not in graph_labels:
                        graph_labels[key] = value.unsqueeze(0)
                    else:
                        graph_labels[key] = torch.cat([graph_labels[key], value], dim=0)

    if task == 'glitch':
        graph_list = []
        graph_labels = {}
        for dirpath, _, filenames in os.walk(raw_data):
            if filenames:  
                for file in filenames:
                    if "config" in file.lower():  
                        config = os.path.join(dirpath, file)
                        glitch_measurement = os.path.join(dirpath, file).replace("_config","")
                        circuit = extract_glitch(config, glitch_measurement)
                        if circuit is not None:
                            if timing == 'segment':
                                if cc_pattern == 'as_node':
                                    g = mesh_segment_2a1v_cc_as_node_dgl(circuit, directed, task)
                                else:
                                    g, label = mesh_segment_2a1v_cc_on_edge_dgl(circuit, directed, task)
                            else:
                                raise ValueError(f"Unknown timing method: {timing}")
                            graph_list.append(g)
                            for key, value in label.items():
                                if key not in graph_labels:
                                    graph_labels[key] = value.unsqueeze(0)
                                else:
                                    graph_labels[key] = torch.cat([graph_labels[key], value.unsqueeze(0)], dim=0)
            
    return graph_list, graph_labels


def get_all_circuits(raw_data, task):
    circuit_data = []
    if task == 'delay':
        files = os.listdir(raw_data)
        for file in files:
            file_path = os.path.join(raw_data, file)
            data = extract_timing(file_path)
            circuit_data.extend(data)
    if task == 'glitch':
        for dirpath, _, filenames in os.walk(raw_data):
            if filenames:
                for file in sorted(filenames):  # Sort filenames alphabetically
                    if "config" in file.lower():
                        config = os.path.join(dirpath, file)
                        glitch_measurement = os.path.join(dirpath, file).replace("_config", "")
                        data = extract_glitch(config, glitch_measurement)
                        if data is not None:
                            print(file, data)
                            circuit_data.extend([data])

    return circuit_data


def batch_process_graphs(root, raw_data, batch_size=1000, directed=False, accumulation=False, task='delay', timing='segment', cc_pattern='as_node'):
    output_dir = os.path.join(root, f'Mesh2a1v_cc_{cc_pattern}_{directed}_DGL')
    os.makedirs(output_dir, exist_ok=True)

    all_circuits = get_all_circuits(raw_data, task) 
    total_circuits = len(all_circuits)
    print(total_circuits)
    for batch_idx in range(0, total_circuits, batch_size):
        batch_graphs = []
        batch_labels = {}
        
        end_idx = min(batch_idx + batch_size, total_circuits)
        print(f"Processing batch {batch_idx//batch_size + 1}: circuits {batch_idx} to {end_idx-1}")
        
        for i in range(batch_idx, end_idx):
            circuit = all_circuits[i]
            g, label = mesh_segment_2a1v_cc_on_edge_dgl(circuit, directed, task)
            batch_graphs.append(g)
            for key, value in label.items():
                if key not in batch_labels:
                    batch_labels[key] = value.unsqueeze(0)
                else:
                    batch_labels[key] = torch.cat([batch_labels[key], value.unsqueeze(0)], dim=0)
        
        # Save this batch
        batch_file = os.path.join(output_dir, f"batch_{batch_idx//batch_size}.dgl")
        save_graphs(batch_file, batch_graphs, batch_labels)
        
        # Clear memory
        del batch_graphs, batch_labels
        gc.collect()  # Force garbage collection


class Mesh2a1vDGL(Dataset):
    def __init__(self, root, raw_data, batch_size=1000, directed=False, accumulation=False, task='delay', timing='segment', cc_pattern='as_node'):
        self.batch_dir = os.path.join(root, f'Mesh2a1v_cc_{cc_pattern}_{directed}_DGL')
        if not os.path.exists(self.batch_dir):
            batch_process_graphs(root, raw_data, batch_size, directed, accumulation, task, timing, cc_pattern)
        self.batch_files = sorted([f for f in os.listdir(self.batch_dir) if f.endswith('.dgl')])
        self.all_graphs = []
        self.all_labels = {}
        
        for batch_file in self.batch_files:
            batch_path = os.path.join(self.batch_dir, batch_file)
            print(f"Loading batch file: {batch_path}")
            graphs, labels = load_graphs(batch_path)
            self.all_graphs.extend(graphs)
            if not self.all_labels:
                for key in labels.keys():
                    self.all_labels[key] = []
            for key, values in labels.items():
                self.all_labels[key].extend(values)
    
    def __getitem__(self, idx):
        graph = self.all_graphs[idx]
        graph_labels = {key: values[idx] for key, values in self.all_labels.items()}
        return graph, graph_labels
    
    def __len__(self):
        return len(self.all_graphs)
            

def dset_split(dataset, dataset_type=None, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Splits 'dataset' into train/valid/test subsets (70/20/10 by default).
    Returns three lists of data objects/tuples: (train, valid, test).
    """
    random.seed(seed)
    
    if dataset_type == 'PYG':
        # For PyG, .len() and .get(i)
        length = dataset.len()
        print(length)
        get_item = dataset.get
    elif dataset_type == 'DGL':
        # For DGL, .__len__() and .__getitem__(i)
        length = dataset.__len__()
        print(length)
        get_item = dataset.__getitem__
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'PYG' or 'DGL'.")

    # Shuffle indices
    # indices = list(range(length))
    # random.shuffle(indices)

    # # Compute the split boundaries
    # train_end = int(train_ratio * length)
    # val_end   = int((train_ratio + val_ratio) * length)

    # # Slice into train / valid / test
    # tr_dset = [get_item(i) for i in indices[:train_end]]
    # va_dset = [get_item(i) for i in indices[train_end:val_end]]
    # te_dset = [get_item(i) for i in indices[val_end:]]
    indices = list(range(length))
    random.shuffle(indices)

    # Compute the split boundaries
    train_end = int(train_ratio * length)
    val_end = int((train_ratio + val_ratio) * length)

    # Create index lists for each split
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create subset datasets based on the type
    if dataset_type == 'PYG':
        from torch.utils.data import Subset
        tr_dset = Subset(dataset, train_indices)
        va_dset = Subset(dataset, val_indices)
        te_dset = Subset(dataset, test_indices)
    elif dataset_type == 'DGL':
        from torch.utils.data import Subset
        tr_dset = Subset(dataset, train_indices)
        va_dset = Subset(dataset, val_indices)
        te_dset = Subset(dataset, test_indices)

    return tr_dset, va_dset, te_dset


def create_dataloaders(dataset, batch_size=256, dataset_type=None, collate_fn=None):
    # Split the dataset
    tr_dset, va_dset, te_dset = dset_split(dataset, dataset_type=dataset_type)

    if dataset_type == 'PYG':
        train_loader = DataLoader(tr_dset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(va_dset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader  = DataLoader(te_dset, batch_size=1, shuffle=False, drop_last=True)

    elif dataset_type == 'DGL':
        train_loader = GraphDataLoader(tr_dset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
        valid_loader = GraphDataLoader(va_dset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader  = GraphDataLoader(te_dset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


def compose_net_A(g, directed_node_list, A, net_num, attr_name='edge_attr'):
    """
    Vectorized version of compose_net_A:
      - Gather all (i, j) pairs in one pass.
      - For each pair, gather all edges (k, k+3) for k in range(i, j, 3).
      - Make one call to `edge_ids(...)`.
      - Sum each pair's chunk of attributes, compute 1.0 / total, and store in A.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    directed_node_list : list of int
        Node indices (1-based in your original code) that define the net segments.
    A : torch.Tensor
        The (max_num_nodes+1) x (max_num_nodes+1) adjacency to fill in.
    attr_name : str
        Key for edge attributes in g.edata.

    Returns
    -------
    A : torch.Tensor
        Updated adjacency matrix with all pairs filled in.
    """

    # 1) Gather all (i, j) pairs.
    #    Also note we use i'-1 for the DGL lookups, but you'll store
    #    the result at i', j' in A (adjust carefully as needed).
    pair_i = []
    pair_j = []

    k_offsets = [0]  # prefix-sum trick

    # For building one big list of edges:
    all_src = []
    all_dst = []

    for idx_i in range(len(directed_node_list)):
        for idx_j in range(idx_i + 1, len(directed_node_list)):
            i_val = directed_node_list[idx_i] - net_num  # 0-based for the graph
            j_val = directed_node_list[idx_j] - net_num  # 0-based for the graph

            pair_i.append(directed_node_list[idx_i])  # for placing in A
            pair_j.append(directed_node_list[idx_j])  

            # 2) For each pair, gather k in [i_val, j_val) stepping by 3
            k_range = range(i_val, j_val, 3)
            # Build up the big list of edges for a single `edge_ids` call
            for k in k_range:
                all_src.append(k)
                all_dst.append(k + 3)

            # Keep track of how many (k, k+3) edges we just appended
            k_offsets.append(k_offsets[-1] + len(k_range))

    # Convert all_src, all_dst to torch Tensors
    all_src = torch.tensor(all_src, dtype=torch.long)
    all_dst = torch.tensor(all_dst, dtype=torch.long)

    if len(all_src) == 0:
        # Edge case: if no edges at all, just return A as-is
        return A

    # 3) Make one call to gather *all* edge IDs
    all_edge_ids = g.edge_ids(all_src, all_dst)

    # 4) Gather attributes for these edges
    all_edge_vals = g.edata[attr_name][all_edge_ids]
    all_edge_vals = all_edge_vals.squeeze(-1)  # if needed

    # 5) Chunk these attributes back to each pair.
    #    k_offsets tells us how to slice all_edge_vals for each pair.
    for idx in range(len(pair_i)):
        start = k_offsets[idx]
        end   = k_offsets[idx + 1]

        # Extract the slice of edges that belongs to (pair_i[idx], pair_j[idx])
        pair_edge_vals = all_edge_vals[start:end]

        total = pair_edge_vals.sum()
        if total == 0:
            value = 0.0
        else:
            value = 1.0 / total
        # 6) Place the result in A, using 1-based indices or 0-based 
        i_row = pair_i[idx]
        j_col = pair_j[idx]
        # A[i_row, j_col] = value # symmetrical
        A[j_col, i_row] = value  

    return A

def get_edge_attribute(g, i, j, attr_name='edge_attr'):
    """
    Retrieve the edge attribute between two nodes in a DGL graph.
    Efficiently fetches the attribute for an edge if it exists.

    Parameters:
    - g: DGL graph
    - i: int or tensor, source node(s)
    - j: int or tensor, destination node(s)
    - attr_name: str, the name of the edge attribute field

    Returns:
    - Attribute value if the edge exists, otherwise None for those missing edges.
    """
    edge_ids = g.edge_ids(i, j, return_uv=True)[2]
    if edge_ids.nelement() == 0:
        return f"No edge exists between node {i} and node {j}"
    attr_value = g.edata[attr_name][edge_ids]
    return attr_value

def replace_1s_as_edge(g, A, net_num):
    """
    Updates elements in tensor A where values are 1 using edge attributes from graph g.

    Parameters:
    - g: DGL graph
    - A: Tensor, typically an adjacency matrix or similar
    - attn_type: string indicating which attribute to use

    Returns:
    - Updated A tensor
    """
    # Get indices where A == 1
    indices = (A == 1).nonzero(as_tuple=False)
    if indices.size(0) > 0:
        src, dst = indices[:, 0], indices[:, 1]
        edge_attrs = get_edge_attribute(g, src-net_num, dst-net_num)
        A[indices[:, 0], indices[:, 1]] = edge_attrs[:, 1]

    return A

def build_couple_A(g, max_num_nodes, cc_0, segment):
    """
    Vectorized construction of 'couple_A' for one graph 'g'.
    - g.ndata['feat'] is assumed to be of shape [num_nodes, feat_dim] or [num_nodes].
    - cc_0, segment, etc. same logic as your original code.
    - Returns couple_A of shape (max_num_nodes+1, max_num_nodes+1, feat_dim).
      If your node features are 1D scalars, feat_dim=1.
    """

    couple_idx0 = [cc_0 + 2*i     for i in range(segment)]
    couple_idx1 = [cc_0 + 2*i + 1 for i in range(segment)]

    couple_idx0 = torch.tensor(couple_idx0, device=g.device)
    couple_idx1 = torch.tensor(couple_idx1, device=g.device)

    feats0 = g.ndata['feat'][couple_idx0]  # shape [segment, feat_dim]
    feats1 = g.ndata['feat'][couple_idx1]  # shape [segment, feat_dim]

    # Prepare the final couple_A, storing in the last dimension
    couple_A = torch.zeros(
        (max_num_nodes + 1, max_num_nodes + 1, feats0.shape[-1]),
        device=g.device
    )

    # Build index pairs for assignment:
    #   For each i in range(segment):
    #     vic  = 3*i + 1
    #     agg1 = vic + 1
    #     agg2 = vic + 2
    #
    # We assign feats0[i] to (vic, agg1) and (agg1, vic)
    # We assign feats1[i] to (vic, agg2) and (agg2, vic)

    pairs_0 = []  # (vic, agg1) and (agg1, vic)
    pairs_1 = []  # (vic, agg2) and (agg2, vic)

    for i in range(segment):
        vic  = 3*i + 1
        agg1 = vic + 1
        agg2 = vic + 2
        pairs_0.append((vic,  agg1))
        pairs_0.append((agg1, vic ))
        pairs_1.append((vic,  agg2))
        pairs_1.append((agg2, vic ))

    pairs_0 = torch.tensor(pairs_0, device=g.device)  # shape [2*segment, 2]
    pairs_1 = torch.tensor(pairs_1, device=g.device)  # shape [2*segment, 2]

    feats0_rep = feats0.repeat_interleave(2, dim=0)
    feats1_rep = feats1.repeat_interleave(2, dim=0)

    couple_A[pairs_0[:, 0], pairs_0[:, 1]] = feats0_rep
    couple_A[pairs_1[:, 0], pairs_1[:, 1]] = feats1_rep

    return couple_A

def collate_cc_as_edge(graphs):
    net_A_list = []
    couple_A_list = []
    net_attn_mask_list = []
    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g, _ in graphs]
    max_num_nodes = max(num_nodes)

    sink_mask = []
    sink_vic_mask = []
    sink_agg1_mask = []
    sink_agg2_mask = []
    
    g_list = []
    sg_list = []
    net_num = 3

    # decompose graph to mesh subgraphs.
    for g, _ in graphs:
        node_num = g.number_of_nodes()
        segment = int(node_num/3)-1
        g_copy = g.clone()
        g_copy.ndata['feat'] = g.ndata['feat'].clone()
        g_copy.edata['edge_attr'] = g.edata['edge_attr'].clone()
        try:
            del g_copy.ndata['spd']
            del g_copy.ndata['path']
        except:
            pass
        g_list.append(g_copy)
        for i in range(segment):
            m = 3*i #victim
            vic_agg1_mesh_nodes = [m, m+1, m+3, m+4] 
            vic_agg1_mesh_target_mask = [0, 0, 0, 1]
            
            vic_agg2_mesh_nodes = [m, m+2, m+3, m+5]
            vic_agg2_mesh_target_mask = [0, 0, 0, 1]
            
            vic_agg1_agg2_mesh_nodes = [m, m+1, m+2, m+3, m+4, m+5]  
            vic_agg1_agg2_mesh_target_mask = [0, 0, 0, 1, 0, 0]
            
            vic_agg1_mesh = g_copy.subgraph(vic_agg1_mesh_nodes)
            vic_agg1_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_mesh_target_mask)
            
            vic_agg1_agg2_mesh = g_copy.subgraph(vic_agg1_agg2_mesh_nodes)
            vic_agg1_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_agg2_mesh_target_mask)
            
            vic_agg2_mesh = g_copy.subgraph(vic_agg2_mesh_nodes)
            vic_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg2_mesh_target_mask)

            try:
                del vic_agg1_agg2_mesh.ndata['spd']
                del vic_agg1_agg2_mesh.ndata['path']
                del vic_agg1_mesh.ndata['spd']
                del vic_agg1_mesh.ndata['path']
                del vic_agg2_mesh.ndata['spd']
                del vic_agg2_mesh.ndata['path']
            except:
                pass
            sg_list.append(vic_agg1_agg2_mesh)
            sg_list.append(vic_agg1_mesh)
            sg_list.append(vic_agg2_mesh)
            
    bg = dgl.batch(g_list)
    bsg = dgl.batch(sg_list)

    #1.compose mask for sinks; 2.compose InterNet A, IntraNet A; 3.spd, path encoding
    for g, label in graphs:
        node_num = g.number_of_nodes()
        segment = int(node_num/3)-1
        sink_mask.append(label['sink'][net_num:node_num+net_num])
        sink_vic_mask.append(label['sink_vic'][net_num:node_num+net_num])
        sink_agg1_mask.append(label['sink_agg1'][net_num:node_num+net_num])
        sink_agg2_mask.append(label['sink_agg2'][net_num:node_num+net_num])
        net_A = label['net_A'][:max_num_nodes+net_num,:max_num_nodes+net_num,:]
        net_A_list.append(net_A)
        couple_A = label['couple_A'][:max_num_nodes+net_num,:max_num_nodes+net_num,:]
        couple_A_list.append(couple_A)
        net_attn_mask = label['net_attn_mask'][:max_num_nodes+net_num,:max_num_nodes+net_num]
        net_attn_mask_list.append(net_attn_mask)
    attn_net_A = torch.stack(net_A_list, dim=0)
    attn_couple_A = torch.stack(couple_A_list, dim=0)
    net_token_attn_mask = torch.stack(net_attn_mask_list, dim=0)
    attn_mask = torch.zeros(num_graphs, max_num_nodes + net_num, max_num_nodes + net_num)
    node_feat = []
    node_label = []
    node_mask = []
    node_vic_mask = []
    node_agg1_mask = []
    node_agg2_mask = []
    path_data = []
    # -1 indicate unreachable node.
    # use -1 padding as well.
    dist = -torch.ones(
        (num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long
    )

    for i in range(num_graphs):
        # A binary mask where invalid positions are indicated by True.
        attn_mask[i, :, num_nodes[i] + net_num :] = 1

        # +1 to distinguish padded non-existing nodes from real nodes
        node_feat.append(graphs[i][0].ndata["feat"])
        node_label.append(graphs[i][0].ndata["label"])
        node_mask.append(graphs[i][0].ndata["mask"])
        node_vic_mask.append(graphs[i][0].ndata["vic_mask"])
        node_agg1_mask.append(graphs[i][0].ndata["agg1_mask"])
        node_agg2_mask.append(graphs[i][0].ndata["agg2_mask"])
        # Path padding to make all paths to the same length "max_len".
        path = graphs[i][0].ndata["path"]
        path_len = path.size(dim=2)
        # shape of shortest_path: [n, n, max_len]
        max_len = 5 # Hyperparameter
        if path_len >= max_len:
            shortest_path = path[:, :, :max_len]
        else:
            p1d = (0, max_len - path_len)
            shortest_path = torch.nn.functional.pad(path, p1d, "constant", -1)
        pad_num_nodes = max_num_nodes - num_nodes[i]
        p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes) # pad to [max_num_nodes, max_num_nodes, max_len]
        shortest_path = torch.nn.functional.pad(shortest_path, p3d, "constant", -1)
        edata = graphs[i][0].edata["edge_attr"]

        edata = torch.cat(
            (edata, torch.zeros(1, edata.shape[1]).to(edata.device)), dim=0
        )
        path_data.append(edata[shortest_path])

        dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i][0].ndata["spd"]

    # node feat padding
    node_feat = torch.nn.utils.rnn.pad_sequence(node_feat, padding_value=-1, batch_first=True)
    node_label = torch.nn.utils.rnn.pad_sequence(node_label, padding_value=-1, batch_first=True)
    node_mask = torch.nn.utils.rnn.pad_sequence(node_mask, padding_value=-1, batch_first=True)
    node_vic_mask = torch.nn.utils.rnn.pad_sequence(node_vic_mask, padding_value=-1, batch_first=True)
    node_agg1_mask = torch.nn.utils.rnn.pad_sequence(node_agg1_mask, padding_value=-1, batch_first=True)
    node_agg2_mask = torch.nn.utils.rnn.pad_sequence(node_agg2_mask, padding_value=-1, batch_first=True)
    sink_mask = torch.nn.utils.rnn.pad_sequence(sink_mask, padding_value=-1, batch_first=True)
    sink_vic_mask = torch.nn.utils.rnn.pad_sequence(sink_vic_mask, padding_value=-1, batch_first=True)
    sink_agg1_mask = torch.nn.utils.rnn.pad_sequence(sink_agg1_mask, padding_value=-1, batch_first=True)
    sink_agg2_mask = torch.nn.utils.rnn.pad_sequence(sink_agg2_mask, padding_value=-1, batch_first=True)


    return (
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
        torch.stack(path_data), # shape [max_num_nodes, max_num_nodes, max_len, edge_attr_dim]
        dist,
        attn_net_A,
        attn_couple_A
    )
    
def collate_cc_ablation(graphs):
    net_A_list = []
    couple_A_list = []
    net_attn_mask_list = []
    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g, _ in graphs]
    max_num_nodes = max(num_nodes)

    sink_mask = []
    sink_vic_mask = []
    sink_agg1_mask = []
    sink_agg2_mask = []
    
    g_list = []
    sg_list = []
    net_num = 3

    # decompose graph to mesh subgraphs.
    for g, _ in graphs:
        node_num = g.number_of_nodes()
        segment = int(node_num/3)-1
        g_copy = g.clone()
        g_copy.ndata['feat'] = g.ndata['feat'].clone()
        g_copy.edata['edge_attr'] = g.edata['edge_attr'].clone()
        try:
            del g_copy.ndata['spd']
            del g_copy.ndata['path']
        except:
            pass
        g_list.append(g_copy)
        for i in range(segment):
            m = 3*i #victim
            vic_agg1_mesh_nodes = [m, m+1, m+3, m+4] 
            vic_agg1_mesh_target_mask = [0, 0, 0, 1]
            
            vic_agg2_mesh_nodes = [m, m+2, m+3, m+5]
            vic_agg2_mesh_target_mask = [0, 0, 0, 1]
            
            vic_agg1_agg2_mesh_nodes = [m, m+1, m+2, m+3, m+4, m+5]  
            vic_agg1_agg2_mesh_target_mask = [0, 0, 0, 1, 0, 0]
            
            vic_agg1_mesh = g_copy.subgraph(vic_agg1_mesh_nodes)
            vic_agg1_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_mesh_target_mask)
            
            vic_agg1_agg2_mesh = g_copy.subgraph(vic_agg1_agg2_mesh_nodes)
            vic_agg1_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_agg2_mesh_target_mask)
            
            vic_agg2_mesh = g_copy.subgraph(vic_agg2_mesh_nodes)
            vic_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg2_mesh_target_mask)

            try:
                del vic_agg1_agg2_mesh.ndata['spd']
                del vic_agg1_agg2_mesh.ndata['path']
                del vic_agg1_mesh.ndata['spd']
                del vic_agg1_mesh.ndata['path']
                del vic_agg2_mesh.ndata['spd']
                del vic_agg2_mesh.ndata['path']
            except:
                pass
            sg_list.append(vic_agg1_agg2_mesh)
            sg_list.append(vic_agg1_mesh)
            sg_list.append(vic_agg2_mesh)
            
    bg = dgl.batch(g_list)
    bsg = dgl.batch(sg_list)

    #1.compose mask for sinks; 2.compose InterNet A, IntraNet A; 3.spd, path encoding
    for g, label in graphs:
        node_num = g.number_of_nodes()
        segment = int(node_num/3)-1
        sink_mask.append(label['sink'][net_num:node_num+net_num])
        sink_vic_mask.append(label['sink_vic'][net_num:node_num+net_num])
        sink_agg1_mask.append(label['sink_agg1'][net_num:node_num+net_num])
        sink_agg2_mask.append(label['sink_agg2'][net_num:node_num+net_num])
        net_A = label['net_A'][:max_num_nodes+net_num,:max_num_nodes+net_num,:]
        net_A_list.append(net_A)
        couple_A = label['couple_A'][:max_num_nodes+net_num,:max_num_nodes+net_num,:]
        couple_A_list.append(couple_A)
        net_attn_mask = label['net_attn_mask'][:max_num_nodes+net_num,:max_num_nodes+net_num]
        net_attn_mask_list.append(net_attn_mask)
    attn_net_A = torch.stack(net_A_list, dim=0)
    attn_couple_A = torch.stack(couple_A_list, dim=0)
    net_token_attn_mask = torch.stack(net_attn_mask_list, dim=0)
    attn_mask = torch.zeros(num_graphs, max_num_nodes + net_num, max_num_nodes + net_num)
    node_feat = []
    node_label = []
    node_mask = []
    node_vic_mask = []
    node_agg1_mask = []
    node_agg2_mask = []
    in_degree, out_degree = [], []
    path_data = []
    # -1 indicate unreachable node.
    # use -1 padding as well.
    dist = -torch.ones(
        (num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long
    )

    for i in range(num_graphs):
        # A binary mask where invalid positions are indicated by True.
        # Avoid the case where all positions are invalid.
        attn_mask[i, :, num_nodes[i] + net_num :] = 1

        # +1 to distinguish padded non-existing nodes from real nodes
        node_feat.append(graphs[i][0].ndata["feat"])
        node_label.append(graphs[i][0].ndata["label"])
        node_mask.append(graphs[i][0].ndata["mask"])
        node_vic_mask.append(graphs[i][0].ndata["vic_mask"])
        node_agg1_mask.append(graphs[i][0].ndata["agg1_mask"])
        node_agg2_mask.append(graphs[i][0].ndata["agg2_mask"])
        # 0 for padding
        in_degree.append(
            torch.clamp(graphs[i][0].in_degrees() + 1, min=0, max=512)
        )
        out_degree.append(
            torch.clamp(graphs[i][0].out_degrees() + 1, min=0, max=512)
        )

        # Path padding to make all paths to the same length "max_len".
        path = graphs[i][0].ndata["path"]
        path_len = path.size(dim=2)
        # shape of shortest_path: [n, n, max_len]
        max_len = 5 # Hyperparameter
        if path_len >= max_len:
            shortest_path = path[:, :, :max_len]
        else:
            p1d = (0, max_len - path_len)
            # Use the same -1 padding as shortest_dist for
            # invalid edge IDs.
            shortest_path = torch.nn.functional.pad(path, p1d, "constant", -1)
        pad_num_nodes = max_num_nodes - num_nodes[i]
        p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes) # pad to [max_num_nodes, max_num_nodes, max_len]
        shortest_path = torch.nn.functional.pad(shortest_path, p3d, "constant", -1)
        edata = graphs[i][0].edata["edge_attr"]

        # shortest_dist pads non-existing edges (at the end of shortest
        # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
        # for all padded edge features.
        edata = torch.cat(
            (edata, torch.zeros(1, edata.shape[1]).to(edata.device)), dim=0
        )
        path_data.append(edata[shortest_path])

        dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i][0].ndata["spd"]

    # node feat padding
    node_feat = torch.nn.utils.rnn.pad_sequence(node_feat, padding_value=-1, batch_first=True)
    node_label = torch.nn.utils.rnn.pad_sequence(node_label, padding_value=-1, batch_first=True)
    node_mask = torch.nn.utils.rnn.pad_sequence(node_mask, padding_value=-1, batch_first=True)
    node_vic_mask = torch.nn.utils.rnn.pad_sequence(node_vic_mask, padding_value=-1, batch_first=True)
    node_agg1_mask = torch.nn.utils.rnn.pad_sequence(node_agg1_mask, padding_value=-1, batch_first=True)
    node_agg2_mask = torch.nn.utils.rnn.pad_sequence(node_agg2_mask, padding_value=-1, batch_first=True)
    sink_mask = torch.nn.utils.rnn.pad_sequence(sink_mask, padding_value=-1, batch_first=True)
    sink_vic_mask = torch.nn.utils.rnn.pad_sequence(sink_vic_mask, padding_value=-1, batch_first=True)
    sink_agg1_mask = torch.nn.utils.rnn.pad_sequence(sink_agg1_mask, padding_value=-1, batch_first=True)
    sink_agg2_mask = torch.nn.utils.rnn.pad_sequence(sink_agg2_mask, padding_value=-1, batch_first=True)

    # degree padding
    in_degree = torch.nn.utils.rnn.pad_sequence(in_degree, batch_first=True)
    out_degree = torch.nn.utils.rnn.pad_sequence(out_degree, batch_first=True)

    return (
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
        torch.stack(path_data), # shape [max_num_nodes, max_num_nodes, max_len, edge_attr_dim]
        dist,
        attn_net_A,
        attn_couple_A
    )

def collate_cc_as_node(graphs):
    net_A_list = []
    couple_A_list = []
    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g, _ in graphs]
    max_num_nodes = max(num_nodes)
    
    g_list = []
    sg_list = []
    for g, _ in graphs:
        node_num = g.number_of_nodes()
        segment = int((node_num-3)/5)
        cc_0 = 3*(segment+1)
        g_copy = g.clone()
        g_copy.ndata['feat'] = g.ndata['feat'].clone()
        g_copy.edata['edge_attr'] = g.edata['edge_attr'].clone()
        try:
            del g_copy.ndata['spd']
            del g_copy.ndata['path']
        except:
            pass
        g_list.append(g_copy)
        for i in range(segment):
            m = 3*i #victim
            cc = cc_0 + 2*i
            if i != segment -1:
                vic_agg1_mesh_nodes = [m, m+1, m+3, m+4, cc, cc+2] 
                vic_agg1_mesh_target_mask = [0, 0, 0, 1, 0, 0]
                
                vic_agg2_mesh_nodes = [m, m+2, m+3, m+5, cc+1, cc+3,]
                vic_agg2_mesh_target_mask = [0, 0, 0, 1, 0, 0]
                
                vic_agg1_agg2_mesh_nodes = [m, m+1, m+2, m+3, m+4, m+5, cc, cc+1, cc+2, cc+3]  
                vic_agg1_agg2_mesh_target_mask = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            else:
                vic_agg1_mesh_nodes = [m, m+1, m+3, m+4, cc] 
                vic_agg1_mesh_target_mask = [0, 0, 0, 1, 0]
                
                vic_agg2_mesh_nodes = [m, m+2, m+3, m+5, cc+1]
                vic_agg2_mesh_target_mask = [0, 0, 0, 1, 0]
                
                vic_agg1_agg2_mesh_nodes = [m, m+1, m+2, m+3, m+4, m+5, cc, cc+1]  
                vic_agg1_agg2_mesh_target_mask = [0, 0, 0, 1, 0, 0, 0, 0]

            vic_agg1_mesh = g_copy.subgraph(vic_agg1_mesh_nodes)
            vic_agg1_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_mesh_target_mask)
            
            vic_agg1_agg2_mesh = g_copy.subgraph(vic_agg1_agg2_mesh_nodes)
            vic_agg1_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_agg2_mesh_target_mask)
            
            vic_agg2_mesh = g_copy.subgraph(vic_agg2_mesh_nodes)
            vic_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg2_mesh_target_mask)

            try:
                del vic_agg1_agg2_mesh.ndata['spd']
                del vic_agg1_agg2_mesh.ndata['path']
                del vic_agg1_mesh.ndata['spd']
                del vic_agg1_mesh.ndata['path']
                del vic_agg2_mesh.ndata['spd']
                del vic_agg2_mesh.ndata['path']
            except:
                pass
            sg_list.append(vic_agg1_agg2_mesh)
            sg_list.append(vic_agg1_mesh)
            sg_list.append(vic_agg2_mesh)
            
    bg = dgl.batch(g_list)
    bsg = dgl.batch(sg_list)
    for g, _ in graphs:
        node_num = g.number_of_nodes()
        segment = int((node_num-3)/5)
        cc_0 = 3*(segment+1)
        zero_A = torch.zeros(max_num_nodes + 1, max_num_nodes + 1)
        # clock-wise directed adjency matrix for nets  
        agg1_A = zero_A.clone()
        agg2_A = zero_A.clone()
        vic_A = zero_A.clone()
        agg1_net = []
        agg2_net = []
        vic_net = []
        for i in range(segment+1):
            m = 3*i + 1 # add 1 for graph token
            vic_net.append(m)
            agg1_net.append(m+1)
            agg2_net.append(m+2)
        vic_A = compose_net_A(g, vic_net, vic_A)
        agg1_A = compose_net_A(g, agg1_net, agg1_A)
        agg2_A = compose_net_A(g, agg2_net, agg2_A)
        net_A = vic_A + agg1_A + agg2_A
        # net_A = torch.stack([agg1_A, agg2_A, vic_A], dim=-1) # NxNx3
        net_A_list.append(net_A.unsqueeze(-1))

        couple_A = build_couple_A(g, max_num_nodes, cc_0, segment)
        couple_A_list.append(couple_A.unsqueeze(-1)  # or not, depends on your shape
                            if couple_A.ndim == 2 else
                            couple_A)
        spd, path = dgl.shortest_dist(g, root=None, return_paths=True) # shortest path distance and path
        g.ndata["spd"] = spd
        g.ndata["path"] = path
    attn_net_A = torch.stack(net_A_list, dim=0)
    attn_couple_A = torch.stack(couple_A_list, dim=0)
    
    
    attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
    node_feat = []
    node_label = []
    node_mask = []
    node_vic_mask = []
    node_agg1_mask = []
    node_agg2_mask = []
    sink_mask = []
    sink_vic_mask = []
    sink_agg1_mask = []
    sink_agg2_mask = []
    in_degree, out_degree = [], []
    path_data = []
    # -1 indicate unreachable node.
    # use -1 padding as well.
    dist = -torch.ones(
        (num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long
    )

    for i in range(num_graphs):
        # A binary mask where invalid positions are indicated by True.
        # Avoid the case where all positions are invalid.
        attn_mask[i, :, num_nodes[i] + 1 :] = 1

        # +1 to distinguish padded non-existing nodes from real nodes
        node_feat.append(graphs[i][0].ndata["feat"])
        node_label.append(graphs[i][0].ndata["label"])
        node_mask.append(graphs[i][0].ndata["mask"])
        node_vic_mask.append(graphs[i][0].ndata["vic_mask"])
        node_agg1_mask.append(graphs[i][0].ndata["agg1_mask"])
        node_agg2_mask.append(graphs[i][0].ndata["agg2_mask"])
        # 0 for padding
        in_degree.append(
            torch.clamp(graphs[i][0].in_degrees() + 1, min=0, max=512)
        )
        out_degree.append(
            torch.clamp(graphs[i][0].out_degrees() + 1, min=0, max=512)
        )

        # Path padding to make all paths to the same length "max_len".
        path = graphs[i][0].ndata["path"]
        path_len = path.size(dim=2)
        # shape of shortest_path: [n, n, max_len]
        max_len = 5 # Hyperparameter
        if path_len >= max_len:
            shortest_path = path[:, :, :max_len]
        else:
            p1d = (0, max_len - path_len)
            # Use the same -1 padding as shortest_dist for
            # invalid edge IDs.
            shortest_path = torch.nn.functional.pad(path, p1d, "constant", -1)
        pad_num_nodes = max_num_nodes - num_nodes[i]
        p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes) # pad to [max_num_nodes, max_num_nodes, max_len]
        shortest_path = torch.nn.functional.pad(shortest_path, p3d, "constant", -1)
        edata = graphs[i][0].edata["edge_attr"]

        # shortest_dist pads non-existing edges (at the end of shortest
        edata = torch.cat(
            (edata, torch.zeros(1, edata.shape[1]).to(edata.device)), dim=0
        )
        path_data.append(edata[shortest_path])

        dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i][0].ndata["spd"]

    # node feat padding
    node_feat = torch.nn.utils.rnn.pad_sequence(node_feat, padding_value=-1, batch_first=True)
    node_label = torch.nn.utils.rnn.pad_sequence(node_label, padding_value=-1, batch_first=True)
    node_mask = torch.nn.utils.rnn.pad_sequence(node_mask, padding_value=-1, batch_first=True)
    node_vic_mask = torch.nn.utils.rnn.pad_sequence(node_vic_mask, padding_value=-1, batch_first=True)
    node_agg1_mask = torch.nn.utils.rnn.pad_sequence(node_agg1_mask, padding_value=-1, batch_first=True)
    node_agg2_mask = torch.nn.utils.rnn.pad_sequence(node_agg2_mask, padding_value=-1, batch_first=True)

    sink_mask = torch.nn.utils.rnn.pad_sequence(sink_mask, padding_value=-1, batch_first=True)
    sink_vic_mask = torch.nn.utils.rnn.pad_sequence(sink_vic_mask, padding_value=-1, batch_first=True)
    sink_agg1_mask = torch.nn.utils.rnn.pad_sequence(sink_agg1_mask, padding_value=-1, batch_first=True)
    sink_agg2_mask = torch.nn.utils.rnn.pad_sequence(sink_agg2_mask, padding_value=-1, batch_first=True)

    # degree padding
    in_degree = torch.nn.utils.rnn.pad_sequence(in_degree, batch_first=True)
    out_degree = torch.nn.utils.rnn.pad_sequence(out_degree, batch_first=True)

    return (
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
        torch.stack(path_data), # shape [max_num_nodes, max_num_nodes, max_len, edge_attr_dim]
        dist,
        attn_net_A,
        attn_couple_A
    )


def collate_graphomer(graphs):
    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g, _ in graphs]
    max_num_nodes = max(num_nodes)

    sink_mask = []
    sink_vic_mask = []
    sink_agg1_mask = []
    sink_agg2_mask = []
    
    for g, label in graphs:
        node_num = g.number_of_nodes()
        sink_mask.append(label['sink'][3:node_num+3])
        sink_vic_mask.append(label['sink_vic'][3:node_num+3])
        sink_agg1_mask.append(label['sink_agg1'][3:node_num+3])
        sink_agg2_mask.append(label['sink_agg2'][3:node_num+3])

    attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
    node_feat = []
    node_label = []
    node_mask = []
    node_vic_mask = []
    node_agg1_mask = []
    node_agg2_mask = []
    in_degree, out_degree = [], []
    path_data = []
    # -1 indicate unreachable node.
    # use -1 padding as well.
    dist = -torch.ones(
        (num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long
    )

    for i in range(num_graphs):
        # A binary mask where invalid positions are indicated by True.
        # Avoid the case where all positions are invalid.
        attn_mask[i, :, num_nodes[i] + 1 :] = 1

        # +1 to distinguish padded non-existing nodes from real nodes
        node_feat.append(graphs[i][0].ndata["feat"])
        node_label.append(graphs[i][0].ndata["label"])
        node_mask.append(graphs[i][0].ndata["mask"])
        node_vic_mask.append(graphs[i][0].ndata["vic_mask"])
        node_agg1_mask.append(graphs[i][0].ndata["agg1_mask"])
        node_agg2_mask.append(graphs[i][0].ndata["agg2_mask"])

        # 0 for padding
        in_degree.append(
            torch.clamp(graphs[i][0].in_degrees() + 1, min=0, max=512)
        )
        out_degree.append(
            torch.clamp(graphs[i][0].out_degrees() + 1, min=0, max=512)
        )

        # Path padding to make all paths to the same length "max_len".
        path = graphs[i][0].ndata["path"]
        path_len = path.size(dim=2)
        # shape of shortest_path: [n, n, max_len]
        max_len = 5 # Hyperparameter
        if path_len >= max_len:
            shortest_path = path[:, :, :max_len]
        else:
            p1d = (0, max_len - path_len)
            # Use the same -1 padding as shortest_dist for
            # invalid edge IDs.
            shortest_path = torch.nn.functional.pad(path, p1d, "constant", -1)
        pad_num_nodes = max_num_nodes - num_nodes[i]
        p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes) # pad to [max_num_nodes, max_num_nodes, max_len]
        shortest_path = torch.nn.functional.pad(shortest_path, p3d, "constant", -1)
        edata = graphs[i][0].edata["edge_attr"]

        # shortest_dist pads non-existing edges (at the end of shortest
        # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
        # for all padded edge features.
        edata = torch.cat(
            (edata, torch.zeros(1, edata.shape[1]).to(edata.device)), dim=0
        )
        path_data.append(edata[shortest_path])

        dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i][0].ndata["spd"]

    # node feat padding
    node_feat = torch.nn.utils.rnn.pad_sequence(node_feat, padding_value=-1, batch_first=True)
    node_label = torch.nn.utils.rnn.pad_sequence(node_label, padding_value=-1, batch_first=True)
    node_mask = torch.nn.utils.rnn.pad_sequence(node_mask, padding_value=-1, batch_first=True)
    node_vic_mask = torch.nn.utils.rnn.pad_sequence(node_vic_mask, padding_value=-1, batch_first=True)
    node_agg1_mask = torch.nn.utils.rnn.pad_sequence(node_agg1_mask, padding_value=-1, batch_first=True)
    node_agg2_mask = torch.nn.utils.rnn.pad_sequence(node_agg2_mask, padding_value=-1, batch_first=True)
    sink_mask = torch.nn.utils.rnn.pad_sequence(sink_mask, padding_value=-1, batch_first=True)
    sink_vic_mask = torch.nn.utils.rnn.pad_sequence(sink_vic_mask, padding_value=-1, batch_first=True)
    sink_agg1_mask = torch.nn.utils.rnn.pad_sequence(sink_agg1_mask, padding_value=-1, batch_first=True)
    sink_agg2_mask = torch.nn.utils.rnn.pad_sequence(sink_agg2_mask, padding_value=-1, batch_first=True)

    # # degree padding
    in_degree = torch.nn.utils.rnn.pad_sequence(in_degree, batch_first=True)
    out_degree = torch.nn.utils.rnn.pad_sequence(out_degree, batch_first=True)

    return (
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
        torch.stack(path_data), # shape [max_num_nodes, max_num_nodes, max_len, edge_attr_dim]
        dist
    )


def collate_cc_as_edge_pe(graphs):
    net_A_list = []
    couple_A_list = []
    net_attn_mask_list = []
    num_graphs = len(graphs)
    num_nodes = [g.num_nodes() for g, _ in graphs]
    max_num_nodes = max(num_nodes)

    sink_mask = []
    sink_vic_mask = []
    sink_agg1_mask = []
    sink_agg2_mask = []
    pe_list = []
    g_list = []
    sg_list = []
    net_num = 3

    # decompose graph to mesh subgraphs.
    for g, _ in graphs:
        node_num = g.number_of_nodes()
        segment = int(node_num/3)-1
        g_copy = g.clone()
        g_copy.ndata['feat'] = g.ndata['feat'].clone()
        g_copy.edata['edge_attr'] = g.edata['edge_attr'].clone()
        try:
            del g_copy.ndata['spd']
            del g_copy.ndata['path']
        except:
            pass
        g_list.append(g_copy)
        for i in range(segment):
            m = 3*i #victim
            vic_agg1_mesh_nodes = [m, m+1, m+3, m+4] 
            vic_agg1_mesh_target_mask = [0, 0, 0, 1]
            
            vic_agg2_mesh_nodes = [m, m+2, m+3, m+5]
            vic_agg2_mesh_target_mask = [0, 0, 0, 1]
            
            vic_agg1_agg2_mesh_nodes = [m, m+1, m+2, m+3, m+4, m+5]  
            vic_agg1_agg2_mesh_target_mask = [0, 0, 0, 1, 0, 0]
            
            vic_agg1_mesh = g_copy.subgraph(vic_agg1_mesh_nodes)
            vic_agg1_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_mesh_target_mask)
            
            vic_agg1_agg2_mesh = g_copy.subgraph(vic_agg1_agg2_mesh_nodes)
            vic_agg1_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg1_agg2_mesh_target_mask)
            
            vic_agg2_mesh = g_copy.subgraph(vic_agg2_mesh_nodes)
            vic_agg2_mesh.ndata['target_mask'] = torch.tensor(vic_agg2_mesh_target_mask)

            try:
                del vic_agg1_agg2_mesh.ndata['spd']
                del vic_agg1_agg2_mesh.ndata['path']
                del vic_agg1_mesh.ndata['spd']
                del vic_agg1_mesh.ndata['path']
                del vic_agg2_mesh.ndata['spd']
                del vic_agg2_mesh.ndata['path']
            except:
                pass
            sg_list.append(vic_agg1_agg2_mesh)
            sg_list.append(vic_agg1_mesh)
            sg_list.append(vic_agg2_mesh)
            
    bg = dgl.batch(g_list)
    bsg = dgl.batch(sg_list)

    #1.compose mask for sinks; 2.compose InterNet A, IntraNet A; 3.spd, path encoding
    for g, label in graphs:
        pe = dgl.random_walk_pe(g, 16)
        # pe_lape = dgl.lap_pe(g, 8, padding=True)
        # pe = torch.cat([pe_rwse, pe_lape], dim=1)
        pe_list.append(pe)
        node_num = g.number_of_nodes()
        segment = int(node_num/3)-1
        sink_mask.append(label['sink'][net_num:node_num+net_num])
        sink_vic_mask.append(label['sink_vic'][net_num:node_num+net_num])
        sink_agg1_mask.append(label['sink_agg1'][net_num:node_num+net_num])
        sink_agg2_mask.append(label['sink_agg2'][net_num:node_num+net_num])
        net_A = label['net_A'][:max_num_nodes+net_num,:max_num_nodes+net_num,:]
        net_A_list.append(net_A)
        couple_A = label['couple_A'][:max_num_nodes+net_num,:max_num_nodes+net_num,:]
        couple_A_list.append(couple_A)
        net_attn_mask = label['net_attn_mask'][:max_num_nodes+net_num,:max_num_nodes+net_num]
        net_attn_mask_list.append(net_attn_mask)
    attn_net_A = torch.stack(net_A_list, dim=0)
    attn_couple_A = torch.stack(couple_A_list, dim=0)
    net_token_attn_mask = torch.stack(net_attn_mask_list, dim=0)
    attn_mask = torch.zeros(num_graphs, max_num_nodes + net_num, max_num_nodes + net_num)
    node_feat = []
    node_label = []
    node_mask = []
    node_vic_mask = []
    node_agg1_mask = []
    node_agg2_mask = []
    # in_degree, out_degree = [], []
    path_data = []
    # -1 indicate unreachable node.
    # use -1 padding as well.
    dist = -torch.ones(
        (num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long
    )

    for i in range(num_graphs):
        # A binary mask where invalid positions are indicated by True.
        # Avoid the case where all positions are invalid.
        attn_mask[i, :, num_nodes[i] + net_num :] = 1

        # +1 to distinguish padded non-existing nodes from real nodes
        node_feat.append(graphs[i][0].ndata["feat"])
        node_label.append(graphs[i][0].ndata["label"])
        node_mask.append(graphs[i][0].ndata["mask"])
        node_vic_mask.append(graphs[i][0].ndata["vic_mask"])
        node_agg1_mask.append(graphs[i][0].ndata["agg1_mask"])
        node_agg2_mask.append(graphs[i][0].ndata["agg2_mask"])
        # # 0 for padding
        # in_degree.append(
        #     torch.clamp(graphs[i][0].in_degrees() + 1, min=0, max=512)
        # )
        # out_degree.append(
        #     torch.clamp(graphs[i][0].out_degrees() + 1, min=0, max=512)
        # )

        # Path padding to make all paths to the same length "max_len".
        path = graphs[i][0].ndata["path"]
        path_len = path.size(dim=2)
        # shape of shortest_path: [n, n, max_len]
        max_len = 5 # Hyperparameter
        if path_len >= max_len:
            shortest_path = path[:, :, :max_len]
        else:
            p1d = (0, max_len - path_len)
            # Use the same -1 padding as shortest_dist for
            # invalid edge IDs.
            shortest_path = torch.nn.functional.pad(path, p1d, "constant", -1)
        pad_num_nodes = max_num_nodes - num_nodes[i]
        p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes) # pad to [max_num_nodes, max_num_nodes, max_len]
        shortest_path = torch.nn.functional.pad(shortest_path, p3d, "constant", -1)
        edata = graphs[i][0].edata["edge_attr"]

        # shortest_dist pads non-existing edges (at the end of shortest
        # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
        # for all padded edge features.
        edata = torch.cat(
            (edata, torch.zeros(1, edata.shape[1]).to(edata.device)), dim=0
        )
        path_data.append(edata[shortest_path])

        dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i][0].ndata["spd"]

    # node feat padding
    node_feat = torch.nn.utils.rnn.pad_sequence(node_feat, padding_value=-1, batch_first=True)
    node_label = torch.nn.utils.rnn.pad_sequence(node_label, padding_value=-1, batch_first=True)
    node_mask = torch.nn.utils.rnn.pad_sequence(node_mask, padding_value=-1, batch_first=True)
    node_vic_mask = torch.nn.utils.rnn.pad_sequence(node_vic_mask, padding_value=-1, batch_first=True)
    node_agg1_mask = torch.nn.utils.rnn.pad_sequence(node_agg1_mask, padding_value=-1, batch_first=True)
    node_agg2_mask = torch.nn.utils.rnn.pad_sequence(node_agg2_mask, padding_value=-1, batch_first=True)
    sink_mask = torch.nn.utils.rnn.pad_sequence(sink_mask, padding_value=-1, batch_first=True)
    sink_vic_mask = torch.nn.utils.rnn.pad_sequence(sink_vic_mask, padding_value=-1, batch_first=True)
    sink_agg1_mask = torch.nn.utils.rnn.pad_sequence(sink_agg1_mask, padding_value=-1, batch_first=True)
    sink_agg2_mask = torch.nn.utils.rnn.pad_sequence(sink_agg2_mask, padding_value=-1, batch_first=True)
    pe = torch.nn.utils.rnn.pad_sequence(pe_list, padding_value=-1, batch_first=True)
    # # degree padding
    # in_degree = torch.nn.utils.rnn.pad_sequence(in_degree, batch_first=True)
    # out_degree = torch.nn.utils.rnn.pad_sequence(out_degree, batch_first=True)

    return (
        pe,
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
        # in_degree,
        # out_degree,
        attn_mask,
        net_token_attn_mask,
        torch.stack(path_data), # shape [max_num_nodes, max_num_nodes, max_len, edge_attr_dim]
        dist,
        attn_net_A,
        attn_couple_A
    )


