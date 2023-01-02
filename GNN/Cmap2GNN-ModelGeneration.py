import os
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio import SeqIO

from torch_geometric.data import Data
from torch_geometric.data import DataLoader

# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# read the AF_helix_sheet for the labels
AF_helix_file_path = '../Datasets/AF_helix_sheet.tsv'
AF_helix = pd.read_csv(AF_helix_file_path, sep='\t', header=0, index_col=0)
labels = AF_helix['label'] == 'Alpha helix'
labels = labels.astype(int) # Alpha helix - 1, Beta strand - 0
# print(labels['A0A1D8PQG0'])

dir = '../Datasets/pdbfiles/'

def get_protein_coord(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    CA_pos_x = []
    CA_pos_y = []
    CA_pos_z = []
    for x in range(len(residues)):
        pos = residues[x]["CA"].get_coord()
        CA_pos_x.append(pos[0])
        CA_pos_y.append(pos[1])
        CA_pos_z.append(pos[2])

    distances = np.zeros((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            dist = np.linalg.norm(one-two)
            if dist < 25:
                distances[x, y] = dist
            

    return seqs[0], CA_pos_x, CA_pos_y, CA_pos_z, distances

# dataset construction

sum_cmap_path = "../FeatureEngineering/contact_maps_thresh_1/"

# create a dictionary to map amino acids to a certain number
aa2num = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 
        'E':5, 'Q':6, 'G':7, 'H':8, 'I':9,
        'L':10, 'K':11, 'M':12, 'F':13, 'P':14,
        'S':15, 'T':16, 'W':17, 'Y':18, 'V':19}
num_aa = 20

data_paths = os.listdir(sum_cmap_path)
dataset = []
for path in data_paths:

    # print(path)

    with open(sum_cmap_path + path, 'r') as f:
        adjacency = np.loadtxt(f, dtype=int)
        adjacency_tensor = torch.tensor(adjacency)

    # node features - amino acids in ASCII + node position
    filename = os.fsdecode(dir + path[:-5] + '.pdb')
    pdbfile = os.path.abspath(filename)
    seq, CA_coords_x, CA_coords_y, CA_coords_z, distance_matrix = get_protein_coord(pdbfile)
    # print(distance_matrix[:10, :10])

    seq_mat = np.empty((len(seq), num_aa))
    for i, aa in enumerate(seq):
        aa_num = aa2num[aa]
        temp = torch.tensor(aa_num, dtype=torch.int64)
        seq_mat[i] = F.one_hot(temp, num_aa)
    seq_mat = torch.tensor(seq_mat, dtype=torch.double)
    # print(seq_mat)

    node_pos = np.column_stack((CA_coords_x, CA_coords_y, CA_coords_z))
    node_pos = torch.from_numpy(node_pos).to(dtype=torch.double)
    # print(node_pos)
    node_feature = torch.cat((seq_mat, node_pos), dim=1)
    
    # edge_index
    edge_index = np.nonzero(adjacency)
    # print(edge_index)
    edge_index = torch.from_numpy(np.asarray(edge_index))
    # print(edge_index[0][:10], edge_index[1][:10])

    # edge features - times of threshold (25)
    edge_attr = np.empty((len(edge_index[0]), 1))
    for x in range(len(edge_index[0])):
        # print(edge_index[0][x], edge_index[1][x])
        # print(distance_matrix[edge_index[0][x], edge_index[1][x]])
        times = distance_matrix[edge_index[0][x], edge_index[1][x]] / 25
        # print(times)
        edge_attr[x, 0] = times    
    edge_attr = torch.tensor(edge_attr, dtype=torch.double)
    # print(adjacency)
    # print(edge_index)
    # print(edge_attr)

    # label
    label = labels[path[:-5]]
    label = torch.tensor(label)

    data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, num_nodes = len(adjacency), y=label)
    dataset.append(data)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
loader_iter = iter(data_loader)
batch = next(loader_iter)

# shuffle
random.shuffle(dataset)
# split
split_idx = int(len(dataset) * 0.8)
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]

# print("len train:", len(train_dataset))
# print("len test:", len(test_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

from torch_geometric.nn import global_mean_pool
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        
        # your code here
        
        #    1. Define the first convolution layer using `GCNConv()`. Set `out_channels` to 64;
        self.layer1 = GCNConv(23, 64)    # only consider one graph but not all the graphs in one batch? - yes, and the first number is the number of node features
        #    2. Define the first activation layer using `nn.ReLU()`;
        self.layer2 = nn.ReLU(inplace=True)
        #    3. Define the second convolution layer using `GCNConv()`. Set `out_channels` to 64;
        self.layer3 = GCNConv(64, 64)
        #    4. Define the second activation layer using `nn.ReLU()`;
        self.layer4 = nn.ReLU(inplace=True)
        #    5. Define the third convolution layer using `GCNConv()`. Set `out_channels` to 64;
        self.layer5 = GCNConv(64, 64)
        #    6. Define the dropout layer using `nn.Dropout()`;
        self.layer6 = nn.Dropout(p=0.5)
        #    7. Define the linear layer using `nn.Linear()`. Set `output_size` to 2.
        self.layer7 = nn.Linear(64, 2)


    def forward(self, x, edge_index, edge_attr, batch):
        """
        TODO:
            1. Pass the data through the frst convolution layer;
            2. Pass the data through the activation layer;
            3. Pass the data through the second convolution layer;
            4. Obtain the graph embeddings using the readout layer with `global_mean_pool()`;
            5. Pass the graph embeddgins through the dropout layer;
            6. Pass the graph embeddings through the linear layer.
            
        Arguments:
            x: [num_nodes, 7], node features
            edge_index: [2, num_edges], edges
            batch: [num_nodes], batch assignment vector which maps each node to its 
                   respective graph in the batch

        Outputs:
            probs: probabilities of shape (batch_size, 2)
        """
        
        # your code here

        #     1. Pass the data through the frst convolution layer;
        x = self.layer1(x, edge_index, edge_attr)
        #     2. Pass the data through the activation layer;
        x = self.layer2(x)
        #     3. Pass the data through the second convolution layer;
        x = self.layer4(self.layer3(x, edge_index, edge_attr))
        #     4. Obtain the graph embeddings using the readout layer with `global_mean_pool()`;
        x = self.layer5(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        #     5. Pass the graph embeddgins through the dropout layer;
        x = self.layer6(x).to(torch.float64)
        #     6. Pass the graph embeddings through the linear layer.
        probs = self.layer7(x)
        
        return probs
        
gcn = GCN().double()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# optimizer
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.00001)
# loss
criterion = torch.nn.CrossEntropyLoss()

def train(train_loader):
    gcn.train()  # set the model to training mode
    for data in train_loader:  # Iterate in batches over the training dataset.
        """
        TODO: train the model for one epoch.
        
        Note that you can acess the batch data using `data.x`, `data.edge_index`, `data.batch`, `data.y`.
        """
        
        # your code here

        # load the input features, input graph and labels to device 
        input_features = data.x.to(device)
        input_edge_index = data.edge_index.to(device)
        input_edge_attr = data.edge_attr.to(device)
        input_batch = data.batch.to(device)
        input_labels = data.y.to(device)

        optimizer.zero_grad()
        output = gcn(input_features, input_edge_index, input_edge_attr, input_batch)
        loss = criterion(output, input_labels)
        # print('loss: ', loss)
        loss.backward()
        optimizer.step()


def test(loader):
    gcn.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = gcn(data.x, data.edge_index, data.edge_attr, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(20):
    train(train_loader)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch + 1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

test_acc = test(test_loader)
print(test_acc)