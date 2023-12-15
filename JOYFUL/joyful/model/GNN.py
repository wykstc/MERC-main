import torch.nn as nn
import torch
from torch_geometric.nn import RGCNConv, TransformerConv
import copy
import numpy as np
from GCL.models import DualBranchContrast
import GCL.losses as L

torch.cuda.manual_seed(24)

def sim(h1, h2):
    z1 = nn.functional.normalize(h1, dim=-1, p=2)
    z2 = nn.functional.normalize(h2, dim=-1, p=2)

    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to('cuda:0')
    loss = contrast_model(z1,z2)
    return loss

def contrastive_loss_wo_cross_network(h1, h2, ho):
    intra1 = sim(ho, h1)
    intra2 = sim(ho, h2)
    return intra1+intra2

def random_feature_mask(input_feature, drop_percent, device=torch.device('cuda:0')):
    p = torch.ones(input_feature.shape,dtype=torch.float).bernoulli_(1-drop_percent).to(device)
    aug_feature = input_feature * p
    return aug_feature

def random_edge_pert(edge_index, num_nodes, pert_percent, device=torch.device('cuda:0')):
    num_edges = edge_index.shape[1]
    pert_num_edges = int(num_edges*pert_percent)
    pert_idxs = np.random.choice(num_edges, pert_num_edges, replace=False)
    edge_index[1, pert_idxs] = torch.LongTensor(np.random.randint(0, num_nodes, pert_num_edges)).to(device)
    return edge_index

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, args):
        super(GNN, self).__init__()
        self.num_relations = 2 * args.n_speakers ** 2

        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations)

        self.transform_conv = TransformerConv(h1_dim, h2_dim, heads=args.gnn_nheads, concat=True)

        self.bn = nn.BatchNorm1d(h2_dim * args.gnn_nheads)

    def forward(self, node_features, edge_index, edge_type, trainW):

        if trainW:
            drop_percent1 = 0.25
            drop_percent2 = 0.25
            pert_percent1 = 0.1
            pert_percent2 = 0.1

            num_nodes = node_features.shape[0]
            aug1_embedding = random_feature_mask(node_features, drop_percent1, device=torch.device('cuda:0'))
            aug1_edge_index = random_edge_pert(edge_index, num_nodes, pert_percent1, device=torch.device('cuda:0'))

            aug2_embedding = random_feature_mask(node_features, drop_percent2, device=torch.device('cuda:0'))
            aug2_edge_index = random_edge_pert(edge_index, num_nodes, pert_percent2, device=torch.device('cuda:0'))

            h1 = self.conv1(aug1_embedding, aug1_edge_index, edge_type)
            h2 = self.conv1(aug2_embedding, aug2_edge_index, edge_type)

            ho = self.conv1(node_features, edge_index, edge_type)

            loss = contrastive_loss_wo_cross_network(h1, h2, ho)

            x = nn.functional.leaky_relu(self.bn(self.transform_conv(ho, edge_index)))
            return x, loss
        else:
            ho = self.conv1(node_features, edge_index, edge_type)
            x = nn.functional.leaky_relu(self.bn(self.transform_conv(ho, edge_index)))
            return x, 0


