import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected


def mask_edge(edge_index, p):
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    #print("edge_index1: ",edge_index)
    #print("e_ids: ",e_ids)
    #print("e_ids.shape: ",e_ids.shape)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    #print("mask: ",mask)
    #print("mask.shape: ",mask.shape)
    mask = torch.bernoulli(mask).to(torch.bool)
    #print("mask: ",mask)
    #print("mask.shape: ",mask.shape)
    """
    i =0
    for j in range(len(mask)):
      if mask[j]==True:
        i=i+1
    print("i: ",i)
    """
    #print("edge_index2: ", edge_index[:, ~mask])
    return edge_index[:, ~mask], edge_index[:, mask]


class Mask(nn.Module):
    def __init__(self, p):
        super(Mask, self).__init__()
        self.p = p

    def forward(self, edge_index):
        #print("edge_index1: ",edge_index)
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        #print(len(remaining_edges[0])+len(masked_edges[0]))
        #print(len(remaining_edges[0]))
        remaining_edges = to_undirected(remaining_edges)
        #print(len(remaining_edges[0]))
        return remaining_edges, masked_edges
