import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch.nn import Linear
import numpy as np
from torch.nn import Sequential

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    
    
    """
    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx
    """

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgae(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', num_nodes=1000):
        super(GCN_mgae, self).__init__()
        self.decoder_mask = decoder_mask

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

        if decoder_mask == 'mask':
            self.n_emb = torch.nn.Embedding(num_nodes, out_channels)
            self.mask_lins = torch.nn.ModuleList()
            self.mask_lins.append(torch.nn.Linear(out_channels * 2, out_channels * 2))
            self.mask_lins.append(torch.nn.Linear(out_channels * 2, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # if self.decoder_mask == 'mask':
        #     for conv in self.mask_lins:
        #         conv.reset_parameters()
            # torch.nn.init.xavier_uniform_(self.n_emb.weight.data)
            # torch.nn.init.normal_(self.n_emb.weight, std=0.1)

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgae_ablation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_input='last', num_nodes=1000):
        super(GCN_mgae_ablation, self).__init__()
        self.decoder_input = decoder_input

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_input == 'last':
            return x
        else:
            x = torch.cat(xx, dim=-1)
            return x

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', num_nodes=1000):
        super(GCN_mgaev2, self).__init__()
        self.decoder_mask = decoder_mask

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x


class GIN_mgaev2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', eps: float = 0.,  bias=True, xavier=True):
        super(GIN_mgaev2, self).__init__()
        self.decoder_mask = decoder_mask
        self.initial_eps = eps
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.ReLU()
        for i in range(num_layers - 1):
            start_dim = hidden_channels if i else in_channels
            nn = Sequential(Linear(start_dim, hidden_channels, bias=bias),
                            self.act,
                            Linear(hidden_channels, hidden_channels, bias=bias))
            # if xavier:
            #     self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
        nn = Sequential(Linear(hidden_channels, hidden_channels, bias=bias),
                        self.act,
                        Linear(hidden_channels, out_channels, bias=bias))
        # if xavier:
        #     self.weights_init(nn)
        conv = GINConv(nn)
        self.convs.append(conv)

        self.dropout = dropout

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for conv in self.convs:
            # self.weights_init(conv.nn)
            # conv.eps.data.fill_(self.initial_eps)
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GIN_mgaev33(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', eps: float = 0.,  bias=True, xavier=True):
        super(GIN_mgaev33, self).__init__()
        self.decoder_mask = decoder_mask
        self.initial_eps = eps
        self.convs = torch.nn.ModuleList()
        self.act = torch.nn.ReLU()
        for i in range(num_layers - 1):
            start_dim = hidden_channels if i else in_channels
            nn = Sequential(Linear(start_dim, hidden_channels, bias=bias),
                            self.act,
                            Linear(hidden_channels, hidden_channels, bias=bias))
            # if xavier:
            #     self.weights_init(nn)
            conv = GINConv(nn)
            self.convs.append(conv)
        nn = Sequential(Linear(hidden_channels, hidden_channels, bias=bias),
                        self.act,
                        Linear(hidden_channels, out_channels, bias=bias))
        # if xavier:
        #     self.weights_init(nn)
        conv = GINConv(nn)
        self.convs.append(conv)

        self.dropout = dropout

    def weights_init(self, module):
        for m in module.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def reset_parameters(self):
        for conv in self.convs:
            # self.weights_init(conv.nn)
            # conv.eps.data.fill_(self.initial_eps)
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            xx.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgaev3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', num_nodes=1000):
        super(GCN_mgaev3, self).__init__()
        self.decoder_mask = decoder_mask

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        #self.convs.append(GraphConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
                #GraphConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))
        #self.convs.append(GraphConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(F.relu(x))
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x


class GCN_mgaev33(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, decoder_mask='nmask', num_nodes=1000):
        super(GCN_mgaev33, self).__init__()
        self.decoder_mask = decoder_mask

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, add_self_loops=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, add_self_loops=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def mask_decode(self, x):
        x = torch.cat([self.n_emb.weight, x], dim=-1)
        for lin in self.mask_lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mask_lins[-1](x)
        return x

    def forward(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            xx.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        return xx

    def outEmb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        x = torch.cat(xx, dim=1)
        return x

    def generate_emb(self, x, adj_t):
        xx = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            xx.append(x)
        x = self.convs[-1](x, adj_t)
        xx.append(x)
        if self.decoder_mask == 'mask':
           x = self.mask_decode(x)
        x = torch.cat(xx, dim=1)
        return x




class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, de_v='v2'):
        super(LPDecoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        if de_v == 'v1':
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        #print("h: ",h)
        #print("h.shzpe: ",h.shape)
        #print("edge: ",edge)
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
        
class LPDecoder1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout, de_v='v2'):
        super(LPDecoder1, self).__init__()
        #n_layer = encoder_layer * encoder_layer
        n_layer = 1
        self.lins = torch.nn.ModuleList()
        if de_v == 'v1':
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
            self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()


    def forward(self, h, edge):
        #print("h: ",h)
        #print("h.shzpe: ",h.shape)
        #print("edgeLP1: ",edge)
        #print("h: ",h)
        #print("edge: ",edge)
        src_x = h[edge[0]] 
        dst_x = h[edge[1]]
        #print("src_x.shape: ",src_x.shape)
        #print("src_x: ",src_x)
        x = src_x * dst_x
        #print("x: ",x)
        #print("x.shape: ",x.shape)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LPDecoder_ogb(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout):
        super(LPDecoder_ogb, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        # self.lins.append(torch.nn.Linear(in_channels * n_layer, in_channels * n_layer))
        # self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        # self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LPDecoder_ogb_layer3(torch.nn.Module):
    def __init__(self, in_channels, hid1, hid2, out_channels, encoder_layer, dropout):
        super(LPDecoder_ogb_layer3, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()

        self.lins.append(torch.nn.Linear(in_channels * n_layer, hid1))
        self.lins.append(torch.nn.Linear(hid1, hid2))
        self.lins.append(torch.nn.Linear(hid2, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def forward(self, h, edge):
        src_x = [h[i][edge[0]] for i in range(len(h))]
        dst_x = [h[i][edge[1]] for i in range(len(h))]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


class LPDecoder_ablation(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout, num_layers, decoder_input='last', decoder_type='inner'):
        super(LPDecoder_ablation, self).__init__()
        self.decoder_type = decoder_type
        if decoder_type == 'inner':
            pass
        else:
            if decoder_input == 'last':
                in_channels = in_channels
            else:
                in_channels = in_channels * num_layers
            self.lins = torch.nn.ModuleList()
            self.lins.append(torch.nn.Linear(in_channels * 2, in_channels * 2))
            self.lins.append(torch.nn.Linear(in_channels * 2, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

            self.dropout = dropout

    def reset_parameters(self):
        if self.decoder_type == 'mlp':
            for lin in self.lins:
                lin.reset_parameters()

    def decode_inner(self, x_i, x_j):
        return (x_i * x_j).sum(dim=-1)

    def decode_mlp(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

    def forward(self, h, edge):
        src_x = h[edge[0]]
        dst_x = h[edge[1]]
        if self.decoder_type == 'inner':
            x = self.decode_inner(src_x, dst_x)
        else:
            x = self.decode_mlp(src_x, dst_x)
        return torch.sigmoid(x)


class FeatPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(FeatPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x



class GCN_tune(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 dropout, src_layer, dst_layer):
        super(GCN_tune, self).__init__()
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        num_layers = max(self.src_layer, self.dst_layer)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x_hidden = []
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x_hidden.append(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x_hidden.append(x)
        return x_hidden[self.src_layer - 1], x_hidden[self.dst_layer - 1]



class SearchGraph_l2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, num_nodes,
                 temperature=0.07):
        super(SearchGraph_l2, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.num_nodes = num_nodes

        self.arc = torch.nn.Parameter(torch.ones(size=[num_nodes, hidden_channels], dtype=torch.float) / self.num_nodes)
        # self.trans = torch.nn.ModuleList()
        # for i in range(num_layers - 1):
        #     if i == 0:
        #         self.trans.append(Linear(in_channels, hidden_channels, bias=False))
        #     else:
        #         self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        # self.trans.append(Linear(hidden_channels, 1, bias=False))

    # def reset_parameters(self):
    #     for conv in self.trans:
    #         conv.reset_parameters()

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        # for conv in self.trans[:-1]:
        #     x = conv(x)
        #     x = F.relu(x)
        # x = self.trans[-1](x)
        # x = torch.squeeze(self.arc, dim=2)
        x = self.arc
        arch_set = torch.softmax(x / self.temperature, dim=1)
        device = arch_set.device
        if not self.training:
            n, c = arch_set.shape
            eyes_atten = torch.eye(c)
            atten_, atten_indice = torch.max(arch_set, dim=1)
            arch_set = eyes_atten[atten_indice]
            arch_set = arch_set.to(device)

        return arch_set


class SearchGraph_rs(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, arch_layers,
                 temperature=0.07):
        super(SearchGraph_rs, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.arch_layers = arch_layers

        self.search_len = num_layers**2
        self.search_space = torch.eye(self.search_len)
        self.choice_list = torch.arange(self.search_len, dtype=torch.float)

    def forward(self, x, grad=False):
        # x with shape [batch, num_layer, dim]
        n, c, d = x.shape
        device = x.device
        # rs_indice = torch.multinomial(self.choice_list, n, replacement=True)
        rs_indice = torch.randint(0, self.search_len, (n,), dtype=torch.long)
        arch_set = self.search_space[rs_indice]
        arch_set = arch_set.to(device)
        return arch_set


class SearchGraph_qa(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, arch_layers,
                 temperature=0.07):
        super(SearchGraph_qa, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers
        self.arch_layers = arch_layers

        self.search_len = sum([num_layers - i for i in range(num_layers)])
        self.search_space = torch.eye(self.search_len)
        self.choice_list = torch.arange(self.search_len, dtype=torch.float)

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        n, c, d = x.shape
        device = x.device
        # rs_indice = torch.multinomial(self.choice_list, n, replacement=True)
        rs_indice = torch.randint(0, self.search_len, (n,), dtype=torch.long)
        arch_set = self.search_space[rs_indice]
        arch_set = arch_set.to(device)
        return arch_set


class SearchGraph_l22(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 temperature=0.07):
        super(SearchGraph_l22, self).__init__()

        self.temperature = temperature
        self.num_layers = num_layers

        self.trans = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.trans.append(Linear(in_channels, hidden_channels, bias=False))
            else:
                self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.trans.append(Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        for conv in self.trans:
            conv.reset_parameters()

    def forward(self, x):
        # x with shape [batch, num_layer, dim]
        for conv in self.trans[:-1]:
            x = conv(x)
            x = F.relu(x)
        x = self.trans[-1](x)
        x = torch.squeeze(x, dim=2)
        arch_set = torch.softmax(x / self.temperature, dim=1)
        device = arch_set.device
        if not self.training:
            n, c = arch_set.shape
            eyes_atten = torch.eye(c)
            atten_, atten_indice = torch.max(arch_set, dim=1)
            arch_set = eyes_atten[atten_indice]
            arch_set = arch_set.to(device)

        return arch_set


class SearchGraph_l31(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, cat_type='multi',
                 temperature=0.07):
        super(SearchGraph_l31, self).__init__()
        self.temperature = temperature
        self.num_layers = num_layers
        self.cat_type = cat_type
        if self.cat_type == 'multi':
            in_channels = in_channels
        else:
            in_channels = in_channels * 2
        self.trans = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i == 0:
                self.trans.append(Linear(in_channels, hidden_channels, bias=False))
            else:
                self.trans.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.trans.append(Linear(hidden_channels, 1, bias=False))

    def reset_parameters(self):
        for conv in self.trans:
            conv.reset_parameters()

    def forward(self, x, grad=False):
        # x with shape [batch, num_layer, dim]
        for conv in self.trans[:-1]:
            x = conv(x)
            x = F.relu(x)
        x = self.trans[-1](x)
        x = torch.squeeze(x, dim=2)
        arch_set = torch.softmax(x / self.temperature, dim=1)
        if not self.training:
            if grad:
                return arch_set.detach()
            else:
                device = arch_set.device
                n, c = arch_set.shape
                eyes_atten = torch.eye(c)
                atten_, atten_indice = torch.max(arch_set, dim=1)
                arch_set = eyes_atten[atten_indice]
                arch_set = arch_set.to(device)
                return arch_set
        else:
            return arch_set

