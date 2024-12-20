import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import networkx as nx
from models import DGI, LogReg
import utils
from torch_geometric.data import Data
import statistics
import argparse
import matplotlib.pyplot as plt



def get_res(edges_pos, edges_neg, embeddings, adj_sparse, data_name):
    score_matrix = np.dot(embeddings, embeddings.T)    
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(utils.sigmoid(score_matrix[edge[0], edge[1]]))  
        pos.append(adj_sparse[edge[0], edge[1]])  

    
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(utils.sigmoid(score_matrix[edge[0], edge[1]]))  
        neg.append(adj_sparse[edge[0], edge[1]])  

    
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    auc, ap, f1 = utils.evaluate_auc(labels_all, preds_all, data_name)    
    
    return auc, ap, f1


    

def DGIM(data_name):
    np.random.seed(42)

    torch.manual_seed(42)
    
    parser = argparse.ArgumentParser(description='DGI')
    parser.add_argument('--b', dest='beta', type=int, default=100, help='')
    parser.add_argument('--c', dest='num_clusters', type=float, default=128, help='')
    parser.add_argument('--a', dest='alpha', type=float, default=0.5, help='')
    parser.add_argument('--test_rate', dest='test_rate', type=float, default=0.1, help='')
    
    
    args = parser.parse_args()
        
    cuda0 = torch.cuda.is_available()  
    
    beta = args.beta
    alpha = args.alpha
    num_clusters = int(args.num_clusters)
    
        
    # training params
    batch_size = 1
    nb_epochs = 100
    runs = 5
    patience = 50
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 16
    sparse = False
    nonlinearity = 'prelu'  
    embsize = 128
    
    torch.cuda.empty_cache()
    
        
    for m in range(1):
    
             
        dataset = torch.load(f"../../datasets/{data_name}.pt") 
        geneCoexpression = Data()
        geneCoexpression.num_nodes = len(dataset['gene']['node_id'])
        geneCoexpression.edge_index = dataset['interacts']['edge_index']
        geneCoexpression.x = torch.rand(geneCoexpression.num_nodes,embsize)-1.5*torch.ones(geneCoexpression.num_nodes,embsize)
        geneCoexpression.num_features = 128
        all_edges = geneCoexpression.edge_index
        
        graph = {}
        edge_index = geneCoexpression.edge_index
        
        for i in range(len(edge_index[0])):
            if int(edge_index[0][i]) in graph:
                graph[int(edge_index[0][i])].append(int(edge_index[1][i]))
            else:
                graph[int(edge_index[0][i])] = []
                graph[int(edge_index[0][i])].append(int(edge_index[1][i]))
            if int(edge_index[1][i]) in graph:
                graph[int(edge_index[1][i])].append(int(edge_index[0][i]))
            else:
                graph[int(edge_index[1][i])] = []
                graph[int(edge_index[1][i])].append(int(edge_index[0][i]))
            
        
        myKeys = list(graph.keys())
        myKeys.sort()
        
        sorted_graph = {i: graph[i] for i in myKeys}
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(sorted_graph))
        
        
        adj_sparse = adj
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
        test_edges, test_edges_false = utils.mask_test_edges(adj, test_frac=args.test_rate, val_frac=0.05)#val_frac=0.05
        
        adj = adj_train
            
        adj = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
    
        if sparse:
            sp_adj = sparse_mx_to_torch_sparse_tensor(adj)
        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()
    
        
        emb = geneCoexpression.x
        features = torch.FloatTensor(torch.rand(1, emb.shape[0], 128))
        features[0] = emb        
        nb_nodes = features.shape[1]
        ft_size = features.shape[2]
        
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
        
    
        if cuda0:
            
            features = features.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
            else:
                adj = adj.cuda()
            
    
        b_xent = nn.BCEWithLogitsLoss()
        b_bce = nn.BCELoss()
        
    
        all_accs = []
        
        res = {}
                
        for beta in [args.beta]:          
            print()
            for K in [int(args.num_clusters)]:
                
                for alpha in [args.alpha]:
                    
                    model = DGI(ft_size, hid_units, nonlinearity)
                    for run in range(runs):
                        
                        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
                        cnt_wait = 0
                        best = 1e9
                        best_t = 0
                        val_best = 0
        
                        if cuda0:
                            model.cuda()
                        for epoch in range(nb_epochs):
                            model.train()
                            optimiser.zero_grad()
        
                            idx = np.random.permutation(nb_nodes)
                            shuf_fts = features[:, idx, :]
        
                            lbl_1 = torch.ones(batch_size, nb_nodes)
                            lbl_2 = torch.zeros(batch_size, nb_nodes)
                            lbl = torch.cat((lbl_1, lbl_2), 1)
        
                            if cuda0:
                                shuf_fts = shuf_fts.cuda()
                                lbl = lbl.cuda()
        
                            logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
        
                            loss = b_xent(logits, lbl)
        
                            if loss < best:
                                best = loss
                                best_t = epoch
                                cnt_wait = 0
                                torch.save(model.state_dict(), data_name + '-link.pkl')
        
                            else:
                                cnt_wait += 1
                                   
                            loss.backward()
                            optimiser.step()
    
                        model.load_state_dict(torch.load(data_name + '-link.pkl'))
        
                        embeds, S = model.embed(features, sp_adj if sparse else adj, sparse, None)
                        embs = embeds[0, :]
                        embs = embs / embs.norm(dim=1)[:, None]    
                        
                        auc, ap, f1 = get_res(test_edges, test_edges_false, embs.cpu().detach().numpy(), adj_sparse, data_name)
                        if run == 0:
                            res['AUC'] = [auc]
                            res['AP'] = [ap]
                            res['f1'] = [f1]
                            
                        else: 
                            res['AUC'].append(auc)
                            res['AP'].append(ap)
                            res['f1'].append(f1)
                        
                    
                    print(f" Final Testing result- {data_name}  ")
                    utils.print_res(res)            
                    
                    
                    
                    
                    
    
                                

if __name__ == '__main__':
    DGIM('sch') # t2d, pd, hd, sch


