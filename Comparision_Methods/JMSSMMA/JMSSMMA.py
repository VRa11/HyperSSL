import torch
import argparse
from mask import Mask
from utils import get_data, set_seed, print_res
from model import GNNEncoder, EdgeDecoder, DegreeDecoder, GMAE
import numpy as np
# main parameter

def JMSSMMA(data_name):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1, help="Choose Datasets (1 or 2)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for model and dataset.")
    parser.add_argument('--dim', type=int, default=128, help='Feature Dimension of Similarity Matrix')# earlier 1024
    parser.add_argument('--alpha', type=float, default=0.007, help='loss weight for degree prediction.')
    parser.add_argument('--p', type=float, default=0.7, help='Mask ratio')
    args = parser.parse_args()
    set_seed(args.seed)
    runs = 5
    
    splits = get_data(data_name, args.dim)
    #print("splits['train']['edge_index']: ", splits['train']['edge_index'])
    encoder = GNNEncoder(in_channels=args.dim, hidden_channels=64, out_channels=128)
    edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
    mask = Mask(p=args.p)
    
    model = GMAE(encoder, edge_decoder, degree_decoder, mask).cuda()
    res = {}
    
    auc_t = 0
    ap_t = 0
    f1_t = 0
    pred_t = []
    labels_t = []
    for run in range(runs):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
        for epoch in range(100):
            model.train()
            loss = model.train_epoch(splits['train'], optimizer, alpha=args.alpha)
        model.eval()
        test_data = splits['test']
        z = model.encoder(test_data.x, test_data.edge_index)
        
        #test_auc, test_ap, acc, sen, pre, spe, F1, mcc = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
        auc, ap, f1 = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
        
        if run == 0:
            res['AUC'] = [auc]
            res['AP'] = [ap]
            res['f1'] = [f1]
            #res['prec'] = [precision]
            #res['rec'] = [recall]
        else: 
            res['AUC'].append(auc)
            res['AP'].append(ap)
            res['f1'].append(f1)
       
       
    print(f" Final Testing result- {data_name}  ")
    print_res(res)

if __name__ == '__main__':
    JMSSMMA('hd') # pd, t2d, hd, sch