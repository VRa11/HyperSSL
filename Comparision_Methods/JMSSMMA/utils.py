import torch
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
import json as js


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred):
    #print("y_pred: ",y_pred)
    #print("y_true: ",y_true)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP*TN-FP*FN)/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    F1_score = 2*(precision*sensitivity)/(precision+sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data(data_name, output_dim):
    location_feature = f'data/{data_name}_feature.txt'
    miRNA = np.loadtxt(location_feature)  # (541, 541)
    
    location_adj = f'data/{data_name}_adj.txt' 
    association = np.loadtxt(location_adj, delimiter=',')  # (541, 831)
  
    m_emb = []
    for m in range(len(miRNA)):
        m_emb.append(miRNA[m].tolist())
    m_emb = [lst + [0] * (output_dim - len(m_emb[0])) for lst in m_emb]
    m_emb = torch.Tensor(m_emb)    

    feature = m_emb #new line

    adj = []
    
    for m in range(len(miRNA)):
        for s in range(len(miRNA)):
            if association[m][s] == 1:
                adj.append([m, s])  
    
    adj = torch.LongTensor(adj).T
    data = Data(x=feature, edge_index=adj).cuda()

    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                                 is_undirected=True, split_labels=True,
                                                 add_negative_train_samples=True)(data)

    splits = dict(train=train_data, test=test_data)
    return splits
def print_res(res):
    r_auc = res['AUC']
    r_auc_mean = np.mean(r_auc)                   
    r_auc_std = np.std(r_auc)                   
    r_ap = res['AP']
    r_ap_mean = np.mean(r_ap)    
    r_ap_std = np.std(r_ap)                       
    r_f1 = res['f1']
    r_f1_mean = np.mean(r_f1)                    
    r_f1_std = np.std(r_f1)               
                    
    print(f'   AUC: {r_auc_mean} +- {r_auc_std}')
    print(f'   AP: {r_ap_mean} +- {r_ap_std}')
    print(f'   F1: {r_f1_mean} +- {r_f1_std}')


if __name__ == '__main__':
    data = get_data(2, 512)
