import dgl
from dgl import PPR
import torch.nn as nn
import numpy as np
import torch

# Assign a 4-dimensional edge feature vector for each edge.
# g.edata['a'] = torch.randn(5, 4)

# common information
num_node=100
feature_dim=128
edge_list=([0,0,0,1,2,4,5],[1,2,3,4,2,5,4]) # 7 edges
edge_num=len(edge_list[0])
# pprgo information
K=3


# initialize random graph
input_g=dgl.graph(edge_list,num_nodes=num_node)
input_g.ndata['x'] = torch.randn(num_node, feature_dim)
# for directed graph
# input_g.edata['w'] = torch.rand(edge_num)


# personalized page rank
ppr=PPR()
ppr_g=ppr(input_g)

# sort top k information
ppr_info=ppr_g.edata['w'].reshape(num_node,num_node)

    



def find_top_k_idx(index):
    '''
        return the neighbour indices of target index
    '''
    return torch.argmax(ppr_info[index],keepdim=True)[:K]


def f_theta(x):
    return x


# node level forward
H=f_theta(input_g.ndata['x'])

zs=[]
for i in range(input_g.ndata['x'].shape[0]):
    top_k_idxs=find_top_k_idx(i)

    
    topk_h=H[top_k_idxs] # k*d
    # topk_diag=ppr_info[i][top_k_idxs].diag()
    # z_i=topk_diag@topk_h

    topk_vs=ppr_info[i][top_k_idxs] # k*1
    z_i=(topk_vs*topk_h).sum(dim=-2)
    zs.append(z_i)

Z=torch.stack(zs)


# vectorization =======================================================
H=f_theta(input_g.ndata['x']) # n*d
topk_idxs=torch.argmax(ppr_info,dim=1,keepdim=True)[:,:K] # n*k
topk_matrix=ppr_info.gather(1,topk_idxs) # n*k
topk_matrix=topk_matrix.unsqueeze(-1) # n*k*1
topk_h=H[topk_idxs] # n*k*d
Z2=(topk_matrix*topk_h).sum(dim=-2) # n*d


# check done, 结果完全一样
exit()