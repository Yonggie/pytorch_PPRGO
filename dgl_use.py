import dgl
from dgl import PPR
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
input_g.ndata['pr'] = torch.randn(num_node, feature_dim)
# required by personalized page rank (PPR)
# input_g.edata['w'] = torch.rand(edge_num)


# personalized page rank
ppr=PPR()
ppr_g=ppr(input_g)

# sort top k information
ppr_info=ppr_g.edata['w'].reshape(num_node,num_node)
    
# from collections import defaultdict

# adj_list=defaultdict(list) # source index:[dist_idx0,dist_idx1,...]
# srcs,dists=input_g.edges()
# weights=input_g.edata['w']
# for src,dist,w in  zip(srcs,dists,weights):
#     adj_list[src.item()].append((dist.item(),w.item()))

# sorted_adj_list={k:sorted(v,key=lambda x:x[1],reverse=True) for k,v in adj_list.items()}
# top_k={k:[item[0] for item in v[:K]] for k,v in sorted_adj_list.items()}

def find_top_k_idx(index):
    '''
        return the neighbour indices of target index
    '''
    return torch.argmax(ppr_info[index],dim=0)[:K]


def f_theta(x):
    return mlp(x)

H=f_theta(X)
zs=[]
for i in enumerate(reps.shape[0]):
    top_k_idxs=find_top_k_idx(i)

    topk_vec=ppr_info[i][top_k_idxs]
    topk_h=H[top_k_idxs]
    
    z_i=torch.sum(topk_vec@topk_h,dim=1)
    
    zs.append(z_i)

Z=torch.cat([zs],dim=0)

