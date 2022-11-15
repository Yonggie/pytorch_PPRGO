
# 做掉安装anaconda pytorch dgl
# 做掉page rank “算法”
# 初步梳理pprgo模型
# 熟悉dgl框架

# page rank



# after pagerank
import torch
pagerank_vecs=torch.rand(10,128)
reps=torch.rand(10,128)

def find_top_k_idx(index):
    '''
        return the neighbour indices of target index
    '''
    return top_k_rep[index],top_k_pr_vec[index]

def f_theta(x):
    return mlp(x)

zs=[]
for i in enumerate(reps.shape[0]):
    top_k_idxs=find_top_k_idx(i)
    
    top_k_pr_vecs=pagerank_vecs[top_k_idxs]
    
    top_k_reps=reps[top_k_idxs]
    
    top_k_hs=f_theta(top_k_reps)

    z_i=torch.sum(top_k_pr_vecs@top_k_hs,dim=0)

    zs.append(z_i)

Z=torch.cat([zs],dim=0)







