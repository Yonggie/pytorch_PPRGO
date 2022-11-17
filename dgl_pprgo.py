import dgl
from dgl import PPR
import torch.nn as nn
import torch.nn.functional as F
import torch


class PPRGO(nn.Module):
    
    def __init__(self,K,x_in,x_out,re_ppr=False):
        '''
            K: top k neighbour of PPR
            x_in: X feature input dimension
            x_out: output dimension
        '''
        super().__init__()
        self.K=K
        self.re_ppr=re_ppr
        self.ppr_info=None
        self.f_theta=nn.Linear(x_in,x_out)
        
    
    def compute(self,g):
        ppr=PPR(avg_degree=0)
        ppr_info=ppr(g).edata['w'].reshape(g.num_nodes(),g.num_nodes())
        self.ppr_info=ppr_info
        torch.save(self.ppr_info,'model_data/ppr_info.pt')

        return ppr_info


    def get_ppr_info(self,g):
        import os
        if self.re_ppr:
            print('processing PPR, it may take a while...')
            self.ppr_info=self.compute(g)
        else:
            if os.path.exists('model_data/ppr_info.pt'):
                print('load PPR information from the existing.')
                self.ppr_info=torch.load('model_data/ppr_info.pt')
            else:
                print('processing PPR, it may take a while...')
                self.ppr_info=self.compute(g)

        


    def forward(self,g):
        
        if self.ppr_info is None:
            self.get_ppr_info(g)
        
        H=self.f_theta(g.ndata['x'])
        topk_idxs=torch.argmax(self.ppr_info,dim=1,keepdim=True)[:,:self.K] # n*k
        topk_matrix=self.ppr_info.gather(1,topk_idxs) # n*k
        topk_matrix=topk_matrix.unsqueeze(-1) # n*k*1
        topk_h=H[topk_idxs] # n*k*d
        Z=(topk_matrix*topk_h).sum(dim=-2) # n*d

        return Z


# example
# num_node=100
# feature_dim=128
# out_dim=32
# edge_list=([0,0,0,1,2,4,5],[1,2,3,4,2,5,4]) # 7 edges
# edge_num=len(edge_list[0])
# # pprgo information
# K=3

# pprgo=PPRGO(K,feature_dim,out_dim)
# toy_g=dgl.graph(edge_list,num_nodes=num_node)
# toy_g.ndata['x'] = torch.randn(num_node, feature_dim)
# out=pprgo(toy_g)
# print(out)


from dgl.data import FraudDataset

dataset = FraudDataset('yelp')
# yelp: node 45,954(14.5%);
# amazon: node 11,944(9.5%);
hete_g=dataset[0]

num_classes = dataset.num_classes
label = hete_g.ndata['label']
node_labels = hete_g.ndata['label']
train_mask = hete_g.ndata['train_mask'].bool()
valid_mask = hete_g.ndata['val_mask'].bool()
test_mask = hete_g.ndata['test_mask'].bool()

x_dim=128
graph = dgl.to_homogeneous(hete_g)
graph.ndata['x'] = torch.randn(graph.num_nodes(), x_dim)
node_features=graph.ndata['x']


def evaluate(model, graph, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


K=3
out_dim=32
EPOCH=5000
iterval=100

model = PPRGO(K,x_dim,out_dim)
optimizer = torch.optim.Adam(model.parameters())

print('start training...')
for epoch in range(EPOCH):
    model.train()
    # forward propagation by using all nodes
    logits = model(graph)
    # compute loss
    loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
    
    # backward propagation
    
    if epoch%iterval==0:
        # compute validation accuracy
        acc = evaluate(model, graph, node_labels, valid_mask)  
        print(f'epoch {epoch}, loss {loss.item()}, acc {acc}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save model if necessary.  Omitted in this example.

print('testing...')
with torch.no_grad():
    accs=[]
    for _ in range(10):
        acc = evaluate(model, graph, node_labels, test_mask)  
        accs.append(acc)
import numpy as np
print(f'final test: {np.mean(accs)}, std {np.std(accs)}')
exit()
