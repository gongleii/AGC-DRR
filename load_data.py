import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset

import networkx as nx
import gae_dblp.opt

def wrong_edge(num):
    v1 = np.random.randint(100,size = num)
    v2 = np.random.randint(100,size = num)
    random_edge = np.zeros((2*num,2),dtype=np.int32)
    for i in range(num):
       e1 = v1[i]
       e2 = v2[i]
       random_edge[2 * i][0] = e1
       random_edge[2 * i][1] = e2
       random_edge[2 * i + 1][0] = e2
       random_edge[2 * i + 1][1] = e1

    #print(random_edge)
    return random_edge


def new_graph(edge_index,weight,n,device):
    edge_index = edge_index.cpu().numpy()
    indices = torch.from_numpy(
        np.vstack((edge_index[0], edge_index[1])).astype(np.int64)).to(device)
    values = weight
    shape = torch.Size((n,n))
    return torch.sparse.FloatTensor(indices, values, shape)

def load_graph(k, graph_k_save_path, graph_save_path, data_path,walk_length,num_walk):
    if k:
        path = graph_k_save_path
    else:
        path = graph_save_path

    print("Loading path:", path)

    data = np.loadtxt(data_path, dtype=float)

    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

  #  random_edge = wrong_edge(100)
  #  edges = np.vstack((edges,random_edge))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    G = nx.DiGraph()

    # add edges
    for i in range(len(edges)):
        src = str(edges[i][0])
        dst = str(edges[i][1])
        G.add_edge(src, dst)
        G[src][dst]['weight'] = 1.0
        #  print("88888888888888",G.edges)






    # g = Graph(G)

    model = Node2vec_onlywalk(num = n,graph=G, path_length=walk_length, num_paths=num_walk, dim=4, workers=8,
                              window=5, p=2, q=0.5, dw=False)

    return adj,model.walker#,random_edge


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normal(x):
    rowmu = (np.mean(x,axis=1)).reshape((x.shape[0],1)).repeat(x.shape[1],1)
    rowstd = (np.std(x,axis=1)).reshape((x.shape[0],1)).repeat(x.shape[1],1)

    return (x-rowmu)/rowstd



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))