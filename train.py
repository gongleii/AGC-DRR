import torch
from opt import args
from utils import eva,target_distribution
from torch.optim import Adam
import torch.nn.functional as F
from load_data import *

import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity
acc_reuslt = []
acc_reuslt.append(0)
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Train_gae(model,view_learner, data, adj, label,edge_index):
    acc_reuslt = []
    acc_reuslt.append(0)
    nmi_result = []
    ari_result = []
    f1_result = []

    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):

        view_learner.train()
        view_learner.zero_grad()
        model.eval()
        z_igae, c= model(data, adj)

        n = z_igae.shape[0]
        edge_logits = view_learner(data,adj,edge_index)


        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

        aug_adj= new_graph(torch.tensor(edge_index).to('cuda'),batch_aug_edge_weight,n,'cuda')
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * adj
        aug_adj = aug_adj.cpu().detach().numpy()+np.eye(n)
        aug_adj = torch.from_numpy(normalize(aug_adj)).to(torch.float32).to('cuda')


        aug_z_igae,aug_c= model(data, aug_adj)



        edge_drop_out_prob = 1 - batch_aug_edge_weight
        reg = edge_drop_out_prob.mean()

        view_loss = (args.reg_lambda * reg)+model.calc_loss(c.T,aug_c.T)+model.calc_loss(c, aug_c)

        (-view_loss).backward()
        view_optimizer.step()

        view_learner.eval()


        model.train()
        model.zero_grad()
        z_igae, c = model(data, adj)

        n = z_igae.shape[0]
        #with torch.no_grad():
        edge_logits = view_learner(data, adj, edge_index)


        batch_aug_edge_weight = torch.sigmoid(edge_logits).squeeze()  # p

        aug_adj = new_graph(torch.tensor(edge_index).to('cuda'), batch_aug_edge_weight, n,'cuda')
        aug_adj = aug_adj.to_dense()
        aug_adj = aug_adj * adj
        aug_adj = aug_adj.cpu().detach().numpy() + np.eye(n)
        aug_adj = torch.from_numpy(normalize(aug_adj)).to(torch.float32).to('cuda')

        aug_z_igae, aug_c = model(data, aug_adj)

        z_mat =torch.matmul(z_igae, aug_z_igae.T)


        model_loss = model.calc_loss(c.T, aug_c.T) + F.mse_loss(z_mat, torch.eye(n).to('cuda'))+ model.calc_loss(c, aug_c)
        model_loss.backward()
        optimizer.step()
        model.eval()

        print('{} loss: {}'.format(epoch, model_loss))
        z = (c + aug_c)/2
     #   kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
        i = z.argmax(dim=-1)
        acc, nmi, ari, f1 = eva(label, i.data.cpu().numpy(), epoch)
    #    acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

