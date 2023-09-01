from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from my_knn import get_hypergraph, get_initial_value, get_neg_hypergraph
from CNN_auto_encoder import Ae
from getdata import Load_my_Dataset
from utils import cluster_accuracy
import warnings
from InitializeD import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)
warnings.filterwarnings("ignore")
epochs = 600

class C_EDESC(nn.Module):
    def __init__(self,
                 n_input,
                 n_z,
                 n_clusters,
                 pretrain_path='data/pre.pkl'):
        super(C_EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.ae = Ae(
            n_input=n_input,
            n_z=n_z)
        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z * 7 * 7, n_clusters))
        print(self.D.shape)
    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # Load pre-trained weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('Load pre-trained model from', path)

    def forward(self, x):
        x_bar, z = self.ae(x)
        z_shape = z.shape
        num_ = z_shape[0]
        z = z.reshape((num_, -1))
        d = args.d
        s = None
        eta = args.eta

        # Calculate subspace affinity
        for i in range(self.n_clusters):
            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * d:(i + 1) * d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + eta * d) / ((eta + 1) * d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, index, epoch, G,
                   NG,y):
        # Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)
        mask = torch.zeros([610, 340]).cuda()
        num_classes = 9
        s_matrix = torch.zeros([610, 340, target.shape[1]]).cuda()
        mask[index] = 1
        s_matrix[index[0], index[1], :] = target
        import kornia as k
        s_matrix = s_matrix.permute(2, 0, 1)
        s_matrix = torch.unsqueeze(s_matrix, dim=0)
        matrix_blur = k.filters.box_blur(s_matrix, kernel_size=(7, 7))
        matrix_blur = torch.squeeze(matrix_blur)
        matrix_blur = matrix_blur.permute(1, 2, 0)
        mask = torch.unsqueeze(mask, dim=0)
        mask = torch.unsqueeze(mask, dim=0)
        mask_blur = k.filters.box_blur(mask, kernel_size=(7, 7))
        mask_blur = torch.squeeze(mask_blur)
        mask_blur = torch.broadcast_to(mask_blur, [num_classes, mask_blur.shape[0], mask_blur.shape[1]])
        mask_blur = mask_blur.permute(1, 2, 0)
        matrix_s = matrix_blur / mask_blur
        new_target = matrix_s[index[0], index[1], :]
        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target.data)
        kl_loss1 = F.kl_div(pred.log(), new_target.data)
        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)
        # Total_loss
        loss_con = self.loss_contrasitive(G, NG, target)
        if epoch < 400:
            total_loss = reconstr_loss + 0.2*(0.5*kl_loss1+0.5*kl_loss)+ loss_d1 + loss_d2
        else:
            total_loss = reconstr_loss + 0.2*(0.5*kl_loss1+0.5*kl_loss)+ loss_d1 + loss_d2 + 0.7*loss_con
        return total_loss, reconstr_loss, kl_loss1, loss_d1, loss_d2, loss_con, kl_loss

    def loss_contrasitive(self, G1, G2, Z):
        Z = torch.nn.functional.normalize(Z, dim=1)

        H1 = G1.reshape(-1)
        Positive = torch.index_select(Z, 0, H1)
        Positive = torch.reshape(Positive, (Z.shape[0], args.lnp, Z.shape[1]))
        H2 = G2.reshape(-1)
        Negative = torch.index_select(Z, 0, H2)
        Negative = torch.reshape(Negative, (Z.shape[0], args.outp, Z.shape[1]))
        Z = torch.unsqueeze(input=Z, dim=1)

        Positive = torch.permute(input=Positive, dims=(0, 2, 1))
        Negative = torch.permute(input=Negative, dims=(0, 2, 1))

        positive_value = torch.matmul(Z, Positive)
        negative_value = torch.matmul(Z, Negative)
        positive_value = torch.exp(positive_value / 1)
        negative_value = torch.exp(negative_value / 1)
        positive_value = torch.sum(positive_value, dim=-1)
        negative_value = torch.sum(negative_value, dim=-1)
        loss = positive_value / (positive_value + negative_value)
        loss = -torch.log(loss)
        loss = loss.mean()
        return loss

def refined_subspace_affinity(s):
    weight = s ** 2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model):
    train_loader = DataLoader(
        dataset, batch_size=4096, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.001)
    for epoch in range(30):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.type(torch.float32)
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))


def train_EDESC():
    model = C_EDESC(
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        pretrain_path=args.pretrain_path).to(device)
    start = time.time()
    data = dataset.x
    y = dataset.y
    scaler = StandardScaler()
    G = get_hypergraph(data, args.lnp)
    NG = get_neg_hypergraph(data, args.outp)
    model.pretrain('')
    optimizer = Adam(model.parameters(), lr=args.lr)
    data = dataset.train
    index = dataset.index
    data = torch.Tensor(data).to(device)
    G = torch.tensor(G).to(device)
    NG = torch.tensor(NG).to(device)

    x_bar, hidden = get_initial_value(model, data)

    random_seed = random.randint(1,10000)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=30,random_state=random_seed)

    # Get clusters from K-means
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy().reshape(dataset.__len__(), -1))
    print("Initial Cluster Centers: ", y_pred)

    # Initialize D
    D = Initialization_D(hidden.reshape(dataset.__len__(), -1), y_pred, args.n_clusters, args.d)
    D = torch.tensor(D).to(torch.float32)
    accmax = 0
    nmimax = 0
    kappa_max=0
    y_pred_last = y_pred
    torch.cuda.empty_cache()
    model.D.data = D.to(device)
    model.train()

    for epoch in range(epochs):
        x_bar, s, z = model(data)
        # Update refined subspace affinity
        tmp_s = s
        s_tilde = refined_subspace_affinity(tmp_s)
        # Evaluate clustering performance
        y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred

        y_best, acc, kappa, nmi,_ = cluster_accuracy(y, y_pred, return_aligned=True)

        if acc > accmax:
            accmax = acc
            kappa_max = kappa
            nmimax = nmi

        ############## Total loss function ######################
        total_loss, reconstr_loss, kl_loss1, loss_d1, loss_d2, loss_smooth, kl_loss = model.total_loss(data, x_bar, z,
                                                                                                         pred=s,
                                                                                                         target=s_tilde,
                                                                                                         dim=args.d,
                                                                                                         n_clusters=args.n_clusters,
                                                                                                         index=index,
                                                                                                         epoch=epoch,
                                                                                                         G=G, NG=NG,
                                                                                                         y = y)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('Iter {}'.format(epoch), ':Current Acc {:.4f}'.format(acc),
                  ':Max Acc {:.4f}'.format(accmax), ', Current nmi {:.4f}'.format(nmi),
                  ':Max nmi {:.4f}'.format(nmimax),', Current kappa {:.4f}'.format(kappa),
                  ':Max kappa {:.4f}'.format(kappa_max))
            print("total_loss", total_loss.data, "reconstr_loss", reconstr_loss.data, "kl_loss", kl_loss.data,
                  "loss_smooth", loss_smooth.data, "kl_loss", kl_loss.data)
    end = time.time()
    print('Running time: ', end - start)
    return accmax, nmimax, kappa_max


import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == "__main__":
    print("hello world")

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lnp', type=int, default=20)
    parser.add_argument('--outp', type=int, default=100)
    parser.add_argument('--n_clusters', default=9, type=int)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--n_z', default=32, type=int)
    parser.add_argument('--eta', default=5, type=int)
    parser.add_argument('--dataset', type=str, default='pavia')
    parser.add_argument('--pretrain_path', type=str, default='data/pre.pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.n_clusters = 9
    args.n_input = 8
    args.image_size = [610, 340]
    dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/pavia/PaviaU.mat",
                              "/home/xianlli/dataset/HSI/pavia/PaviaU_gt.mat")
    args.num_sample = dataset.__len__()
    acc, nmi,kappa = train_EDESC()
    print('ACC {:.4f}'.format(acc), 'nmi {:4f}'.format(nmi),' kappa {:4f}'.format(kappa))









