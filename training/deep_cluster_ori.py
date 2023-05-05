import torch
import sys
import numbers
import argparse
import numpy as np
import torch.nn as nn

from collections import OrderedDict
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import time
import yaml
sys.path.insert(0, 'AL_SRE/support')


##kmeans
def compute_dist(data1,data2):
    ##torch version
    r = data1.unsqueeze(2) - data2.transpose(0, 1)
    # dis_mat = torch.sqrt(torch.sum(r ** 2,dim=1))
    dis_mat = torch.sum(r ** 2.0, dim=1)
    min_value = torch.min(torch.min(dis_mat, dim=1).values, dim=0).values  # [128]
    max_value = torch.max(torch.max(dis_mat, dim=1).values, dim=0).values

    dist1 = dis_mat - min_value
    dist2 = max_value - min_value

    dis_mat = torch.div(dist1, dist2)  # [32,200,128]
    return dis_mat

def get_label(X,centers): ##torch done
    """ Assign samples in `X` to clusters """
    dis_mat = compute_dist(X,centers)
    pseudo_label = torch.argmin(dis_mat,dim=1)
    return pseudo_label

##autoencoder
class AutoEncoder(nn.Module):

    def __init__(self, in_dim,n_clusters,latent_dim=512,hidden_dims=[500,500,2000],):
        super(AutoEncoder, self).__init__()

        self.input_dim = in_dim #784
        self.output_dim = self.input_dim
        self.hidden_dims = hidden_dims #[500,500,2000]
        self.latent_dim = latent_dim
        self.hidden_dims.append(self.latent_dim) #[500,500,2000,10]
        self.dims_list = (self.hidden_dims + self.hidden_dims[:-1][::-1])  # mirrored structure [500,500,2000,10,2000,500,500]
        self.n_layers = len(self.dims_list) #7

        self.n_clusters = n_clusters #10

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == self.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim), #[784,500]
                        'activation0': nn.ReLU()
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim), #[500,500], [500,2000], [2000,10]
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx])
                    }
                )
        self.encoder = nn.Sequential(layers) #[768,10]

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1] #[10,2000,500,500]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim), #[500,784]
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1]), #[10,2000], [2000,500], [500,500]
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1])
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output),output


class DCN(nn.Module):

    def __init__(self, in_dim,n_clusters,latent_dim=512,hidden_dims=[500,500,2000],beta=1.0,
                 lamda=1.0,pre_epoch=10,log_interval=100,lr=1e-4,wd=5e-4,gamma=0.01):
        super(DCN, self).__init__()

        # self.in_dim = in_dim
        self.beta = beta  # coefficient of the clustering term #1
        self.lamda = lamda  # coefficient of the reconstruction term #1
        # self.epoch = pre_epoch
        self.input_dim = in_dim
        self.hidden_dims = hidden_dims
        # self.log_interval = log_interval
        self.latent_dim = latent_dim
        # self.cluster_frames = int(cluster_frames)

        self.n_clusters = n_clusters
        # self.lr = lr
        # self.wd = wd
        self.gamma = gamma
        self.eps =1e-10

        # self.loss_method = loss_method

        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))

        if not self.lamda > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))

        if len(self.hidden_dims) == 0: #[500,500,2000]
            raise ValueError('No hidden layer specified.')

        # self.kmeans = batch_KMeans(n_clusters,latent_dim)
        self.autoencoder = AutoEncoder(in_dim,n_clusters,latent_dim,hidden_dims) #.cuda()

        self.mse_loss = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,weight_decay=self.wd)

        # self.classier = nn.Linear(self.latent_dim,self.n_clusters)
        # self.loss_classier = nn.CrossEntropyLoss()
        self.clusters = torch.nn.Parameter(torch.randn(self.n_clusters,self.latent_dim))

    # def init_cluster(self, X):
    #     """ Generate initial clusters using sklearn.Kmeans """
    #     model = KMeans(n_clusters=self.n_clusters,n_init=20)
    #     model.fit(X)
    #     clusters = model.cluster_centers_  # copy clusters
    #     self.clusters = torch.FloatTensor(clusters)

    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, latent_X, cluster_id):
        batch_size = latent_X.size()[0]
        clusters = self.clusters
        clusters_all = torch.index_select(clusters,0,cluster_id)

        ##差值
        subtrac = latent_X - clusters_all
        var = torch.sum(subtrac **2,dim=1,keepdim=True)
        var2 = 0.5 * self.beta * var
        dist_loss = torch.sum(var2)
        dist_loss = dist_loss / batch_size
        return dist_loss

    def kmeans_iteration(self, X,j, tol=1e-4, iter_limit=10):
        # running soft-kmeans  X: N * D tensor
        gamma = self.gamma
        iteration = 0
        # initial_state = self.clusters.clone()
        initial_state = self.clusters.data[j, :].clone()
        while True:
            # dis = compute_dist(X,initial_state)  # [B,C] 每个embedding与center距离矩阵
            dis = compute_dist(X, self.clusters)
            # initial_state = self.clusters
            # soft_prediction size: N * num_clusters #[B,C]
            soft_prediction = (torch.exp(-gamma * dis).T / torch.sum(torch.exp(-gamma * dis), dim=1)+self.eps).T
            # soft_prediction = (torch.exp(-dis).T / torch.sum(torch.exp(-dis), dim=1) + self.eps).T
            # soft_prediction = F.softmax(soft_prediction)
            ##embedding与每个center距离  占   到所有center距离和比例  [C,B]
            initial_state_pre = initial_state.clone()  # C*D
            # centers size: num_clusters, D     #
            # ss = torch.sum(soft_prediction, dim=0,keepdim=True)
            initial_state = (torch.matmul(soft_prediction.T, X).T / torch.sum(soft_prediction, dim=0)+self.eps).T
            center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

            self.clusters.data[j, :] = initial_state.data[j, :]
            iteration = iteration + 1
            # center_shift_value = float(center_shift.data)
            if center_shift ** 2 < tol:
                break
            if iter_limit != 0 and iteration >= iter_limit:
                break

    def train_batch_kmeans(self, embeddings,labels):
        # embedding: [B * W] labels: [B]
        # speaker_ids = torch.unique(labels).view(n_clusters, -1)  ##不能整除时？
        pseudo_ids = torch.unique(labels).unsqueeze(1)
        n_clusters = pseudo_ids.size()[0]

        for i in range(n_clusters):
            j = pseudo_ids[i]
            batch_set = torch.nonzero(labels == j).squeeze(1)  # 求出每个类别labels对应位置
            embedding_set, label_set = embeddings[batch_set], labels[batch_set]
            self.kmeans_iteration(embedding_set, j)

    # def get_tsne(self, data,verbose=True):
    #     # print("data:",data.shape)
    #     # batch_size = data.size()[0]
    #     data = data.transpose(1,2)
    #     # dim_size = data.size()[-1]
    #     data = data.contiguous().view(-1, self.input_dim)
    #
    #     # Get the latent features
    #     with torch.no_grad():
    #         latent_X = self.autoencoder(data, latent=True)
    #
    #     # [Step-1] Update the assignment results
    #     cluster_id = get_label(latent_X, self.clusters)
    #     new_embeddings, latent_embeddings = self.autoencoder.forward(data)  ##重构embedding
    #
    #     return latent_embeddings,cluster_id

    def fit(self, data,verbose=True):
        # print("data:",data.shape) [200,2560,25]
        # batch_size = data.size()[0]
        data = data.transpose(1,2)
        # dim_size = data.size()[-1]
        data = data.contiguous().view(-1, self.input_dim)

        # Get the latent features
        with torch.no_grad():
            latent_X = self.autoencoder(data, latent=True)

        # [Step-1] Update the assignment results
        cluster_id = get_label(latent_X, self.clusters)
        new_embeddings, latent_embeddings = self.autoencoder.forward(data)  ##重构embedding
        # predictions, batch_label = self.train_batch_kmeans(latent_embeddings, cluster_id) ##更新center
        self.train_batch_kmeans(latent_embeddings, cluster_id)
        # predictions = compute_dist(latent_embeddings,self.clusters)
        # predictions = F.softmax(predictions)
        # ppl_loss = self.loss_func(predictions, cluster_id)
        ppl_loss = self._loss(latent_embeddings, cluster_id)
        rec_loss = self.mse_loss(data, new_embeddings)
        loss = self.lamda * rec_loss + self.beta * ppl_loss
        # print("ppl_loss,rec_loss,loss:", ppl_loss,rec_loss,loss)
        return loss

class DCN_ppl(nn.Module):

    def __init__(self, in_dim,n_clusters,gamma=0.01):
        super(DCN_ppl, self).__init__()

        self.input_dim = in_dim

        self.n_clusters = n_clusters

        self.gamma = gamma
        self.eps =1e-10

        # self.kmeans = batch_KMeans(n_clusters,latent_dim)

        self.clusters = torch.nn.Parameter(torch.randn(self.n_clusters,self.latent_dim))

    # def init_cluster(self, X):
    #     """ Generate initial clusters using sklearn.Kmeans """
    #     model = KMeans(n_clusters=self.n_clusters,n_init=20)
    #     model.fit(X)
    #     clusters = model.cluster_centers_  # copy clusters
    #     self.clusters = torch.FloatTensor(clusters)

    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, embedding, cluster_id):
        batch_size = embedding.size()[0]
        clusters = self.clusters
        clusters_all = torch.index_select(clusters,0,cluster_id)

        ##差值
        subtrac = embedding - clusters_all
        var = torch.sum(subtrac **2,dim=1,keepdim=True)
        var2 = 0.5 * self.beta * var
        dist_loss = torch.sum(var2)
        dist_loss = dist_loss / batch_size
        return dist_loss

    def kmeans_iteration(self, X,j, tol=1e-4, iter_limit=10):
        # running soft-kmeans  X: N * D tensor
        gamma = self.gamma
        iteration = 0
        # initial_state = self.clusters.clone()
        initial_state = self.clusters.data[j, :].clone()
        while True:
            # dis = compute_dist(X,initial_state)  # [B,C] 每个embedding与center距离矩阵
            dis = compute_dist(X, self.clusters)

            soft_prediction = (torch.exp(-gamma * dis).T / torch.sum(torch.exp(-gamma * dis), dim=1)+self.eps).T

            initial_state_pre = initial_state.clone()  # C*D

            initial_state = (torch.matmul(soft_prediction.T, X).T / torch.sum(soft_prediction, dim=0)+self.eps).T
            center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

            self.clusters.data[j, :] = initial_state.data[j, :]
            iteration = iteration + 1

            if center_shift ** 2 < tol:
                break
            if iter_limit != 0 and iteration >= iter_limit:
                break

    def train_batch_kmeans(self, embeddings,labels):
        pseudo_ids = torch.unique(labels).unsqueeze(1)
        n_clusters = pseudo_ids.size()[0]

        for i in range(n_clusters):
            j = pseudo_ids[i]
            batch_set = torch.nonzero(labels == j).squeeze(1)  # 求出每个类别labels对应位置
            embedding_set, label_set = embeddings[batch_set], labels[batch_set]
            self.kmeans_iteration(embedding_set, j)

    def fit(self, data):

        data = data.transpose(1,2)
        # dim_size = data.size()[-1]
        data = data.contiguous().view(-1, self.input_dim)

        # [Step-1] Update the assignment results
        cluster_id = get_label(data, self.clusters)
        self.train_batch_kmeans(data, cluster_id)

        ppl_loss = self._loss(data, cluster_id)

        return ppl_loss
