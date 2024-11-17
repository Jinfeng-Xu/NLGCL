
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class NLGCL(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NLGCL, self).__init__(config, dataset)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']  # bool type: whether to require pow when regularization

        self.cl_temp = config['cl_temp']
        self.cl_reg = config['cl_reg']
        self.alpha = config['alpha']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for _ in range(self.n_layers):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)

        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list


    def InfoNCE(self, view1, view2, view):
        view1, view2, view = F.normalize(view1), F.normalize(view2), F.normalize(view)
        pos_score = torch.mul(view1, view2).sum(dim=1)
        pos_score = torch.exp(pos_score / self.cl_temp)
        ttl_score = torch.matmul(view1, view.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.cl_temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score).sum()
        return cl_loss

    def neighbor_cl_loss(self, embeddings_list, user, pos_item, neg_item):
        ego_embedding_u, ego_embedding_i = torch.split(embeddings_list[0], [self.n_users, self.n_items])
        cl_u = 0
        cl_i = 0
        for layer_idx in range(1, self.n_layers + 1):
            cur_embedding_u, cur_embedding_i = torch.split(embeddings_list[layer_idx], [self.n_users, self.n_items])
            # user side
            # pos_dis_u = self.InfoNCE(ego_embedding_u[user], cur_embedding_i[pos_item],
            #                            ego_embedding_u[user])
            # neg_dis_u = 0. * self.InfoNCE(ego_embedding_u[user], cur_embedding_i[neg_item],
            #                            ego_embedding_u[user])
            pos_dis_u = self.InfoNCE(cur_embedding_i[pos_item], ego_embedding_u[user],
                                     ego_embedding_u[user])
            neg_dis_u = 0. * self.InfoNCE(cur_embedding_i[neg_item], ego_embedding_u[user],
                                          ego_embedding_u[user])
            cl_u = cl_u + self.pair_loss(pos_dis_u, neg_dis_u)
            #item side
            # pos_dis_i = self.InfoNCE(ego_embedding_i[pos_item], cur_embedding_u[user],
            #                            ego_embedding_i[pos_item])
            # neg_dis_i = 0. * self.InfoNCE(ego_embedding_i[neg_item], cur_embedding_u[user],
            #                            ego_embedding_i[neg_item])
            pos_dis_i = self.InfoNCE(cur_embedding_u[user], ego_embedding_i[pos_item],
                                     ego_embedding_i[pos_item])
            neg_dis_i = 0. * self.InfoNCE(cur_embedding_u[user], ego_embedding_i[neg_item],
                                          ego_embedding_i[neg_item])
            cl_i = cl_i + self.pair_loss(pos_dis_i, neg_dis_i)
            # update embeddings
            ego_embedding_u, ego_embedding_i = cur_embedding_u, cur_embedding_i
        return cl_u, cl_i

    def pair_loss(self, pos, neg):
        res = pos - neg + 1e-6
        return res

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        # calculate ego CL Loss
        ego_cl_loss_u, ego_cl_loss_i = self.neighbor_cl_loss(embeddings_list, user, pos_item, neg_item)
        ego_cl_loss = self.alpha * ego_cl_loss_u + (1 - self.alpha) * ego_cl_loss_i

        return mf_loss, self.reg_weight * reg_loss, ego_cl_loss * self.cl_reg


    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
