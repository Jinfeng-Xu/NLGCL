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
    """NLGCL: A Contrastive Learning between Neighbor Layers for Graph Collaborative Filtering"""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NLGCL, self).__init__(config, dataset)

        # Load model hyperparameters
        self.latent_dim = config['embedding_size']  # Embedding dimension size
        self.n_layers = config['n_layers']          # Number of GCN layers
        self.reg_weight = config['reg_weight']      # L2 regularization weight
        self.require_pow = config['require_pow']    # Whether to use powered regularization
        
        # Contrastive learning hyperparameters
        self.cl_temp = config['cl_temp']            # Temperature for contrastive loss
        self.cl_reg = config['cl_reg']              # Weight for contrastive loss
        self.alpha = config['alpha']                # Weight for user/item contrastive loss balance

        # Initialize model components
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        
        # Initialize loss functions
        self.mf_loss = BPRLoss()    # Bayesian Personalized Ranking loss
        self.reg_loss = EmbLoss()    # Embedding regularization loss

        # Cache variables for full ranking acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # Parameter initialization and tracking
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_ego_embeddings(self):
        """Combine user and item embeddings into a single matrix"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        return torch.cat([user_emb, item_emb], dim=0)

    def forward(self):
        """Perform graph convolutional forward propagation"""
        # Start with ego embeddings (initial embeddings)
        all_emb = self.get_ego_embeddings()
        emb_list = [all_emb]  # Store embeddings from all layers
        
        # Perform multi-layer graph convolution
        for _ in range(self.n_layers):
            all_emb = self.gcn_conv(all_emb, self.edge_index, self.edge_weight)
            emb_list.append(all_emb)
        
        # Combine embeddings from all layers using mean pooling
        lightgcn_emb = torch.stack(emb_list, dim=1)
        lightgcn_emb = torch.mean(lightgcn_emb, dim=1)
        
        # Split combined embeddings into user and item embeddings
        user_emb, item_emb = torch.split(lightgcn_emb, [self.n_users, self.n_items])
        return user_emb, item_emb, emb_list

    def InfoNCE(self, anchor, positive, all_samples):
        """Compute InfoNCE contrastive loss"""
        # Normalize embeddings for cosine similarity calculation
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        all_samples = F.normalize(all_samples, dim=1)
        
        # Positive sample similarity
        pos_score = torch.sum(anchor * positive, dim=1)
        pos_score = torch.exp(pos_score / self.cl_temp)
        
        # Negative sample similarity (against all samples)
        ttl_score = torch.matmul(anchor, all_samples.t())
        ttl_score = torch.exp(ttl_score / self.cl_temp).sum(dim=1)
        
        # Compute final contrastive loss
        return -torch.log(pos_score / ttl_score).sum()

    def neighbor_cl_loss(self, emb_list, users, pos_items, neg_items):
        """Compute neighbor layer contrastive loss"""
        # Get initial embeddings (layer 0)
        user_emb_0, item_emb_0 = torch.split(emb_list[0], [self.n_users, self.n_items])
        cl_user_loss = 0.0
        cl_item_loss = 0.0

        # Compute contrastive loss at each layer
        for layer_idx in range(1, self.n_layers + 1):
            # Get current layer embeddings
            user_emb_k, item_emb_k = torch.split(emb_list[layer_idx], [self.n_users, self.n_items])
            
            # User-side contrastive loss
            cl_user_loss = self.InfoNCE(
                anchor=item_emb_k[pos_items],
                positive=user_emb_0[users],
                all_samples=user_emb_0[users]
            ) + 1e-6

            
            # Item-side contrastive loss
            cl_item_loss = self.InfoNCE(
                anchor=user_emb_k[users],
                positive=item_emb_0[pos_items],
                all_samples=item_emb_0[pos_items]
            ) + 1e-6
            
            # Update reference embeddings for next layer
            user_emb_0, item_emb_0 = user_emb_k, item_emb_k

        return cl_user_loss, cl_item_loss

    def calculate_loss(self, interaction):
        """Calculate total loss for training"""
        # Clear cache during training
        self.restore_user_e = None
        self.restore_item_e = None

        # Get batch data
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        # Forward propagation
        user_all_emb, item_all_emb, emb_list = self.forward()
        
        # Compute BPR loss
        u_emb = user_all_emb[users]
        pos_emb = item_all_emb[pos_items]
        neg_emb = item_all_emb[neg_items]
        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        bpr_loss = self.mf_loss(pos_scores, neg_scores)
        
        # Compute regularization loss
        reg_loss = self.reg_loss(
            self.user_embedding(users),
            self.item_embedding(pos_items),
            self.item_embedding(neg_items),
            require_pow=self.require_pow
        )
        
        # Compute contrastive loss
        cl_u, cl_i = self.neighbor_cl_loss(emb_list, users, pos_items, neg_items)
        cl_loss = self.alpha * cl_u + (1 - self.alpha) * cl_i
        
        # Return weighted components of total loss
        return bpr_loss, self.reg_weight * reg_loss, self.cl_reg * cl_loss

    def predict(self, interaction):
        """Predict interaction scores for user-item pairs"""
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        
        # Get embeddings and compute dot products
        user_emb, item_emb, _ = self.forward()
        u_emb = user_emb[users]
        i_emb = item_emb[items]
        return torch.mul(u_emb, i_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        """Predict scores for all items for given users"""
        users = interaction[self.USER_ID]
        
        # Cache embeddings for efficiency
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()
        
        # Compute scores via matrix multiplication
        user_emb = self.restore_user_e[users]
        return torch.matmul(user_emb, self.restore_item_e.t()).flatten()