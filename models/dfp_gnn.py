
import torch.nn as nn

from layers.mlfpn_gcn import MLFPN_GCN
from layers.fusion_layer import FusionLayer
from layers.cluster_layer import ClusterLayer
from utils.ops_al import dot_product_decode
from layers.mlfpn_fc import MLFPN_FC
from utils.graph_concat import all_fg_consstruct
import torch

class DFPGNN(nn.Module):

    def __init__(self, view_dims, pm_hidden_dims, am_hidden_dims, sm_hidden_dims, num_clusters):
        super(DFPGNN, self).__init__()
        self.view_dims = view_dims
        self.pm_hidden_dims = pm_hidden_dims
        self.sm_hidden_dims = sm_hidden_dims
        self.num_views = len(view_dims)

        # define the preliminary module for all views
        self.preliminary_module = nn.ModuleList()
        self.decoder_module = nn.ModuleList()
        for i in range(self.num_views):
            temp_dims = []
            temp_dims.append(view_dims[i])
            temp_dims.extend(pm_hidden_dims)
            self.preliminary_module.append(MLFPN_GCN(temp_dims, nn.ReLU()))
            self.decoder_module.append(MLFPN_FC(list(reversed(temp_dims)), nn.ReLU()))  #############加解码层，注意维度

        aemp_dims = []
        aemp_dims.extend(am_hidden_dims)
        self.full_graph_module = MLFPN_GCN(aemp_dims, nn.ReLU())
        # define the fusion module
        self.fusion_module = FusionLayer(num_views=self.num_views)

        # define the clustering layer
        self.cluster_layer = ClusterLayer(num_clusters, self.sm_hidden_dims[-1])

    def forward(self, feats, adjs):
        # computation in the preliminary module
        hidden_pr_list = []
        X_bar_list = []
        adj_bar_list = []
        for i in range(self.num_views):#graph autoencoder
            hidden_pr = self.preliminary_module[i](feats[i], adjs[i])
            hidden_pr_list.append(hidden_pr)
            X_bar = self.decoder_module[i](hidden_pr)
            X_bar_list.append(X_bar)
            adj_bar = dot_product_decode(hidden_pr)
            adj_bar_list.append(adj_bar)

        # computation in the fusion module
        combined_feature_pr = self.fusion_module(hidden_pr_list)#原始view-specific graph的混合graph

        #construct hegrou graph
        adj_all = all_fg_consstruct(hidden_pr_list, adj_bar_list, self.num_views)
        hidden_tensor_he = torch.stack(hidden_pr_list, 0)
        hidden_tensor_he = hidden_tensor_he.reshape(-1, self.pm_hidden_dims[-1])
        hidden_tensor_all = self.full_graph_module(hidden_tensor_he, adj_all)
        hidden_tensor_all = hidden_tensor_all.reshape(self.num_views, -1, self.pm_hidden_dims[-1])

        hidden_list_all = []
        for i in range(self.num_views):
            hidden_list_all.append(hidden_tensor_all[i])

        combined_feature = self.fusion_module(hidden_list_all)
        q = self.cluster_layer(combined_feature)

        return combined_feature, combined_feature_pr, q, X_bar_list, adj_bar_list

    def get_preliminary_ae(self, index: int) -> nn.Module:
        """
            return the index-th encoder of the initial ae module for initializing weights
        :param index: the index of current
        :return: the index-th encoder of the initial ae module
        """
        if index > len(self.preliminary_module) or index < 0:
            raise ValueError('Requested subautoencoder cannot be constructed, index out of range.')
        return self.preliminary_module[index]

    def get_preliminary_de(self, index: int) -> nn.Module:
        """
            return the index-th encoder of the initial ae module for initializing weights
        :param index: the index of current
        :return: the index-th encoder of the initial ae module
        """
        if index > len(self.decoder_module) or index < 0:
            raise ValueError('Requested subautoencoder cannot be constructed, index out of range.')
        return self.decoder_module[index]