'''
@Project: DFP-GNN
@File   : ops_pt
@Time   : 2021/8/26 21:13
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Module of the pretraining process
'''
import os
import time
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from torch.optim import RMSprop
from sklearn.cluster import KMeans

from models.sgae import SGAE
from models.dfp_gnn import DFPGNN
from utils.ops_io import load_data_graph


def pretraining(pre_train, args):

    labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list = load_data_graph(
            direction_path=args.direction, dataset_name=args.dataset_name, normalization=args.normalization,
            load_saved=True, k_nearest_neighobrs=args.pm_knns)

    args.num_classes = len(np.unique(labels))
    args.num_views = len(feature_list)
    view_dims = []
    for j in range(args.num_views):
        view_dims.append(feature_list[j].shape[1])
    print(view_dims)
    pm_hidden_dims = [args.pm_first_dim, args.pm_second_dim, args.pm_third_dim]
    sm_hidden_dims = [args.sm_first_dim, args.sm_second_dim, args.sm_third_dim]
    am_hidden_dims = [args.am_first_dim, args.am_second_dim]

    # pretraining or loading the weights of DFP-GNN
    ec_feat_save_direction = './data/ec_feature/'
    if not os.path.exists(ec_feat_save_direction):
        os.makedirs(ec_feat_save_direction)
    ec_feat_save_path = ec_feat_save_direction + args.dataset_name + '.npy'
    pt_weight_save_direction = './data/pt_weight/'
    if not os.path.exists(pt_weight_save_direction):
        os.makedirs(pt_weight_save_direction)
    pt_weight_save_path = pt_weight_save_direction + args.dataset_name + '.pkl'

    print("############### begin to pretraining all submodules ###############")
    model = DFPGNN(view_dims, pm_hidden_dims, am_hidden_dims, sm_hidden_dims, args.num_classes)

    pt_begin_time = time.time()

    ec_feature, adj_list = pre_training_pm(pre_train, args.dataset_name, model=model, feature_list=feature_list, adj_wave_list=adj_wave_list,
                                 adj_hat_list=adj_hat_list, norm_list=norm_list,
                                 weight_tensor_list=weight_tensor_list, optimizer_type=args.pt_pm_optimizer,
                                 learning_rate=args.pt_pm_lr, momentum=args.pt_pm_momentum,
                                 weight_decay=args.pt_pm_weight_decay, num_epochs=args.pt_pm_num_epochs,
                                 sp_weight=args.pt_pm_sp_weight, max_loss_patience=args.pt_pm_loss_patience,
                                 show_patience=args.pt_pm_show_patience)


    pt_cost_time = time.time() - pt_begin_time
    print("Pretraining time: ", pt_cost_time)
    torch.save(model.state_dict(), pt_weight_save_path)


    return adj_list


def pre_training_pm(pre_train, dataset_name, model, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list, optimizer_type,
                    learning_rate, momentum, weight_decay, num_epochs, sp_weight, max_loss_patience, show_patience):

    print("###################### Pretraining the preliminary module ######################")
    embedding_list = []
    adj_list = []
    for i in range(model.num_views):
        print("Begin to train the " + str(i+1) + " initial autoencoder ...")
        feature = feature_list[i].cuda()
        adj_wave = adj_wave_list[i].cuda()
        adj_hat = adj_hat_list[i].cuda()
        norm = norm_list[i]
        weight_tensor = weight_tensor_list[i].cuda()

        # construct the dimension list for each view
        temp_dims = []
        temp_dims.append(model.view_dims[i])
        temp_dims.extend(model.pm_hidden_dims)

        # construct the SGAE model for each view
        sgae = SGAE(dims=temp_dims, act_func=nn.ReLU()).cuda()
        if pre_train == True:
            gnew_weight_save_path = './data/graph_new_weight/' + dataset_name + str(i) + '.pkl'
            sgae.load_state_dict(torch.load(gnew_weight_save_path))
        # construct the optimizer and reconstructed loss function
        if optimizer_type == "RMSprop":
            optimizer = RMSprop(sgae.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = RMSprop(sgae.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        loss_function = nn.MSELoss()

        # begin to train...
        best_loss = float("inf")
        loss_patience = 0
        best_epoch = 0
        for epoch in range(num_epochs):
            sgae.train()
            hidden, X_bar, A_bar = sgae(feature, adj_hat)

            optimizer.zero_grad()
            loss = sp_weight * norm * F.binary_cross_entropy(A_bar.view(-1), adj_wave.to_dense().view(-1), weight=weight_tensor)
            loss += loss_function(X_bar, feature)
            loss.backward()
            optimizer.step(closure=None)

            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_epoch = epoch + 1
                loss_patience = 0
            else:
                loss_patience += 1



        # obtain the embedded feature
        sgae.eval()
        hidden, fea_bar, adj_bar = sgae(feature, adj_hat)
        embedding = hidden.detach().cpu().numpy()
        embedding_list.append(embedding)
        adj = adj_bar.detach().cpu().numpy()
        adj_list.append(adj)

        gnew_weight_save_direction = './data/graph_new_weight/'
        if not os.path.exists(gnew_weight_save_direction):
            os.makedirs(gnew_weight_save_direction)
        gnew_weight_save_path = gnew_weight_save_direction + dataset_name + str(i) + '.pkl'
        torch.save(sgae.state_dict(), gnew_weight_save_path)

        # copy the trained weights to the corresponding part of the DFP-GNN model
        model_dict = model.get_preliminary_ae(i).state_dict()
        model_dict.update(sgae.encoder.state_dict())
        model.get_preliminary_ae(i).load_state_dict(model_dict)

    np_list = np.array(embedding_list)
    ec_feature = np.mean(np_list, axis=0)


    return ec_feature, adj_list

