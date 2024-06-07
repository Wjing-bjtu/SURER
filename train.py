'''
@Project: DFP-GNN
@File   : test_train
@Time   : 2021/9/12 15:10
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    main function 'pt_pm_lr', 0.0002), ('pm_knns', 50), ('ft_lr', 2e-05
'''
import os
import nni
import time
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from sklearn.cluster import KMeans
from utils.ops_al import target_distribution
from copy import deepcopy
from models.dfp_gnn_cl_nouse import DFPGNN
from utils.ops_pt import pretraining
from utils.ops_ev import get_evaluation_results
from utils.load_data_clusterft import load_data_clusterft
from utils.ops_io import load_data_graphft, load_data_graph
from utils.constra_loss import Instance_Loss


params = nni.get_next_parameter()

params = {'pt_pm_lr': 0.0001, 'pm_knns': 25, 'seed': 55164, 'ft_lr': 1e-05, 'ft_sp_weight': 3, 'pt_pm_sp_weight': 0.01, 'ft_num_epochs': 350, 'pt_pm_num_epochs': 450, 'n_repeated': 10, 'ft_update_interval': 80, 'ft_cl_weight': 0.01, 'ft_pl_weight': 0.5}
print('params',params)
def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


if __name__ == '__main__':
    # Configuration settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='2', help='The number of cuda device.')
    parser.add_argument('--seed', type=str, default=params['seed'], help='The number of cuda device.')
    parser.add_argument('--n_repeated', type=int, default=params['n_repeated'], help='Number of repeated experiments')#50
    parser.add_argument('--direction', type=str, default='./data/datasets/', help='direction of datasets')
    parser.add_argument('--dataset_name', type=str, default='100leaves', help='The dataset used for training/testing')
    parser.add_argument('--normalization', type=str, default='normalize', help='default normalize')
    parser.add_argument('--am_first_dim', type=int, default=256, help='the dim of the first layer in PM')
    parser.add_argument('--am_second_dim', type=int, default=256, help='the dim of the second layer in PM')
    parser.add_argument('--ft_sp_weight', type=float, default=params['ft_sp_weight'], help='weight of structure preservation loss')  # 0.001
    parser.add_argument('--ft_cl_weight', type=float, default=params['ft_cl_weight'], help='weight of clustering loss')
    parser.add_argument('--ft_pl_weight', type=float, default=params['ft_pl_weight'], help='weight of clustering loss')
    parser.add_argument('--ft_update_interval', type=int, default=params['ft_update_interval'],
                        help='weight of structure preservation loss')  # 50


    parser.add_argument('--pr_gcn', type=int, default=1, help='the epoch of the pretrain gcn')
    parser.add_argument('--pm_first_dim', type=int, default=512, help='the dim of the first layer in PM')
    parser.add_argument('--pm_second_dim', type=int, default=2048, help='the dim of the second layer in PM')
    parser.add_argument('--pm_third_dim', type=int, default=256, help='the dim of the third layer in PM')
    parser.add_argument('--pm_knns', type=int, default=params['pm_knns'], help='the number of nearest neighbors in PM')#50 params['pm_knns']
    parser.add_argument('--pm_pruning_one', action='store_true', default=True, help='Whether to use prunning one in PM')
    parser.add_argument('--pm_pruning_two', action='store_true', default=True, help='Whether to use prunning two in PM')
    parser.add_argument('--pm_common_neighbors', type=int, default=2, help='threshold of common neighbors in PM')
    parser.add_argument('--pt_pm_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--pt_pm_lr', type=float, default=params['pt_pm_lr'], help='learning rate in pretraining stage.')#0.00001 params['pt_pm_lr']
    parser.add_argument('--pt_pm_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--pt_pm_weight_decay', type=float, default=0.000001, help='value of layer-wise weight decay.')
    parser.add_argument('--pt_pm_num_epochs', type=int, default=params['pt_pm_num_epochs'], help='number of layer-wise training epochs.')#500
    parser.add_argument('--pt_pm_loss_patience', type=int, default=50, help='value of loss patience in pretraining')
    parser.add_argument('--pt_pm_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--pt_pm_sp_weight', type=float, default=params['pt_pm_sp_weight'], help='weight of structure preservation loss')#0.001

    parser.add_argument('--sm_first_dim', type=int, default=256, help='the dim of the first layer in SM')
    parser.add_argument('--sm_second_dim', type=int, default=64, help='the dim of the second layer in SM')
    parser.add_argument('--sm_third_dim', type=int, default=256, help='the dim of the third layer in SM')

    parser.add_argument('--fm_pruning_one', action='store_true', default=True, help='Whether to use prunning one in FM')
    parser.add_argument('--fm_pruning_two', action='store_true', default=True, help='Whether to use prunning two in FM')
    parser.add_argument('--fm_common_neighbors', type=int, default=2, help='threshold of common neighbors in FM')
    parser.add_argument('--pt_sm_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--pt_sm_lr', type=float, default=0.00001, help='learning rate in pretraining stage.')
    parser.add_argument('--pt_sm_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--pt_sm_weight_decay', type=float, default=0.000001, help='value of layer-wise weight decay.')
    parser.add_argument('--pt_sm_num_epochs', type=int, default=500, help='number of layer-wise training epochs.')#500
    parser.add_argument('--pt_sm_loss_patience', type=int, default=100, help='value of loss patience in pretraining')
    parser.add_argument('--pt_sm_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--pt_sm_sp_weight', type=float, default=0.001, help='weight of structure preservation loss')

    parser.add_argument('--ft_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--ft_lr', type=float, default=params['ft_lr'], help='learning rate in pretraining stage.')#params['pt_pm_lr']0.00001
    parser.add_argument('--ft_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--ft_weight_decay', type=float, default=0.00001, help='value of layer-wise weight decay.')
    parser.add_argument('--ft_num_epochs', type=int, default=params['ft_num_epochs'], help='number of layer-wise training epochs.')#1000
    parser.add_argument('--ft_loss_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--ft_show_patience', type=int, default=100, help='value of show patience in pretraining')



    parser.add_argument('--ft_tolerance', type=float, default=0.0000001, help='weight of clustering loss')

    parser.add_argument('--save_patience', type=int, default=20001, help='the frequency about saving the result')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True




    setup_seed(args.seed)
    '''
    @Project: DFP-GNN
    @File   : ops_ft
    @Time   : 2021/8/26 21:10
    @Author : Shunxin Xiao
    @Email  : xiaoshunxin.tj@gmail
    @Desc
        Finetune module
    '''


    def finetune(weights, best_val_acc, model, feature_list, adj_hat_list, adj_hat, adj_wave, norm, weight_tensor,
                 optimizer_type,
                 learning_rate, weight_decay, momentum, num_epochs, sp_weight, show_patience, update_interval,
                 cl_weight, tolerance, max_loss_patience, labels, save_patience, dataset_name,pl_weight):
        print("###################### Finetune the whole DFP-GNN model ######################")
        model.cuda()
        for i in range(model.num_views):
            feature_list[i] = feature_list[i].cuda()
            adj_hat_list[i] = adj_hat_list[i].to_dense().cuda()
            adj_wave[i] = adj_wave[i].cuda()
        adj_hat = adj_hat.cuda()
        # adj_wave = adj_wave.cuda()
        weight_tensor = weight_tensor.cuda()

        # construct the optimizer and reconstructed loss function
        if optimizer_type == "RMSprop":
            optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        loss_function = nn.MSELoss()

        model.eval()
        _, hidden, _, _, _, _ = model(feature_list, adj_hat_list)
        embedding = hidden.detach().cpu().cpu()
        kmeans = KMeans(n_clusters=len(np.unique(labels)), n_init=5)  # n_jobs=8
        y_pred_pr = kmeans.fit_predict(embedding)
        ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred_pr)
        print("Initial ACC score={:.4f}".format(ACC))
        y_pred_last = y_pred_pr
        criterion = Instance_Loss(feature_list[0].shape[0], 0.2).cuda()

        # begin to train...
        best_loss = float("inf")
        loss_patience = 0
        best_epoch = 0

        List_ID = []
        List_loss = []
        List_lr = []
        List_ls = []
        List_lc = []
        List_ACC = []
        List_NMI = []
        List_Purity = []
        List_ARI = []
        List_P = []
        List_R = []
        List_F = []

        for epoch in range(num_epochs):
            model.train()

            if epoch % update_interval == 0:
                _, _, tmp_q, _, _, _ = model(feature_list, adj_hat_list)

                # update target distribution p
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                # evaluate clustering performance
                y_pred = tmp_q.cpu().numpy().argmax(1)
                # print(",".join(str(x) for x in y_pred))
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

            hidden, hidden_pr, q, X_bar_list, A_bar_list, hidden_list_all = model(feature_list, adj_hat_list)
            ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(ACC), ', f1 {:.4f}'.format(F1))
            nni.report_intermediate_result(0)

            if ACC > best_val_acc:
                best_val_acc = ACC
                # self.best_graph = normalized_adj.detach()
                weights = deepcopy(model.state_dict())

            optimizer.zero_grad()
            lossr_list = []
            losss_list = []
            # lossg_list = []
            lossc_list = []
            losscg_list = []
            for v in range(model.num_views):
                lossr_list.append(loss_function(feature_list[v], X_bar_list[v]))
                losss_list.append(
                    sp_weight * F.binary_cross_entropy(A_bar_list[v].view(-1), adj_wave[v].to_dense().view(-1)))
                # lossg_list.append(torch.norm(A_bar_list[v] - A_bar_list[v].t(), p="fro"))
                # lossg_list.append(torch.norm(A_bar_list[v], 1))
                for w in range(v + 1, model.num_views):
                    lossc_list.append( criterion.forward_feature(hidden_list_all[v], hidden_list_all[w]))
                    # losscg_list.append(criterion.forward_feature(A_bar_list[v], A_bar_list[w]))

            loss_lr = sum(lossr_list)
            loss_ls = sum(losss_list)
            # loss_g = sum(lossg_list)
            loss_cl = sum(lossc_list)
            # loss_gl = sum(losscg_list)
            loss_lc = cl_weight * F.kl_div(q.log(), p)
            loss_func  = nn.CrossEntropyLoss(reduction="none")
            loss_lp = pl_weight * loss_func(q, torch.tensor(y_pred_pr).long().cuda())     #定义损失函数
            loss_lp = torch.mean(loss_lp)
            loss = loss_ls + loss_lr + loss_lc +loss_lp +loss_cl #+ loss_gl
            loss.backward()
            optimizer.step(closure=None)

            loss_value = float(loss.item())
            value_lr = float(loss_lr.item())
            value_ls = float(loss_ls.item())
            value_lc = float(loss_lc.item())

            if (epoch + 1) % save_patience == 0:
                List_ID.append(epoch + 1)
                List_loss.append(loss_value)
                List_lr.append(value_lr)
                List_ls.append(value_ls)
                List_lc.append(value_lc)
                with torch.no_grad():
                    _, _, _, _, _, pred_q, _ = model(feature_list, adj_hat_list)
                    predicted = pred_q.data.cpu().numpy().argmax(1)
                    ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), predicted)
                    List_ACC.append(ACC)
                    List_NMI.append(NMI)
                    List_Purity.append(Purity)
                    List_ARI.append(ARI)
                    List_P.append(P)
                    List_R.append(R)
                    List_F.append(F1)

            print(loss_value)
            if (epoch + 1) % show_patience == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_value))

        #################saving the node representation
        model.eval()
        model.load_state_dict(weights)
        _, _, q, X_bar_list, A_bar_list, _ = model(feature_list, adj_hat_list)

        return q.data.cpu().numpy().argmax(1), weights, best_val_acc


    del_files("./data/adj_matrix/")
    del_files("./data/ec_feature/")
    del_files("./data/graph_new_weight/")
    del_files("./data/pt_weight/")


    all_ACC = []
    all_NMI = []
    all_Purity = []
    all_ARI = []
    all_F = []
    all_P = []
    all_R = []
    all_PT_TIME = []
    all_FT_TIME = []
    pre_train = False
    best_ft_weight = None
    best_acc = 0
    # for i in range(args.pr_gcn):
    #     args.pt_pm_num_epochs = 1000
    #     adj_list_new = pretraining(pre_train, args)

    for i in range(args.n_repeated):
       # args.pt_pm_num_epochs = 500
        if i > 0:
            pre_train = True
        adj_list_new = pretraining(pre_train, args)
     ##load_saved=True不需要改，预训练模型用来重构x和a相当于只训练图更新，finetune是联合图更新和异构图聚类一起学习
        labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list = load_data_clusterft(adj_list_new,
            direction_path=args.direction, dataset_name=args.dataset_name, normalization=args.normalization,
            load_saved=True, k_nearest_neighobrs=args.pm_knns, prunning_one=args.pm_pruning_one,
            prunning_two=args.pm_pruning_two, common_neighbors=args.pm_common_neighbors)

        # labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list = load_data_graph(
        #     direction_path=args.direction, dataset_name=args.dataset_name, normalization=args.normalization,
        #     load_saved=False, k_nearest_neighobrs=args.pm_knns, prunning_one=args.pm_pruning_one,
        #     prunning_two=args.pm_pruning_two, common_neighbors=args.pm_common_neighbors)

        args.num_classes = len(np.unique(labels))
        args.num_views = len(feature_list)
        view_dims = []
        for j in range(args.num_views):
            view_dims.append(feature_list[j].shape[1])
        print('view_dims',view_dims)
        pm_hidden_dims = [args.pm_first_dim, args.pm_second_dim, args.pm_third_dim]
        am_hidden_dims = [args.am_first_dim, args.am_second_dim]
        sm_hidden_dims = [args.sm_first_dim, args.sm_second_dim, args.sm_third_dim]

        # pretraining or loading the weights of DFP-GNN
        ec_feat_save_direction = './data/ec_feature/'
        if not os.path.exists(ec_feat_save_direction):
            os.makedirs(ec_feat_save_direction)
        ec_feat_save_path = ec_feat_save_direction + args.dataset_name + '.npy'
        pt_weight_save_direction = './data/pt_weight/'
        if not os.path.exists(pt_weight_save_direction):
            os.makedirs(pt_weight_save_direction)
        pt_weight_save_path = pt_weight_save_direction +args.dataset_name + '.pkl'
        ft_weight_save_direction = './data/ft_weight/'+args.dataset_name + '.pkl'

        # exit()
        print("############### loading the pretrained wieghts.... ###############")
        model = DFPGNN(view_dims, pm_hidden_dims,am_hidden_dims, sm_hidden_dims, args.num_classes)
        # if i > 0:
        #     model.load_state_dict(torch.load(ft_weight_save_direction))
        # else:
        #     model.load_state_dict(torch.load(pt_weight_save_path))
        # if i == 0:
        #     model.load_state_dict(torch.load(pt_weight_save_path))
        # else:
        #     model.load_state_dict(best_weight)
        model.load_state_dict(torch.load(pt_weight_save_path))

       # adj_wave, adj_hat, norm, weight_tensor = load_embedded_combined_data_new(args.dataset_name)#adj_wave, adj_hat，norm后续没用   weight_tensor也没用，损失函数已经改成不需要权重的了

        # finetune the whole DFP-GNN model
        ft_begin_time = time.time()#input adj adj_hat adj_wave is numpy
        ##best_ft_weight 全局变量
        predicted, best_ft_weight, best_val_acc = finetune(best_ft_weight,best_acc, model=model, feature_list=feature_list, adj_hat_list=adj_hat_list,
                             adj_hat=adj_hat_list[0], adj_wave=adj_wave_list, norm=norm_list[0], weight_tensor=weight_tensor_list[0],
                             optimizer_type=args.ft_optimizer, learning_rate=args.ft_lr, momentum=args.ft_momentum,
                             weight_decay=args.ft_weight_decay, num_epochs=args.ft_num_epochs, sp_weight=args.ft_sp_weight,
                             show_patience=args.ft_show_patience, update_interval=args.ft_update_interval,
                             cl_weight=args.ft_cl_weight, labels=labels, tolerance=args.ft_tolerance,
                             max_loss_patience=args.ft_loss_patience, save_patience=args.save_patience,
                             dataset_name=args.dataset_name,pl_weight=args.ft_pl_weight)
        if best_val_acc > best_acc:
            best_acc = best_val_acc
        ft_cost_time = time.time() - ft_begin_time
        ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), predicted)
        print('ACC, NMI, Purity, ARI, P, R, F1',ACC, NMI, Purity, ARI, P, R, F1)
        # nni.report_final_result(ACC)

        # pred_save_path = './data/pred/' + args.dataset_name + '.mat'
        # sio.savemat(pred_save_path, {'pred': predicted})

        all_ACC.append(ACC)
        all_NMI.append(NMI)
        all_Purity.append(Purity)
        all_ARI.append(ARI)
        all_P.append(P)
        all_R.append(R)
        all_F.append(F1)
        all_FT_TIME.append(ft_cost_time)
    nni.report_final_result(ACC)
    # append result to .txt file
    fp = open("results.txt", "a+", encoding="utf-8")
    # fp = open("results_" + args.dataset_name + ".txt", "a+", encoding="utf-8")
    fp.write("dataset_name: {}\n".format(args.dataset_name))
    fp.write("ft_sp_weight: {}\n".format(args.ft_sp_weight))
    fp.write("ft_cl_weight: {}\n".format(args.ft_cl_weight))
    fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("NMI: {:.2f}\t{:.2f}\n".format(np.mean(all_NMI) * 100, np.std(all_NMI) * 100))
    fp.write("Purity: {:.2f}\t{:.2f}\n".format(np.mean(all_Purity) * 100, np.std(all_Purity) * 100))
    fp.write("ARI: {:.2f}\t{:.2f}\n".format(np.mean(all_ARI) * 100, np.std(all_ARI) * 100))
    fp.write("P: {:.2f}\t{:.2f}\n".format(np.mean(all_P) * 100, np.std(all_P) * 100))
    fp.write("R: {:.2f}\t{:.2f}\n".format(np.mean(all_R) * 100, np.std(all_R) * 100))
    fp.write("F: {:.2f}\t{:.2f}\n".format(np.mean(all_F) * 100, np.std(all_F) * 100))
    # fp.write("Pretrain Time: {:.2f}\t{:.2f}\n".format(np.mean(all_PT_TIME), np.std(all_PT_TIME)))
    fp.write("Finetune Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_FT_TIME), np.std(all_FT_TIME)))
    fp.close()
'''
@Project: DFP-GNN
@File   : test_train
@Time   : 2021/9/12 15:10
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    main function
'''


