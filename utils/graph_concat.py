
import torch

def all_fg_consstruct(features, adj_new,true_viewnum):
    view_graph = torch.eye(features[0].shape[0]).cuda()
    view_graph = torch.repeat_interleave(view_graph,repeats = true_viewnum, dim = 1)
    view_graph = torch.repeat_interleave(view_graph, repeats=true_viewnum, dim=0)

    for i in range(true_viewnum):
        if i == 0:
            adj_graph = adj_new[0]
        else:
            adj_graph = adjConcat(adj_graph, adj_new[i])
    # adj_new = torch.add(adj_graph , view_graph)
    # adj_new = torch.sub(adj_new-torch.eye(features[0][0].shape[0]*features.shape[1]))
    adj_new = adj_graph + view_graph
    adj_new = adj_new - torch.eye(features[0].shape[0] * true_viewnum).cuda()
    return adj_new


def adjConcat(a, b):
    # 将a,b两个矩阵沿对角线方向斜着合并，空余处补零[a,0.0,b]
    #     得到a和b的维度，先将a和b*a的零矩阵按行（竖着）合并得到c，再将a*b的零矩阵和b按行合并得到d
    #     将c和d横向合并
    #     '''
    lena = a.shape[0]  # len(a)
    lenb = b.shape[0]  # len(b)
    p = torch.zeros((lenb, lena)).cuda()
    q = torch.zeros((lena, lenb)).cuda()
    left = torch.vstack((a.to_dense(), p))  # 先将a和一个len(b)*len(a)的零矩阵垂直拼接，得到左半边
    right = torch.vstack((q, b.to_dense()))  # 再将一个len(a)*len(b)的零矩阵和b垂直拼接，得到右半边
    result = torch.hstack((left, right))  # 将左右矩阵水平拼接
    return result