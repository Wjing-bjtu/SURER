

import os

def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


if __name__ == "__main__":
    # 删除当前test目录下所有文件
    del_files("./data/adj_matrix/")
    del_files("./data/ec_feature/")
    del_files("./data/graph_new_weight/")
    del_files("./data/pt_weight/")
