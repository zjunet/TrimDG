import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def preprocess_dgraph(dataset_name: str):
    """
    read the original data file and return the DataFrame that has columns ['u', 'i', 'ts', 'label', 'idx']
    :param dataset_name: str, dataset name
    :return:
    """

    u_list = data.edge_index[0].cpu().numpy().tolist()  # 源节点列表
    i_list = data.edge_index[1].cpu().numpy().tolist()  # 目标节点列表
    ts_list = data.edge_timestamp.cpu().numpy().tolist()  # 时间戳列表
    label_list = data.y.cpu().numpy().squeeze().tolist()  # 标签列表（需确认是否为边标签）
    idx_list = list(range(len(u_list)))  # 边索引列表
    edge_attribute = data.edge_attr.cpu().numpy().squeeze().tolist()
    # label_list = list(np.array(label_list)[np.array(u_list)])
    label_list = data.ye.cpu().numpy().tolist()
    assert all(np.diff(ts_list) >= 0), "时间戳必须非递减"
    return pd.DataFrame({'u': u_list, 'i': i_list, 'ts': ts_list, 'label': label_list, 'idx': idx_list}), np.array(
        edge_attribute).reshape(len(u_list), -1)

    # u_list, i_list, ts_list, label_list = [], [], [], []
    # feat_l = []
    # idx_list = []
    # with open(dataset_name) as f:
    #     s = next(f)
    #     previous_time = -1
    #     for idx, line in enumerate(f):
    #         e = line.strip().split(',')
    #         u = int(e[0])
    #         i = int(e[1])
    #         ts = float(e[2])
    #         assert ts >= previous_time
    #         previous_time = ts
    #         label = float(e[3])
    #         # edge features
    #         feat = np.array([float(x) for x in e[4:]])
    #         u_list.append(u)
    #         i_list.append(i)
    #         ts_list.append(ts)
    #         label_list.append(label)
    #         # edge index
    #         idx_list.append(idx)
    #         feat_l.append(feat)
    # return pd.DataFrame({'u': u_list, 'i': i_list, 'ts': ts_list, 'label': label_list, 'idx': idx_list}), np.array(
    #     feat_l)


def read_dgraphfin(folder):
    print('read_dgraphfin')
    names = ['dgraphfin.npz']
    items = [np.load(folder + '/' + name) for name in names]

    x = items[0]['x']
    y = items[0]['y'].reshape(-1, 1)
    edge_index = items[0]['edge_index']
    edge_type = items[0]['edge_type']
    edge_timestamp = items[0]['edge_timestamp']
    train_mask = items[0]['train_mask']
    valid_mask = items[0]['valid_mask']
    test_mask = items[0]['test_mask']

    x = torch.tensor(x, dtype=torch.float).contiguous()
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index.transpose(), dtype=torch.int64).contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.float)
    edge_timestamp = torch.tensor(edge_timestamp, dtype=torch.float)
    # train_mask = torch.tensor(train_mask, dtype=torch.int64)
    # valid_mask = torch.tensor(valid_mask, dtype=torch.int64)
    # test_mask = torch.tensor(test_mask, dtype=torch.int64)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type, y=y)
    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask
    data.edge_timestamp = edge_timestamp

    u_list = data.edge_index[0].cpu().numpy().tolist()
    label_list = data.y.cpu().numpy().squeeze().tolist()  # 标签列表（需确认是否为边标签）
    label_list = np.array(label_list)[np.array(u_list)]

    sorted_values, sorted_indices = torch.sort(data.edge_timestamp, dim=0, descending=False)
    data.edge_index = data.edge_index[:, sorted_indices]
    data.edge_attr = data.edge_attr[sorted_indices]
    data.edge_timestamp = data.edge_timestamp[sorted_indices]

    u_list = data.edge_index[0].cpu().numpy().tolist()
    label_list = data.y.cpu().numpy().squeeze().tolist()
    label_list = np.array(label_list)[np.array(u_list)]
    data.ye = torch.tensor(label_list, dtype=torch.int64).contiguous()

    def get_edge_mask(mask, label_list):
        src_nodes, dst_nodes = data.edge_index
        train_mask_new = torch.zeros(x.shape[0], dtype=torch.bool)
        train_mask_new[mask] = True
        train_edge_mask = train_mask_new[src_nodes] & train_mask_new[dst_nodes]
        train_edge_mask = train_edge_mask.numpy()
        return train_edge_mask

    train_mask_target = get_edge_mask(data.train_mask, label_list)
    val_mask_target = get_edge_mask(data.valid_mask, label_list)
    test_mask_target = get_edge_mask(data.test_mask, label_list)
    root = "/data/renhong/dyg"
    save_file_path1 = root + f'/processed_data/dgraph/train_mask_dgraph.npy'
    save_file_path2 = root + f'/processed_data/dgraph/val_mask_dgraph.npy'
    save_file_path3 = root + f'/processed_data/dgraph/test_mask_dgraph.npy'

    np.save(save_file_path1, train_mask_target)
    np.save(save_file_path2, val_mask_target)
    np.save(save_file_path3, test_mask_target)

    # train_mask_list = []
    # valid_mask_list = []
    # test_mask_list = []
    # for i in tqdm(range(edge_index.T.shape[0])):
    #     a, b = data.edge_index.T[i][0].item(), edge_index.T[i][1].item()
    #     if a in train_mask and b in train_mask:
    #         train_mask_list.append(i)
    #     elif a in valid_mask and b in valid_mask:
    #         valid_mask_list.append(i)
    #     elif a in test_mask and b in test_mask:
    #         test_mask_list.append(i)

    assert torch.all(torch.diff(data.edge_timestamp) >= 0), "时间戳未严格升序"

    return data


def reindex(df: pd.DataFrame, bipartite: bool = True):
    """
    reindex the ids of nodes and edges
    :param df: DataFrame
    :param bipartite: boolean, whether the graph is bipartite or not
    :return:
    """
    new_df = df.copy()
    if bipartite:
        # check the ids of users and items
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
        assert df.u.min() == df.i.min() == 0
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u
        new_df.i = new_i

    # make the id start from 1
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df


def preprocess_data(graph_data, dataset_name: str, bipartite: bool = True, node_feat_dim: int = 172):
    """
    preprocess the data
    :param dataset_name: str, dataset name
    :param bipartite: boolean, whether the graph is bipartite or not
    :param node_feat_dim: int, dimension of node features
    :return:
    """
    root = "/data/renhong/dyg"
    Path(root + "/processed_data/{}/".format(dataset_name)).mkdir(parents=True, exist_ok=True)
    PATH = root + '/DG_data/{}/{}.csv'.format(dataset_name, dataset_name)
    OUT_DF = root + '/processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    OUT_FEAT = root + '/processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    OUT_NODE_FEAT = root + '/processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    df, edge_feats = preprocess_dgraph(graph_data)
    new_df = reindex(df, bipartite)
    node_feats = graph_data.x.cpu().numpy()

    # edge feature for zero index, which is not used (since edge id starts from 1)
    empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]
    # Stack arrays in sequence vertically(row wise),
    edge_feats = np.vstack([empty, edge_feats])

    # node features with one additional feature for zero index (since node id starts from 1)
    max_idx = max(new_df.u.max(), new_df.i.max())
    empty_feats = np.zeros(node_feats.shape[1])[np.newaxis, :]
    node_feats = np.vstack([empty_feats, node_feats])
    # node_feats = np.zeros((max_idx + 1, node_feat_dim))

    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])

    # new_df.to_csv(PATH, index=False)  # 不保存行索引
    new_df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features


data = read_dgraphfin("/data/renhong/dyg/DG_data/")
preprocess_data(graph_data=data, dataset_name="dgraph", bipartite=False, node_feat_dim=172)
