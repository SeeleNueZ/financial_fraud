import numpy as np
import torch

from deploy.global_client import global_client

from sklearn.cluster import KMeans


class kmeans_global(global_client):
    def __init__(self, data, data_val, n_cluster):
        super().__init__(data=data, data_val=data_val, model=KMeans())
        self.cluster_center = None
        self.n_cluster = n_cluster
        self.dict = None

    @staticmethod
    def decode(number, b, t):
        res = []
        for _ in range(t):
            r = number % b
            res.append(r)
            number = (number - r) // b
        res.reverse()
        return res

    @staticmethod
    def encode(number_list, b, t):
        res = 0
        number_list.reverse()
        for i in range(t):
            res += number_list[i] * (b ** i)
        return res

    def kmeans_task1(self, num_client, num_dim, num_data, config):
        grids_number = self.n_cluster ** num_client
        center_list = np.zeros((grids_number, num_dim))
        center_weights = np.zeros(grids_number)

        for h in range(grids_number):
            # 给总计k^T种结果（排列）进行编号，构建k^T x dim这么大的矩阵，然后对K^T的每一种的结果进行填空
            h_decode = self.decode(h, self.n_cluster, num_client)
            temp = []
            for j in range(num_client):
                # 读取组合，得到该种结果的聚类中心排列
                temp.append(self.msg[j][1][h_decode[j], :])
            # 对应位置填入对应排列
            center_list[h, :] = np.concatenate(temp)
        # print(center_list)
        label_agg = []
        for key, indices in config.items():
            kmeans = KMeans(n_clusters=2, init=self.msg[key][1], n_init=1)
            kmeans.cluster_centers_=self.msg[key][1]

            label = kmeans.fit_predict(self.data[:, indices])
            label_agg.append(label)

        for i in range(num_data):
            temp = []
            # 提取 i 对应的那一条数值，比如0 0 1 1
            for j in range(num_client):
                temp.append(self.msg[j][0][i])
            # print(temp)
            # 折合成对应的编号
            idx = self.encode(temp, self.n_cluster, num_client)
            # 加入i对象的权重，默认是一样的
            center_weights[idx] += 1
        for i in range(self.data.shape[0]):
            temp = []
            # 提取 i 对应的那一条数值，比如0 0 1 1
            for j in range(num_client):
                temp.append(label_agg[j][i])
            # print(temp)
            # 折合成对应的编号
            idx = self.encode(temp, self.n_cluster, num_client)
            # 加入i对象的权重，默认是一样的
            center_weights[idx] += 1
        center_weights = center_weights / np.sum(center_weights)

        server_kmeans = KMeans(n_clusters=self.n_cluster)
        server_kmeans.fit(center_list, sample_weight=center_weights)
        self.model = server_kmeans
        self.cluster_center = server_kmeans.cluster_centers_
        j = 0
        for i in config:
            self.send(self.connected_clients[j], {self.c_id: self.cluster_center[:, config[i]]})
            j += 1
        return server_kmeans

    def kmeans_task2(self, num_client, num_data):
        dis = torch.zeros(self.msg[0].shape[0], self.n_cluster)
        for i in range(0, num_client):
            dis = self.msg[i] ** 2 + dis
            # print("dis", dis)
        dis = torch.sqrt(dis)
        # print(dis)
        return dis

    def get_center(self):
        return self.dict, self.cluster_center
